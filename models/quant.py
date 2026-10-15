"""Sampler quantization: 8-bit (int8 / fp8) matmuls with per-channel and
per-token scales, and fake quantization of KV-cache writes."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import tokamax
from jax.sharding import PartitionSpec as P, reshard


class QArray(NamedTuple):
    """8-bit tensor plus one f32 scale per block (= all non-contracted axes)."""
    q: jax.Array      # same shape as the original tensor
    scale: jax.Array  # q.shape minus the trailing contraction axes


def quantize(x, dtype, contract_dims=1):
    """Symmetric absmax quantization over the trailing contract_dims axes."""
    qmax = float((jnp.iinfo(dtype) if jnp.issubdtype(dtype, jnp.integer) else jnp.finfo(dtype)).max)
    axes = tuple(range(x.ndim - contract_dims, x.ndim))
    scale = jnp.maximum(jnp.abs(x).max(axes).astype(jnp.float32), 1e-30) / qmax
    q = x / scale[(...,) + (None,) * contract_dims]
    if jnp.issubdtype(dtype, jnp.integer):
        q = jnp.round(q)
    return QArray(jnp.clip(q, -qmax, qmax).astype(dtype), scale)


def quantize_tree(weights, dtype):
    """Quantize every matmul weight; norm gains, biases and MoE routers stay
    untouched. Scales replicate (they broadcast against any output sharding
    for free), except stacked expert scales, consumed inside the MoE
    shard_map alongside their sharded weights."""
    def quantize_weight(k, w):
        if "experts" in k:  # stacked experts contract their middle axis: quantize swapped
            q, scale = quantize(w.swapaxes(-1, -2), dtype)
            return QArray(q.swapaxes(-1, -2), scale)
        q, scale = quantize(w, dtype, contract_dims=2 if "o_proj" in k else 1)
        return QArray(q, reshard(scale, P()))

    keep = lambda k, w: w.ndim <= 1 or "norm" in k or "bias" in k or k.endswith("mlp.gate.weight")
    return {k: w if keep(k, w) else quantize_weight(k, w) for k, w in weights.items()}


# Mini-float formats for _snap_float: (mantissa bits, max value, min normal
# exponent). Grid-snapped explicitly rather than via astype round-trips,
# which XLA's GPU simplifier folds away entirely (verified: convert
# bf16->fp8->f32 under jit is the identity — a silent no-op quantizer).
_FLOAT_GRIDS = {jnp.float8_e4m3fn: (3, 448.0, -6), jnp.float4_e2m1fn: (1, 6.0, 0)}


def _snap_float(x, dtype):
    """Round x onto a mini-float grid (round-half-even, subnormals, saturate)."""
    man_bits, max_val, min_normal_exp = _FLOAT_GRIDS[dtype]
    ax = jnp.abs(x)
    exp = jnp.floor(jnp.log2(jnp.maximum(ax, 1e-30)))
    quantum = jnp.exp2(jnp.maximum(exp, min_normal_exp) - man_bits)
    snapped = jnp.round(ax / quantum) * quantum
    return jnp.sign(x) * jnp.minimum(snapped, max_val)


def kv_fake_quantize(upd, spec):
    """Fake quant of KV-cache writes (cfg.sampler.kv_quant_bits). spec is an
    int bit-width (symmetric per-token-per-head grid, vLLM's
    intN_per_token_head), "fp8" (e4m3 at unit scale, vLLM's kv_cache_dtype=fp8
    default), or "fp4" (e2m1 with per-token-per-head absmax scaling). Snapped
    values enter the cache, so the error compounds into every later step."""
    u = upd.astype(jnp.float32)
    if spec == "fp8":
        return _snap_float(u, jnp.float8_e4m3fn).astype(upd.dtype)
    if spec == "fp4":
        scale = jnp.maximum(jnp.abs(u).max(-1, keepdims=True) / 6.0, 1e-9)
        return (_snap_float(u / scale, jnp.float4_e2m1fn) * scale).astype(upd.dtype)
    qmax = 2 ** (spec - 1) - 1
    scale = jnp.maximum(jnp.abs(u).max(-1, keepdims=True) / qmax, 1e-9)
    return (jnp.round(u / scale) * scale).astype(upd.dtype)


def qeinsum(eq, x, w, preferred_element_type=None, out_sharding=None):
    """jnp.einsum that runs the contraction in 8 bit when w is a QArray;
    contraction axes must be trailing in both operands."""
    if not isinstance(w, QArray):
        return jnp.einsum(eq, x, w, preferred_element_type=preferred_element_type, out_sharding=out_sharding)
    contract_dims = w.q.ndim - w.scale.ndim
    lhs, rhs = eq.split("->")[0].split(",")
    assert lhs[-contract_dims:] == rhs[-contract_dims:], f"contraction not trailing in {eq}"
    xq = quantize(x, w.q.dtype, contract_dims)
    acc_type = jnp.int32 if jnp.issubdtype(w.q.dtype, jnp.integer) else jnp.float32
    y = jnp.einsum(eq, xq.q, w.q, preferred_element_type=acc_type, out_sharding=out_sharding)
    y = y * xq.scale[(...,) + (None,) * w.scale.ndim] * w.scale
    return y.astype(x.dtype if preferred_element_type is None else preferred_element_type)


def qragged_dot(x, w, group_sizes, expert_ids):
    """jax.lax.ragged_dot that runs the contraction in 8 bit when w is a QArray.

    Rows of x are sorted by expert (the ragged_dot layout); expert_ids gives each
    row's expert so the per-expert weight scales can rescale the result. Runs
    inside the MoE shard_map, so accumulation is purely local (int32/f32 for
    free) and the caller's explicit psum keeps the wire 16-bit regardless.
    """
    # implementation pinned: tokamax 0.0.13's default Mosaic-GPU kernel
    # exceeds sm_100 shared memory (263KB > 227KB); triton is exact and
    # equally fast at our shapes on both sm_90 and sm_100. The xla fallback
    # covers CPU (tests), where the triton backend raises NotImplementedError.
    if not isinstance(w, QArray):
        # Tokamax 0.0.13's Triton dRHS kernel executes its final contraction
        # with iteration -1 when a group has no rows.  Depending on adjacent
        # memory, that gives a spurious finite gradient, inf, or NaN for the
        # unused expert: https://github.com/openxla/tokamax/issues/887
        # Preserve the exact forward value while disconnecting empty expert
        # weights from autodiff; the select in this VJP masks even non-finite
        # cotangents produced by the kernel's empty-group tile.
        active = (group_sizes > 0)[:, None, None]
        w = jnp.where(active, w, jax.lax.stop_gradient(w))
        return tokamax.ragged_dot(x, w, group_sizes, implementation=("triton", "xla"))
    xq = quantize(x, w.q.dtype, 1)
    # tokamax's grouped-matmul kernels (fwd + both grads) replace both the
    # masked-dense XLA lowering (whose [El, rows, width] f32 intermediates
    # OOM 80GB chips at 30B) and jax's triton kernel (absent from every
    # released wheel; bf16-only on sm_100). 8-bit grids are applied above and
    # are exact in bf16/f32, so contracting dequantized values keeps W8A8
    # numerics up to accumulator rounding.
    y = tokamax.ragged_dot(xq.q.astype(jnp.bfloat16), w.q.astype(jnp.bfloat16), group_sizes,
                           preferred_element_type=jnp.float32, implementation=("triton", "xla"))
    return (y * xq.scale[:, None] * w.scale[expert_ids]).astype(x.dtype)


def take(w, idx, out_sharding=None):
    """Row gather (embedding lookup) from a possibly quantized table."""
    if not isinstance(w, QArray):
        return w.at[idx, :].get(out_sharding=out_sharding)
    rows = w.q.at[idx, :].get(out_sharding=out_sharding)
    scale = w.scale.at[idx].get(out_sharding=jax.typeof(idx).sharding.spec)
    return rows.astype(jnp.bfloat16) * scale[..., None].astype(jnp.bfloat16)
