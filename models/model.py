import json
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from jax.sharding import AxisType, PartitionSpec as P, reshard
from safetensors import safe_open
from safetensors.numpy import save_file
from transformers import AutoTokenizer

from models import quant
from models.quant import qeinsum


@dataclass
class Model:
    weights: dict[str, Any]
    forward: Callable
    head: Callable
    init_kv: Callable
    tokenizer: Any
    config: dict[str, Any]


def apply_rope(x, cfg, pos=0):
    B, T, N, H = x.shape
    positions = pos + jnp.broadcast_to(jnp.arange(T)[None, :], [B, T])
    freq = 1.0 / (cfg["rope_theta"] ** (jnp.arange(0, H, 2, dtype=jnp.float32) / H))
    if rs := cfg.get("rope_scaling"):  # llama3 NTK-by-parts (type validated at load)
        rots = rs["original_max_position_embeddings"] * freq / (2 * jnp.pi)  # rotations over the original context
        smooth = jnp.clip((rots - rs["low_freq_factor"]) / (rs["high_freq_factor"] - rs["low_freq_factor"]), 0, 1)
        freq = (1 - smooth) * freq / rs["factor"] + smooth * freq
    inp = jnp.einsum("bt,h->bth", positions, freq, precision=jax.lax.Precision.HIGHEST)
    sin, cos = jnp.sin(inp).astype(x.dtype), jnp.cos(inp).astype(x.dtype)
    x1, x2 = x[:, :, :, :H // 2], x[:, :, :, H // 2:]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def rms_norm(x, gamma, eps, axis=-1):
    rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=axis, keepdims=True) + eps)
    return (gamma * x / rms).astype(x.dtype)


def forward_moe(cfg, x, w, live=None):
    E, K = cfg["num_experts"], cfg["num_experts_per_tok"]
    if live is None:  # callers without a mask dispatch every row
        live = jnp.ones(x.shape[:2], dtype=bool, out_sharding=P("data", None))
    logits = jnp.einsum("btd,ed->bte", x, w["mlp.gate.weight"], preferred_element_type=jnp.float32, out_sharding=P("data", None, None))
    probs, idx = jax.lax.top_k(jax.nn.softmax(logits, -1), K)
    probs = (probs / probs.sum(-1, keepdims=True)).astype(x.dtype)

    is_q = lambda v: isinstance(v, quant.QArray)

    def experts(x, probs, idx, live, wg, wu, wd):
        shape = x.shape  # this chip's batch shard; tokens are exchanged, expert weights stay put
        x, probs, idx, live = (jax.lax.all_gather(v, "data", axis=0, tiled=True) for v in (x, probs, idx, live))
        S, D = x.shape[0] * x.shape[1], x.shape[-1]
        El = (wg.q if is_q(wg) else wg).shape[0]  # experts on this chip
        eid = idx.reshape(-1) - jax.lax.axis_index(("data", "model")) * El
        # Dead (padding) rows never claim a slot in the gathered-row buffer.
        local = (eid >= 0) & (eid < El) & jnp.repeat(live.reshape(-1), K)
        order = jnp.argsort(jnp.where(local, eid, El))  # rows for remote experts sort last
        # Row buffer 2x the mean chip load: keeps every live local row unless a
        # chip is >2x overloaded, and drops most of the remote rows' gather
        # traffic. The surviving remote rows run through the last local expert
        # (rather than falling outside group_sizes, whose gradients are
        # undefined) and their outputs are masked below.
        order = order[:min(S * K, max(2 * S * K * El // E, 128))]
        sel = jnp.where(local, eid, El)[order]
        rows, gid = x.reshape(S, D)[order // K], jnp.minimum(sel, El - 1)
        sizes = jnp.bincount(gid, length=El).astype(jnp.int32)
        h = jax.nn.silu(quant.qragged_dot(rows, wg, sizes, gid)) * quant.qragged_dot(rows, wu, sizes, gid)
        y = jnp.where((sel < El)[:, None], quant.qragged_dot(h, wd, sizes, gid), 0)
        y = jnp.zeros((S, D), x.dtype).at[order // K].add(y * probs.reshape(-1)[order][:, None])
        y = jax.lax.psum_scatter(y, "data", scatter_dimension=0, tiled=True)
        return jax.lax.psum(y, "model").reshape(shape)

    s = P(("data", "model"), None, None)  # every chip holds E / n_chips full-width experts
    wg, wu, wd = (w[f"mlp.experts.{p}_proj.weight"] for p in ("gate", "up", "down"))
    wg, wu, wd = (quant.QArray(reshard(v.q, s), reshard(v.scale, P(("data", "model"), None))) if is_q(v)
                  else reshard(v, s) for v in (wg, wu, wd))
    spec = lambda v: quant.QArray(s, P(("data", "model"), None)) if is_q(v) else s
    # Ties the expert weight grads into the backward's critical path; otherwise
    # the scheduler sinks all layers' weight-grad dots to the end of the program
    # (their only consumer), keeping every layer's dispatch buffers live.
    x, wg, wu, wd = jax.lax.optimization_barrier((x, wg, wu, wd))
    # check_vma=False: tokamax.ragged_dot's custom VJP does not annotate
    # manual axes and trips the varying-mesh-axes validation.
    return jax.shard_map(experts, in_specs=(P("data", None, None),) * 3 + (P("data", None), spec(wg), spec(wu), spec(wd)),
                         out_specs=P("data", None, None), check_vma=False)(x, probs, idx, live, wg, wu, wd)


def attention(q, k, v, mask=None, *, is_causal=False):
    # Attention is independent across batch rows and KV-head groups. Run it
    # locally so JAX's internal GQA reshape need not infer explicit sharding.
    # This also leaves the causal cuDNN path available on GPUs.
    heads = P("data", None, "model", None)
    attend = lambda q, k, v, mask: jax.nn.dot_product_attention(q, k, v, mask=mask, is_causal=is_causal)
    mask_spec = P("data", None, None, None) if mask is not None else None
    return jax.shard_map(attend, in_specs=(heads, heads, heads, mask_spec), out_specs=heads)(q, k, v, mask)


def forward_layer(cfg, x, w, kv=None, pos=0, live=None, kv_quant_bits=0):
    x_norm = rms_norm(x, w["input_layernorm.weight"], cfg["rms_norm_eps"])

    q = qeinsum("btd,nhd->btnh", x_norm, w["self_attn.q_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model", None))
    k = qeinsum("bsd,khd->bskh", x_norm, w["self_attn.k_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model", None))
    v = qeinsum("bsd,khd->bskh", x_norm, w["self_attn.v_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model", None))

    if "self_attn.q_proj.bias" in w:
        q += w["self_attn.q_proj.bias"]
        k += w["self_attn.k_proj.bias"]
        v += w["self_attn.v_proj.bias"]

    if "self_attn.q_norm.weight" in w:
        q = rms_norm(q, w["self_attn.q_norm.weight"], cfg["rms_norm_eps"])
        k = rms_norm(k, w["self_attn.k_norm.weight"], cfg["rms_norm_eps"])

    q = apply_rope(q, cfg, pos)
    k = apply_rope(k, cfg, pos)

    if kv is not None:  # decode: one token per row, pos is per-slot [B, 1]
        upd = jnp.stack([k, v])
        if kv_quant_bits:  # int-N / fp8 / fp4 — see quant.kv_fake_quantize
            upd = quant.kv_fake_quantize(upd, kv_quant_bits)
        kv = jax.vmap(lambda c, u, p: jax.lax.dynamic_update_slice(c, u, (0, p, 0, 0)),
                      in_axes=(1, 1, 0), out_axes=1)(kv, upd, pos[:, 0])
        k, v = kv
        attn_mask = (jnp.arange(k.shape[1])[None, None, :] <= pos[:, :, None])[:, None]
        attn_out = attention(q, k, v, mask=attn_mask)
    else:  # training: is_causal (not a mask array) keeps cuDNN flash attention
        # eligible; the XLA fallback materializes [B, N, T, T] f32 logits.
        attn_out = attention(q, k, v, is_causal=True)
    x += qeinsum("btnh,dnh->btd", attn_out, w["self_attn.o_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, None))

    x_norm = rms_norm(x, w["post_attention_layernorm.weight"], cfg["rms_norm_eps"])
    if "mlp.gate.weight" in w:
        x += forward_moe(cfg, x_norm, w, live)
    else:
        gate = jax.nn.silu(qeinsum("btd,fd->btf", x_norm, w["mlp.gate_proj.weight"], preferred_element_type=jnp.float32, out_sharding=P("data", None, "model")))
        up = qeinsum("btd,fd->btf", x_norm, w["mlp.up_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model"))
        x += qeinsum("btf,df->btd", gate * up, w["mlp.down_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, None))
    return x, kv


def head(cfg, x, weights, dtype=jnp.bfloat16):
    is_q = lambda w: isinstance(w, quant.QArray)
    out_embed = weights["model.embed_tokens.weight"] if cfg["tie_word_embeddings"] else weights["lm_head.weight"]
    if not is_q(out_embed): out_embed = out_embed.astype(dtype)
    return qeinsum("btd,vd->btv", x, out_embed, preferred_element_type=x.dtype, out_sharding=P("data", None, "model"))


def forward(cfg, x, weights, kv=None, pos=0, dtype=jnp.bfloat16, logits=True, live=None, kv_quant_bits=0):
    is_q = lambda w: isinstance(w, quant.QArray)
    # Compute cast per layer, inside the remat region: casting the whole
    # tree up front keeps every layer's compute copy live into the backward and
    # blocks optimizer-backward fusion (fp32 masters: 33.7G -> 13.9G temps).
    cast = lambda w: w.astype(dtype) if not is_q(w) and w.ndim > 1 else w
    x = reshard(x, P("data", None))
    x = quant.take(cast(weights["model.embed_tokens.weight"]), x, out_sharding=P("data", None, None)).astype(dtype)

    # Layer weights are stacked [L, ...]; scanning keeps the compiled program
    # one layer deep, so each layer's remat/backward buffers are reused
    # instead of packed side by side (unrolled: 62 GiB of GPU temps at B128).
    layers = {k.removeprefix("model.layers."): v for k, v in weights.items() if k.startswith("model.layers.")}

    def layer_fn(x, layer):
        lw, kv_i = layer
        return forward_layer(cfg, x, jax.tree.map(cast, lw, is_leaf=is_q), kv_i, pos, live, kv_quant_bits)

    return_kv = kv is not None
    if kv is None:
        x, kv = jax.lax.scan(jax.remat(layer_fn), x, (layers, kv))
    else:
        # Decode: unrolled layers, each updating its own cache entry in
        # place. The scan path would rebuild a stacked cache (ys) every
        # token and nest a while loop inside the decode loop.
        kv = list(kv)
        for i in range(len(kv)):
            x, kv[i] = layer_fn(x, (jax.tree.map(lambda w: w[i], layers), kv[i]))

    x = rms_norm(x, weights["model.norm.weight"], cfg["rms_norm_eps"])
    if logits:
        x = head(cfg, x, weights, dtype)
    return (x, kv) if return_kv else x


def init_kv(L, K, H, B, T, dtype=jnp.bfloat16):
    # One cache array per layer: a stacked cache would be rewritten whole on
    # every decoded token instead of aliased in the decode loop carry.
    return [jnp.zeros((2, B, T, K, H), dtype, out_sharding=P(None, "data", None, "model", None)) for _ in range(L)]


def _head_dim(cfg):
    return cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])


def _download(source, weights_dir, weights=True):
    path = Path(source).expanduser()
    if not path.exists():
        path = Path(weights_dir).expanduser() / source
        snapshot_download(repo_id=source, local_dir=path, allow_patterns=None if weights else ["*.json", "*.jinja"])
    cfg = json.loads((path / "config.json").read_text())
    assert cfg["model_type"] in ("llama", "qwen2", "qwen3", "qwen3_moe"), cfg["model_type"]
    assert not cfg.get("use_sliding_window"), "sliding-window attention is not implemented"
    assert cfg.get("rope_scaling") is None or cfg["rope_scaling"]["rope_type"] == "llama3"
    tokenizer = AutoTokenizer.from_pretrained(path)
    # Llama 3 tokenizers set no pad token. 128004 is a reserved special token (3.1+ names it <|finetune_right_pad_id|>).
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 128004
    return path, tokenizer, cfg


def _get_sharding(key, fsdp=False):
    """Per-layer weights are stacked [L, ...]: their specs gain a leading None."""
    if "norm" in key or "bias" in key or key.endswith("mlp.gate.weight"): return P()
    if "experts" in key:  # stacked (E, D, F) gate/up and (E, F, D) down: always fully expert-parallel
        spec = P(("data", "model"), None, None)
    elif any(k in key for k in ("q_proj", "k_proj", "v_proj")):
        spec = P("model", None, "data") if fsdp else P("model", None, None)
    elif "o_proj" in key:
        spec = P("data", "model", None) if fsdp else P(None, "model", None)
    elif any(k in key for k in ("gate_proj", "up_proj", "embed_tokens", "lm_head")):
        spec = P("model", "data") if fsdp else P("model")
    elif "down_proj" in key:
        spec = P("data", "model") if fsdp else P(None, "model")
    else:
        raise ValueError(f"Unrecognized key: {key}")
    return P(None, *spec) if key.startswith("model.layers.") else spec


def _load_weights(model_ckpt_dir, cfg, fsdp=False):
    N, K, D, H = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"], _head_dim(cfg)
    weights, experts, layered = {}, defaultdict(dict), defaultdict(dict)

    def add(key, value):  # model.layers.{i}.{name} collects into stacked [L, ...]
        if key.startswith("model.layers."):
            index, _, name = key.removeprefix("model.layers.").partition(".")
            layered[f"model.layers.{name}"][int(index)] = value
        else:
            weights[key] = jax.device_put(value, _get_sharding(key, fsdp))

    for file in sorted(model_ckpt_dir.glob("*.safetensors")):
        with safe_open(file, framework="numpy") as f:
            for key in f.keys():
                value = f.get_tensor(key)
                if ".mlp.experts." in key:  # mlp.experts.{i}.{proj}.weight, stacked below
                    layer, _, rest = key.partition(".mlp.experts.")
                    experts[f"{layer}.mlp.experts.{rest.partition('.')[2]}"][int(rest.partition(".")[0])] = value
                    continue
                if "q_proj.bias" in key: value = value.reshape([N, H])
                if "k_proj.bias" in key or "v_proj.bias" in key: value = value.reshape([K, H])
                if "q_proj.weight" in key: value = value.reshape([N, H, D])
                if "k_proj.weight" in key or "v_proj.weight" in key: value = value.reshape([K, H, D])
                if "o_proj.weight" in key: value = value.reshape([D, N, H])
                add(key, value)
    for key in list(experts):  # (E, out, in) -> (E, in, out): the expert dot contracts the middle axis
        parts = experts.pop(key)
        add(key, np.stack([parts[e] for e in range(cfg["num_experts"])]).swapaxes(1, 2))
    for key, parts in layered.items():
        value = np.stack([parts[i] for i in range(cfg["num_hidden_layers"])])
        weights[key] = jax.device_put(value, _get_sharding(key, fsdp))
    if not weights:
        raise ValueError(f"No safetensors files found in {model_ckpt_dir}")
    return weights


def _random_weights(cfg, key, fsdp=False):
    """N(0, initializer_range) matrices, unit norms and zero biases, laid out as _load_weights lays them out."""
    D, N, K, H = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], _head_dim(cfg)
    layer = {"input_layernorm.weight": (D,), "post_attention_layernorm.weight": (D,),
             "self_attn.q_proj.weight": (N, H, D), "self_attn.k_proj.weight": (K, H, D),
             "self_attn.v_proj.weight": (K, H, D), "self_attn.o_proj.weight": (D, N, H)}
    if cfg["model_type"] == "qwen2" or cfg.get("attention_bias"):
        layer |= {"self_attn.q_proj.bias": (N, H), "self_attn.k_proj.bias": (K, H), "self_attn.v_proj.bias": (K, H)}
    if cfg["model_type"] in ("qwen3", "qwen3_moe"):
        layer |= {"self_attn.q_norm.weight": (H,), "self_attn.k_norm.weight": (H,)}
    if "num_experts" in cfg:
        E, F = cfg["num_experts"], cfg["moe_intermediate_size"]
        layer |= {"mlp.gate.weight": (E, D), "mlp.experts.gate_proj.weight": (E, D, F),
                  "mlp.experts.up_proj.weight": (E, D, F), "mlp.experts.down_proj.weight": (E, F, D)}
    else:
        F = cfg["intermediate_size"]
        layer |= {"mlp.gate_proj.weight": (F, D), "mlp.up_proj.weight": (F, D), "mlp.down_proj.weight": (D, F)}
    shapes = {"model.embed_tokens.weight": (cfg["vocab_size"], D), "model.norm.weight": (D,)}
    if not cfg.get("tie_word_embeddings"):
        shapes["lm_head.weight"] = (cfg["vocab_size"], D)
    shapes |= {f"model.layers.{name}": (cfg["num_hidden_layers"], *shape) for name, shape in layer.items()}

    def init(name, shape, key):
        spec = _get_sharding(name, fsdp)
        if "norm" in name or "bias" in name:
            return (jnp.ones if "norm" in name else jnp.zeros)(shape, jnp.bfloat16, out_sharding=spec)
        return (cfg["initializer_range"] * jax.random.normal(key, shape, out_sharding=spec)).astype(jnp.bfloat16)

    keys = jax.random.split(key, len(shapes))
    return {name: init(name, shape, k) for (name, shape), k in zip(shapes.items(), keys)}


def load(source, weights_dir="~/.cache/postax/weights", tp=1, fsdp=True, init="pretrained", overrides=None, seed=0):
    """Load a Llama / Qwen2 / Qwen3 / Qwen3-MoE checkpoint (local path or HF hub
    id) onto a (data, model) mesh: tp-way tensor parallel, FSDP over data.
    init="random" takes only the config (plus any overrides) and tokenizer from source."""
    assert init in ("pretrained", "random") and (init == "random" or not overrides), (init, overrides)
    jax.set_mesh(jax.make_mesh((jax.device_count() // tp, tp), ("data", "model"),
                               axis_types=(AxisType.Explicit, AxisType.Explicit)))
    path, tokenizer, cfg = _download(source, weights_dir, weights=init == "pretrained")
    if init == "random":  # drop layer_types: HF derives it from num_hidden_layers when the model is reloaded
        cfg = {k: v for k, v in cfg.items() if k != "layer_types"} | dict(overrides or {})
        weights = _random_weights(cfg, jax.random.key(seed), fsdp)
    else:
        weights = _load_weights(path, cfg, fsdp)
    return Model(weights=weights, forward=partial(forward, cfg), head=partial(head, cfg),
                 init_kv=partial(init_kv, cfg["num_hidden_layers"], cfg["num_key_value_heads"], _head_dim(cfg)),
                 tokenizer=tokenizer, config=cfg)


def save(model, output_dir):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    hf = {}
    for k, stacked in model.weights.items():
        stacked = np.asarray(jax.device_get(jnp.asarray(stacked, dtype=jnp.bfloat16)))
        layered = k.startswith("model.layers.")
        name = k.removeprefix("model.layers.")
        for i, v in enumerate(stacked if layered else [stacked]):
            k = f"model.layers.{i}.{name}" if layered else name
            if ".mlp.experts." in k:  # unstack (E, in, out) back to per-expert HF (out, in)
                layer, _, proj = k.partition(".mlp.experts.")
                for e in range(v.shape[0]):
                    hf[f"{layer}.mlp.experts.{e}.{proj}"] = np.ascontiguousarray(v[e].T)  # safetensors ignores strides
                continue
            if "q_proj" in k or "k_proj" in k or "v_proj" in k: v = v.reshape(-1, v.shape[-1]) if v.ndim == 3 else v.reshape(-1)
            if "o_proj.weight" in k: v = v.reshape(v.shape[0], -1)
            hf[k] = v

    save_file(hf, str(output_dir / "model.safetensors"))
    (output_dir / "config.json").write_text(json.dumps(model.config, indent=2) + "\n")
    model.tokenizer.save_pretrained(output_dir)
