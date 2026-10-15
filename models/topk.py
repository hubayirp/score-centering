"""Exact distributed top-k collection of per-step sampling distributions.

Records every decode step's exact top-k head without materializing the full
[batch, seq, vocab] history. Steps accumulate in a [batch, CHUNK, vocab] ring
buffer, then one batched selection per chunk compresses them into the head
buffers. Total ring memory is CHUNK/seq of the full array.

Each vocabulary shard selects its exact local top-k; a second exact top-k over
the concatenated shard candidates gives the exact global result. The
transported payload remains [batch, seq, k].

Carry-based API for jitted decode loops:

    carry = init(batch_size, seq_len, vocab_size, k)
    carry = observe(carry, position, logprobs)     # every step
    head_logprobs, head_ids = finalize(carry, position, seq_len)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard

CHUNK = 32    # steps buffered between batched compressions


class Carry(NamedTuple):
    ring: jax.Array
    head_logprobs: jax.Array
    head_ids: jax.Array


def init(batch_size, seq_len, vocab_size, k):
    rows = P(("data", "model"), None, None)
    # Head buffers are padded to a chunk multiple so the final partial-chunk
    # compress can never be clamped (and misaligned) by dynamic_update_slice.
    padded = -(-seq_len // CHUNK) * CHUNK
    return Carry(ring=jnp.zeros((batch_size, CHUNK, vocab_size), jnp.float32, out_sharding=P("data", None, "model")),
                 head_logprobs=jnp.zeros((batch_size, padded, k), jnp.float32, out_sharding=rows),
                 head_ids=jnp.zeros((batch_size, padded, k), jnp.int32, out_sharding=rows))


def observe(carry, position, logprobs):
    ring = carry.ring.at[:, position % CHUNK].set(logprobs)
    head_logprobs, head_ids = jax.lax.cond(
        (position + 1) % CHUNK == 0,
        lambda: _compress(ring, carry.head_logprobs, carry.head_ids, position + 1 - CHUNK),
        lambda: (carry.head_logprobs, carry.head_ids))
    return Carry(ring, head_logprobs, head_ids)


def finalize(carry, position, seq_len):
    """Compress the final partial chunk and trim the padding. Rows past an
    early stop hold stale values; consumers mask them."""
    head_logprobs, head_ids = _compress(carry.ring, carry.head_logprobs, carry.head_ids, position // CHUNK * CHUNK)
    return head_logprobs[:, :seq_len], head_ids[:, :seq_len]


def _compress(ring, head_logprobs, head_ids, start):
    k = head_logprobs.shape[-1]

    def local_topk(block):
        values, ids = jax.lax.top_k(block, k, is_stable=False)
        return values, ids + jax.lax.axis_index("model") * block.shape[-1]

    spec = P("data", None, "model")
    values, ids = jax.shard_map(local_topk, in_specs=spec, out_specs=(spec, spec))(ring)
    rows = P(("data", "model"), None, None)
    values, ids = reshard(values, rows), reshard(ids, rows)
    top_logprobs, idx = jax.lax.top_k(values, k, is_stable=False)
    top_ids = jnp.take_along_axis(ids, idx, axis=-1)
    return (jax.lax.dynamic_update_slice(head_logprobs, top_logprobs, (0, start, 0)),
            jax.lax.dynamic_update_slice(head_ids, top_ids, (0, start, 0)))
