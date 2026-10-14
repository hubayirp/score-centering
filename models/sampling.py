"""Fixed-batch autoregressive sampling with a KV cache."""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard

from models import topk


class SamplingState(NamedTuple):
    position: int
    key: jax.Array
    tokens: jax.Array
    kv: list
    done: jax.Array
    token_logprobs: jax.Array
    vocab_logprobs: jax.Array | None
    head: topk.Carry | None


def _step(state, forward, weights, prompt_lengths, eos_id):
    key, sample_key = jax.random.split(state.key)
    logits, kv = forward(
        state.tokens[:, state.position, None],
        weights,
        state.kv,
        reshard(
            jnp.full((state.tokens.shape[0], 1), state.position),
            P("data", None),
        ),
        # Finished slots decode padding: mask them out of the MoE dispatch so
        # they cost no expert flops or gather traffic.
        live=~state.done[:, None],
    )
    logits = logits[:, 0].astype(jnp.float32)
    logprobs = jax.nn.log_softmax(logits)
    sampled = jax.random.categorical(sample_key, logits)
    vocab_logprobs = state.vocab_logprobs
    if vocab_logprobs is not None:
        vocab_logprobs = vocab_logprobs.at[:, state.position].set(
            logprobs.astype(vocab_logprobs.dtype))
    head = state.head
    if head is not None:
        head = topk.observe(head, state.position, logprobs)

    buffered_token = state.tokens[:, state.position + 1]
    sampling = (~state.done) & (state.position + 1 >= prompt_lengths)
    next_token = jnp.where(sampling, sampled, buffered_token)
    tokens = state.tokens.at[:, state.position + 1].set(next_token)
    targets = reshard(
        jax.nn.one_hot(next_token, logits.shape[-1], dtype=jnp.float32),
        jax.typeof(logprobs).sharding.spec,
    )
    token_logprobs = state.token_logprobs.at[:, state.position].set(
        jnp.where(state.done, 0.0, (targets * logprobs).sum(-1)))
    done = state.done | (sampling & (sampled == eos_id))
    return SamplingState(
        state.position + 1, key, tokens, kv, done,
        token_logprobs, vocab_logprobs, head)


@partial(jax.jit, static_argnames=(
    "forward", "init_kv", "vocab_size", "logprobs_dtype", "logprobs_topk"))
def _generate(key, forward, init_kv, weights, tokens,
              prompt_lengths, eos_id, vocab_size, logprobs_dtype,
              logprobs_topk):
    batch_size, seq_len = tokens.shape
    state = SamplingState(
        position=0,
        key=key,
        tokens=reshard(tokens, P("data", None)),
        kv=init_kv(batch_size, seq_len),
        done=jnp.zeros(batch_size, bool, out_sharding=P("data")),
        token_logprobs=jnp.zeros(
            (batch_size, seq_len), jnp.float32,
            out_sharding=P("data", None)),
        vocab_logprobs=(
            None if logprobs_dtype is None else jnp.zeros(
                (batch_size, seq_len, vocab_size),
                logprobs_dtype,
                out_sharding=P("data", None, "model"),
            )
        ),
        head=(
            topk.init(batch_size, seq_len, vocab_size, logprobs_topk)
            if logprobs_topk else None
        ),
    )
    state = jax.lax.while_loop(
        lambda state: (state.position + 1 < seq_len) & (~state.done).any(),
        lambda state: _step(
            state, forward, weights, prompt_lengths, eos_id),
        state,
    )
    head_logprobs, head_ids = (
        topk.finalize(state.head, state.position, seq_len)
        if logprobs_topk else (None, None))
    return (state.tokens, state.token_logprobs, state.vocab_logprobs,
            head_logprobs, head_ids)


def generate(key, model, weights, forward, init_kv, tokens, prompt_lengths,
             logprobs_dtype=None, logprobs_topk=0):
    """Fill a fixed token batch after each row's prompt and record the sampling
    distribution: per-token logprobs always, plus either the full [B, T, V]
    logprobs (logprobs_dtype) or the per-step top-k head (logprobs_topk), which
    is what score centering consumes at ~V/k the memory."""
    return _generate(
        reshard(key, P()), forward, init_kv, weights, tokens,
        reshard(jnp.asarray(prompt_lengths, jnp.int32), P("data")),
        model.tokenizer.eos_token_id, model.config["vocab_size"],
        logprobs_dtype, logprobs_topk)
