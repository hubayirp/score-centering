"""Score centering cancels drift: with a constant advantage, the expected
(corrected) REINFORCE gradient under a mismatched sampler q is zero.
Run from the repository root: python -m pytest tests"""
import jax
import jax.numpy as jnp
import pytest

import losses
from losses import ISCorrection

TRAINER_LOGITS = jnp.array([0.5, -0.2, 1.1])
SAMPLER_LOGP = jax.nn.log_softmax(TRAINER_LOGITS + jnp.array([0.1, 0.3, -0.2]))
CORRECTIONS = {
    "pg": None,
    "tis": ISCorrection(high=2.0, outside="clamp"),
    "mis": ISCorrection(low=0.8, high=1.1),  # this band drops some of the tokens
}


def loss(logits, token, correction, center):
    """Loss of one sampled token (the second position of a two-token row)."""
    row = lambda x: jnp.stack([x, jnp.zeros_like(x)])[None]
    batch = {"tokens": jnp.array([[0, token]]), "mask": jnp.array([[False, True]]),
             "advantages": jnp.array([1.0]),
             "sampling_token_logprobs": jnp.array([[SAMPLER_LOGP[token], 0.0]])}
    if center == "full":
        batch["sampling_vocab_logprobs"] = row(SAMPLER_LOGP)
    if center == "topk":  # k = V - 1: the tail is one token, so modeling it as rho * p is exact
        head_logp, head_ids = jax.lax.top_k(SAMPLER_LOGP, 2)
        batch |= {"center_logprobs": row(head_logp), "center_ids": row(head_ids)}
    hidden = jnp.broadcast_to(logits, (1, 2, logits.size))
    return losses.loss_fn_rl(hidden, lambda x: x, batch, correction, center is not None)[0]


def expected_gradient(correction, center):
    tokens = jnp.arange(TRAINER_LOGITS.size)
    grads = jax.vmap(lambda t: jax.grad(loss)(TRAINER_LOGITS, t, correction, center))(tokens)
    return (jnp.exp(SAMPLER_LOGP)[:, None] * grads).sum(0)


@pytest.mark.parametrize("center", ["full", "topk"])
@pytest.mark.parametrize("name", CORRECTIONS)
def test_score_centering_cancels_drift(name, center):
    assert jnp.abs(expected_gradient(CORRECTIONS[name], center)).max() < 1e-6


@pytest.mark.parametrize("name", CORRECTIONS)
def test_drift_without_centering(name):
    assert jnp.abs(expected_gradient(CORRECTIONS[name], None)).max() > 1e-3
