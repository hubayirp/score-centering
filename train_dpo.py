"""Preference optimization on chosen/rejected chat pairs: DPO, and with
dpo.ref_weight=0 and dpo.norm=true the reference-free SimPO."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from tqdm.auto import tqdm

import lm_utils
import models
import utils


DEFAULT_CFG = """
model:
  source: Qwen/Qwen3-0.6B
  weights_dir: ~/.cache/postax/weights
  tp: 1
data:
  path: allenai/olmo-2-0425-1b-preference-mix
  name: null
  split: train
  format: preference
  seq_len: 2048
  batch_size: 16
  valid_rows: 512
dpo:
  beta: 0.1
  ref_weight: 1.0  # 0 = reference-free
  norm: false      # length-normalize the sequence logprobs
  margin: 0.0
  sft_weight: 0.0  # add this times the NTP loss on chosen responses
opt:
  optimizer: adamw
  lr: 1e-6
  epochs: 1
log:
  project: postax-dpo
  wandb_mode: online
  dir: ./outputs
  run_name: null
  every_steps: 10
  valid_every_steps: 200
  save_model: false
run:
  seed: 0
"""


def train(cfg):
    model = models.load(cfg.model.source, cfg.model.weights_dir, tp=cfg.model.tp)
    ref_weights = jax.tree.map(lambda w: w.astype(jnp.bfloat16), model.weights)  # frozen
    model.weights = jax.tree.map(lambda w: w.astype(jnp.float32), model.weights)
    make_train, valid_batches = lm_utils.load_batches(cfg.data, model.tokenizer, cfg.run.seed)
    optimizer = utils.make_optimizer(cfg.opt.optimizer, cfg.opt.lr)
    opt_state = optimizer.init(model.weights)
    run_dir = utils.start_run(cfg)

    def seq_logprobs(weights, batch):
        """Summed logprob and token count of the chosen and rejected responses."""
        out = []
        for side in ("chosen", "rejected"):
            logp, mask = lm_utils.logprobs(
                model.forward, weights, batch[f"{side}_tokens"], batch[f"{side}_mask"])
            out.append(((logp * mask).sum(1), mask.sum(1)))
        return out

    def loss_fn(weights, ref_weights, batch):
        (logp_w, n_w), (logp_l, n_l) = seq_logprobs(weights, batch)
        reward_w, reward_l = logp_w, logp_l
        if cfg.dpo.ref_weight:
            (ref_w, _), (ref_l, _) = seq_logprobs(ref_weights, batch)
            reward_w = reward_w - cfg.dpo.ref_weight * ref_w
            reward_l = reward_l - cfg.dpo.ref_weight * ref_l
        if cfg.dpo.norm:
            reward_w, reward_l = reward_w / n_w, reward_l / n_l
        loss = -jax.nn.log_sigmoid(cfg.dpo.beta * (reward_w - reward_l) - cfg.dpo.margin).mean()
        return loss - cfg.dpo.sft_weight * (logp_w / n_w).mean()

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(weights, opt_state, ref_weights, batch):
        loss, grads = jax.value_and_grad(loss_fn)(weights, ref_weights, batch)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        return optax.apply_updates(weights, updates), opt_state, loss

    @jax.jit
    def valid_step(weights, batch):
        (logp_w, _), (logp_l, _) = seq_logprobs(weights, batch)
        return logp_w, logp_l

    def validate():
        logp_w, logp_l = map(np.concatenate, zip(*[valid_step(model.weights, b) for b in valid_batches]))
        return {"valid/chosen_logprob": logp_w.mean(), "valid/rejected_logprob": logp_l.mean(),
                "valid/accuracy": (logp_w > logp_l).mean(), "valid/margin": (logp_w - logp_l).mean()}

    step = 0
    for epoch in range(cfg.opt.epochs):
        for batch in tqdm(make_train(epoch), desc=f"epoch {epoch}"):
            metrics = {}
            if valid_batches and step % cfg.log.valid_every_steps == 0:
                metrics |= validate()
            model.weights, opt_state, loss = train_step(model.weights, opt_state, ref_weights, batch)
            if step % cfg.log.every_steps == 0:
                metrics |= {"epoch": epoch, "loss": loss}
            if metrics:
                utils.log_metrics(step, metrics, run_dir)
            step += 1

    if cfg.log.save_model and jax.process_index() == 0:
        models.save(model, run_dir / "final_model")
    if jax.process_index() == 0:
        wandb.finish()


if __name__ == "__main__":
    train(utils.parse_cfg(DEFAULT_CFG))
