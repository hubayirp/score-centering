"""Supervised fine-tuning: next-token prediction on a Hugging Face dataset
(format=text trains on all tokens, format=chat on assistant turns only)."""
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
  path: allenai/tulu-3-sft-olmo-2-mixture-0225
  name: null
  split: train
  format: chat  # text | chat
  seq_len: 2048
  batch_size: 32
  valid_rows: 512
opt:
  optimizer: adamw
  lr: 1e-5
  epochs: 1
log:
  project: postax-lm
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
    model.weights = jax.tree.map(lambda w: w.astype(jnp.float32), model.weights)
    make_train, valid_batches = lm_utils.load_batches(cfg.data, model.tokenizer, cfg.run.seed)
    optimizer = utils.make_optimizer(cfg.opt.optimizer, cfg.opt.lr)
    opt_state = optimizer.init(model.weights)
    run_dir = utils.start_run(cfg)

    @jax.jit
    def loss_fn(weights, batch):
        logp, mask = lm_utils.logprobs(model.forward, weights, batch["tokens"], batch["mask"])
        return -(logp * mask).sum() / jnp.maximum(mask.sum(), 1)

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(weights, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(weights, batch)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        return optax.apply_updates(weights, updates), opt_state, loss

    step = 0
    for epoch in range(cfg.opt.epochs):
        for batch in tqdm(make_train(epoch), desc=f"epoch {epoch}"):
            metrics = {}
            if valid_batches and step % cfg.log.valid_every_steps == 0:
                metrics["valid_loss"] = np.mean([loss_fn(model.weights, b) for b in valid_batches])
            model.weights, opt_state, loss = train_step(model.weights, opt_state, batch)
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
