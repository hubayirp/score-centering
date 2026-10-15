"""Supervised fine-tuning or pretraining (model.init=random): next-token prediction on a Hugging Face
dataset (format=text packs documents and trains on all tokens, format=chat on assistant turns only)."""
import itertools
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import lm_utils
import models
import utils


DEFAULT_CFG = """
model:
  source: Qwen/Qwen3-0.6B
  weights_dir: ~/.cache/postax/weights
  tp: 1
  init: pretrained  # pretrained | random (pretraining: only the config and tokenizer come from source)
  config: null  # random init only: config overrides, e.g. model.config.num_hidden_layers=4
data:
  path: allenai/tulu-3-sft-olmo-2-mixture-0225
  name: null
  split: train
  streaming: false  # stream rather than download (large pretraining corpora)
  format: chat  # text (documents packed into full rows, EOS-separated) | chat
  seq_len: 2048
  batch_size: 32
  valid_rows: 512  # dataset rows held out for validation
opt:
  optimizer: adamw  # adamw | sgd
  lr: 1e-5
  b2: 0.997
  weight_decay: 0.02
  warmup_steps: 0
  cosine: false  # decay the LR to 0 over the run
  epochs: 1
  tokens: null  # train on this many tokens (batch_size * seq_len per step) instead of opt.epochs
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
    overrides = OmegaConf.to_container(cfg.model.config) if cfg.model.config else None
    model = models.load(cfg.model.source, cfg.model.weights_dir, tp=cfg.model.tp, init=cfg.model.init,
                        overrides=overrides, seed=cfg.run.seed)
    model.weights = jax.tree.map(lambda w: w.astype(jnp.float32), model.weights)
    make_train, valid_batches, steps_per_epoch = lm_utils.load_batches(cfg.data, model.tokenizer, cfg.run.seed)
    if cfg.opt.tokens:
        steps = math.ceil(cfg.opt.tokens / (cfg.data.batch_size * cfg.data.seq_len))
    else:
        steps = cfg.opt.epochs * steps_per_epoch if steps_per_epoch else None
    assert steps or not cfg.opt.cosine, "opt.cosine needs the run length: set opt.tokens"
    optimizer = utils.make_optimizer(cfg.opt.optimizer, cfg.opt.lr, b2=cfg.opt.b2, weight_decay=cfg.opt.weight_decay,
                                     warmup_steps=cfg.opt.warmup_steps, decay_steps=steps if cfg.opt.cosine else 0)
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

    epochs = itertools.count() if cfg.opt.tokens else range(cfg.opt.epochs)
    batches = ((epoch, batch) for epoch in epochs for batch in make_train(epoch))
    for step, (epoch, batch) in enumerate(tqdm(itertools.islice(batches, steps), total=steps)):
        metrics = {}
        if valid_batches and step % cfg.log.valid_every_steps == 0:
            metrics["valid_loss"] = np.mean([loss_fn(model.weights, b) for b in valid_batches])
        model.weights, opt_state, loss = train_step(model.weights, opt_state, batch)
        if step % cfg.log.every_steps == 0:
            metrics |= {"epoch": epoch, "loss": loss}
        if metrics:
            utils.log_metrics(step, metrics, run_dir)

    if cfg.log.save_model and jax.process_index() == 0:
        models.save(model, run_dir / "final_model")
    if jax.process_index() == 0:
        wandb.finish()


if __name__ == "__main__":
    train(utils.parse_cfg(DEFAULT_CFG))
