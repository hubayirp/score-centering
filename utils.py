import json
from pathlib import Path

import jax
import optax
import wandb
from omegaconf import OmegaConf


def parse_cfg(default_cfg):
    """YAML defaults overridden by key=value command-line arguments."""
    cfg = OmegaConf.create(default_cfg)
    OmegaConf.set_struct(cfg, True)  # unknown keys are errors
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    if jax.process_index() == 0:
        wandb.init(project=cfg.log.project, mode=cfg.log.wandb_mode, name=cfg.log.run_name,
                   config=OmegaConf.to_container(cfg, resolve=True))
    return cfg


def make_optimizer(name, lr):
    return {"sgd": optax.sgd, "adamw": optax.adamw}[name](lr)


def start_run(cfg):
    """Create outputs/<project>/<run name> and save the resolved config there."""
    name = cfg.log.run_name or (wandb.run.id if jax.process_index() == 0 else "run")
    run_dir = Path(cfg.log.dir).expanduser() / cfg.log.project / name
    if jax.process_index() == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        config = OmegaConf.to_container(cfg, resolve=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"run_dir={run_dir}")
    return run_dir


def log_metrics(step, metrics, run_dir):
    metrics = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
    if jax.process_index() == 0:
        print(f"step {step}", metrics)
        wandb.log(metrics, step=step)
        with open(run_dir / "logs.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
