"""
Configuration management using OmegaConf.

Adapted from SPECTRE (MIT License):
  - Simplified config structure (single pretrain.yaml vs multiple)
  - Added MRI data config section
  - WandB project name parameterized
"""
import os
import math
from pathlib import Path

from omegaconf import OmegaConf

from src.utils.misc import fix_random_seeds


# Load default config
_CONFIG_DIR = Path(__file__).parent.resolve()
default_config = OmegaConf.load(_CONFIG_DIR / "pretrain.yaml")


def apply_scaling_rules(cfg, world_size: int = None):
    """Apply learning rate scaling rules based on effective batch size."""
    base_lr = cfg.optim.base_lr
    cfg.optim.lr = base_lr

    if cfg.optim.scaling_rule == "constant":
        return cfg

    try:
        scaling_type, ref_batch_size = cfg.optim.scaling_rule.split("_wrt_")
        ref_batch_size = float(ref_batch_size)
    except ValueError:
        raise NotImplementedError(f"Unknown scaling rule: {cfg.optim.scaling_rule}")

    if world_size is None:
        import torch.distributed as dist
        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

    scale_factor = cfg.train.batch_size_per_gpu * world_size
    scale_factor /= ref_batch_size
    scale_factor *= cfg.train.grad_accum_steps

    if scaling_type == "sqrt":
        cfg.optim.lr *= math.sqrt(scale_factor)
    elif scaling_type == "linear":
        cfg.optim.lr *= scale_factor
    else:
        raise NotImplementedError(f"Unsupported scaling type: {scaling_type}")

    return cfg


def get_cfg_from_args(args):
    """Build config from args + YAML file + CLI overrides."""
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts = [] if args.opts is None else args.opts
    args.opts += [f"train.output_dir={args.output_dir}"]

    default_cfg = default_config
    if args.config_file is not None and os.path.isfile(args.config_file):
        user_cfg = OmegaConf.load(args.config_file)
        cfg = OmegaConf.merge(default_cfg, user_cfg, OmegaConf.from_cli(args.opts))
    else:
        cfg = OmegaConf.merge(default_cfg, OmegaConf.from_cli(args.opts))
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    """Save resolved config to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    saved_path = os.path.join(output_dir, name)
    with open(saved_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_path


def setup(args):
    """Full setup: parse config, set seeds, init distributed, apply LR scaling.

    Returns:
        (cfg, accelerator) tuple.
    """
    from accelerate import Accelerator, DataLoaderConfiguration

    cfg = get_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seeds
    seed = cfg.train.seed
    try:
        import torch.distributed as dist
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    except Exception:
        rank = 0
    fix_random_seeds(seed + rank)

    # Init Accelerate
    dataloader_config = DataLoaderConfiguration(non_blocking=cfg.train.pin_memory)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        log_with="wandb" if cfg.train.log_wandb else None,
        dataloader_config=dataloader_config,
    )

    if cfg.train.log_wandb:
        accelerator.init_trackers(
            project_name=cfg.train.wandb_project,
            config={k: v for d in cfg.values() for k, v in d.items()},
        )

    # Apply LR scaling after distributed init
    apply_scaling_rules(cfg)
    write_config(cfg, args.output_dir)

    return cfg, accelerator
