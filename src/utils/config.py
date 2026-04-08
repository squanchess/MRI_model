"""
基于 OmegaConf 的配置管理工具。

负责从项目的 `configs/` 目录加载 YAML 配置，
合并命令行覆盖项，并应用学习率缩放规则。
"""
import math
import os
from pathlib import Path

from omegaconf import OmegaConf

# 项目根目录：相对于 src/utils/config.py 向上三级
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pretrain_wholebody.yaml"


def load_default_config() -> OmegaConf:
    """加载默认的 `pretrain.yaml` 配置。"""
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Default config not found at {DEFAULT_CONFIG_PATH}. "
            f"Make sure you're running from the project root."
        )
    return OmegaConf.load(DEFAULT_CONFIG_PATH)


def get_cfg_from_args(args) -> OmegaConf:
    """构建配置：默认 YAML -> 用户 YAML -> 命令行覆盖项。

    Args:
        args: 包含 config_file、output_dir、opts 属性的 Namespace。

    Returns:
        合并后的 OmegaConf 配置对象。
    """
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts = [] if args.opts is None else args.opts
    args.opts += [f"train.output_dir={args.output_dir}"]

    default_cfg = load_default_config()

    if args.config_file is not None and os.path.isfile(args.config_file):
        user_cfg = OmegaConf.load(args.config_file)
        cfg = OmegaConf.merge(default_cfg, user_cfg, OmegaConf.from_cli(args.opts))
    else:
        cfg = OmegaConf.merge(default_cfg, OmegaConf.from_cli(args.opts))

    return cfg


def apply_scaling_rules(cfg, world_size: int = None) -> OmegaConf:
    """应用学习率缩放规则：`lr = base_lr * f(effective_batch / ref_batch)`。"""
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

    scale_factor = cfg.train.batch_size_per_gpu * world_size * cfg.train.grad_accum_steps
    scale_factor /= ref_batch_size

    if scaling_type == "sqrt":
        cfg.optim.lr *= math.sqrt(scale_factor)
    elif scaling_type == "linear":
        cfg.optim.lr *= scale_factor
    else:
        raise NotImplementedError(f"Unsupported scaling type: {scaling_type}")

    return cfg


def write_config(cfg, output_dir: str, name: str = "config.yaml") -> str:
    """将解析后的最终配置保存到输出目录。"""
    os.makedirs(output_dir, exist_ok=True)
    saved_path = os.path.join(output_dir, name)
    with open(saved_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_path
