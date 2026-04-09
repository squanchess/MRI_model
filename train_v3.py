"""
DINOv3 MRI 预训练入口脚本。

用法：
    # 单卡训练
    python train_v3.py --config_file configs/pretrain_dinov3.yaml

    # 多卡训练
    accelerate launch train_v3.py --config_file configs/pretrain_dinov3.yaml
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

from src.utils.config import apply_scaling_rules, get_cfg_from_args, write_config
from src.engine.trainer_v3 import TrainerV3


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINOv3 MRI Pretraining")
    parser.add_argument(
        "--config_file", type=str, default="configs/pretrain_dinov3.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/dinov3_mri",
        help="checkpoint 与日志输出目录",
    )
    parser.add_argument(
        "--opts", default=None, nargs=argparse.REMAINDER,
        help="覆盖配置项：key1=val1 key2=val2 ...",
    )
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    cfg = get_cfg_from_args(args)

    try:
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
    except Exception:
        world_size = 1

    cfg = apply_scaling_rules(cfg, world_size=world_size)
    write_config(cfg, cfg.train.output_dir)
    print(f"Effective LR: {cfg.optim.lr:.6f}")

    trainer = TrainerV3(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
