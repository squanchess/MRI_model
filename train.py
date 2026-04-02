"""
DINO MRI Pretraining — Entry Point.

Usage:
    # Single GPU
    python train.py --output_dir outputs/dino_ixi

    # Custom config
    python train.py --config_file configs/pretrain.yaml --output_dir outputs/dino_ixi

    # Override params via CLI
    python train.py --output_dir outputs/debug \
        train.batch_size_per_gpu=4 optim.epochs=5 train.data_dir=data/IXI-T1

    # Multi-GPU
    accelerate launch train.py --output_dir outputs/dino_ixi

    # Convenience script
    bash scripts/pretrain.sh
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

from src.utils.config import get_cfg_from_args, apply_scaling_rules, write_config
from src.engine.trainer import Trainer


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINO MRI Pretraining")
    parser.add_argument(
        "--config_file", type=str, default=None,
        help="Path to YAML config. Defaults to configs/pretrain.yaml",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/dino_ixi",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--opts", default=None, nargs=argparse.REMAINDER,
        help="Override config: key1=val1 key2=val2 ...",
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

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
