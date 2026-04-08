"""
DINO MRI 预训练入口脚本。

用法：
    # 单卡训练
    python train.py --output_dir outputs/dino_ixi

    # 指定自定义配置
    python train.py --config_file configs/pretrain.yaml --output_dir outputs/dino_ixi

    # 通过命令行覆盖配置项
    python train.py --output_dir outputs/debug \
        train.batch_size_per_gpu=4 optim.epochs=5 train.data_dir=data/IXI-T1

    # 多卡训练
    accelerate launch train.py --output_dir outputs/dino_ixi

    # 使用便捷脚本
    bash scripts/pretrain.sh
"""

# 避免部分环境下的 OpenMP 冲突错误
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

# 配置读取与处理
from src.utils.config import apply_scaling_rules, get_cfg_from_args, write_config

# 训练入口
from src.engine.trainer import Trainer


# 命令行参数定义
def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINO MRI Pretraining")
    parser.add_argument(
        "--config_file", type=str, default=None,
        help="YAML 配置文件路径，默认使用 configs/pretrain_wholebody.yaml",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/dino",
        help="checkpoint 与日志输出目录",
    )
    parser.add_argument(
        "--opts", default=None, nargs=argparse.REMAINDER,
        help="覆盖配置项：key1=val1 key2=val2 ...",
    )
    return parser


# 训练主流程
def main():
    # 从配置文件和命令行构造最终配置
    parser = get_args_parser()
    args = parser.parse_args()
    cfg = get_cfg_from_args(args)

    try:
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
    except Exception:
        world_size = 1

    # 根据 world size 调整学习率等超参数
    cfg = apply_scaling_rules(cfg, world_size=world_size)

    # 将最终配置写入输出目录，方便复现实验
    write_config(cfg, cfg.train.output_dir)
    print(f"Effective LR: {cfg.optim.lr:.6f}")

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
