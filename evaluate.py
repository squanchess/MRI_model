"""
评估预训练后的 DINO backbone。

用法：
    python evaluate.py \
        --checkpoint outputs/pretrain_dino/checkpoint.pt \
        --data_dir /data/IXI-T1 \
        --architecture vit_small_patch16_96
"""
import argparse

import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
)
from torch.utils.data import DataLoader

import src.models as models
from src.data.mri_dataset import IXIDataset
from src.engine.evaluator import LinearProbe, extract_features, knn_evaluate


def get_eval_transform(roi_size=(192, 224, 192), spacing=(1.0, 1.0, 1.0)):
    """简单评估变换：确定性处理，不包含数据增强。"""
    return Compose([
        LoadImaged(keys=("image",)),
        EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
        ScaleIntensityRangePercentilesd(
            keys=("image",), lower=0.5, upper=99.5,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        Orientationd(keys=("image",), axcodes="RAS"),
        Spacingd(keys=("image",), pixdim=spacing, mode=("bilinear",)),
        CenterSpatialCropd(keys=("image",), roi_size=roi_size),
        SpatialPadd(keys=("image",), spatial_size=roi_size),
        EnsureTyped(keys=("image",), dtype=torch.float32),
    ])


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained DINO")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="vit_small_patch16_96")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # --- 加载 backbone ---
    backbone = getattr(models, args.architecture)(num_classes=0)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # 处理 DINO checkpoint，提取 student backbone 权重
    state_dict = ckpt.get("model", ckpt)
    backbone_state = {}
    for k, v in state_dict.items():
        if k.startswith("backbone_student."):
            new_key = k.replace("backbone_student.", "")
            backbone_state[new_key] = v
    if backbone_state:
        msg = backbone.load_state_dict(backbone_state, strict=False)
        print(f"Loaded student backbone: {msg}")
    else:
        # 如果不是 DINO 训练产物，则尝试直接加载
        msg = backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights directly: {msg}")

    # --- 构建数据集 ---
    transform = get_eval_transform()
    dataset = IXIDataset(args.data_dir, transform=transform)
    print(f"Dataset: {len(dataset)} volumes")

    # 简单划分训练集/测试集（80/20）
    n = len(dataset)
    n_train = int(0.8 * n)
    train_dataset = torch.utils.data.Subset(dataset, range(n_train))
    test_dataset = torch.utils.data.Subset(dataset, range(n_train, n))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=False)

    # --- 提取特征 ---
    print("Extracting features...")
    train_features, _ = extract_features(backbone, train_loader, device)
    test_features, _ = extract_features(backbone, test_loader, device)
    print(f"  Train features: {train_features.shape}")
    print(f"  Test features:  {test_features.shape}")

    # --- 分析特征质量（无需标签）---
    print("\nFeature statistics:")
    print(f"  Mean norm: {train_features.norm(dim=1).mean():.4f}")
    print(f"  Std norm:  {train_features.norm(dim=1).std():.4f}")

    # 余弦相似度分布应当具有一定离散性，而不是塌缩到一起
    import torch.nn.functional as F
    normed = F.normalize(train_features, dim=1)
    cos_sim = normed @ normed.t()
    # 排除对角线上的自相似
    mask = ~torch.eye(cos_sim.shape[0], dtype=torch.bool)
    off_diag = cos_sim[mask]
    print(f"  Off-diagonal cosine sim: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
    print(f"  (Good: mean close to 0, std > 0.1. Bad: mean ~1 = collapsed)")

    # 若文件名中包含站点信息，可在这里继续扩展站点级别的 kNN 评估
    print("\nDone. For kNN/linear probe with labels, use --label_csv.")


if __name__ == "__main__":
    main()
