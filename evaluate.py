"""
Evaluate pretrained DINO backbone.

Usage:
    python evaluate.py \
        --checkpoint outputs/pretrain_dino/checkpoint.pt \
        --data_dir /data/IXI-T1 \
        --architecture vit_small_patch16_96
"""
import argparse

import torch
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd, Orientationd, Spacingd,
    CenterSpatialCropd, SpatialPadd, EnsureTyped,
)

import src.models as models
from src.data.mri_dataset import IXIDataset
from src.engine.evaluator import extract_features, knn_evaluate, LinearProbe


def get_eval_transform(roi_size=(192, 224, 192), spacing=(1.0, 1.0, 1.0)):
    """Simple evaluation transform — deterministic, no augmentation."""
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

    # --- Load backbone ---
    backbone = getattr(models, args.architecture)(num_classes=0)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Handle DINO checkpoint (extract student backbone weights)
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
        # Try loading directly
        msg = backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights directly: {msg}")

    # --- Build dataset ---
    transform = get_eval_transform()
    dataset = IXIDataset(args.data_dir, transform=transform)
    print(f"Dataset: {len(dataset)} volumes")

    # Simple train/test split (80/20)
    n = len(dataset)
    n_train = int(0.8 * n)
    train_dataset = torch.utils.data.Subset(dataset, range(n_train))
    test_dataset = torch.utils.data.Subset(dataset, range(n_train, n))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=False)

    # --- Extract features ---
    print("Extracting features...")
    train_features, _ = extract_features(backbone, train_loader, device)
    test_features, _ = extract_features(backbone, test_loader, device)
    print(f"  Train features: {train_features.shape}")
    print(f"  Test features:  {test_features.shape}")

    # --- Feature quality analysis (no labels needed) ---
    print("\nFeature statistics:")
    print(f"  Mean norm: {train_features.norm(dim=1).mean():.4f}")
    print(f"  Std norm:  {train_features.norm(dim=1).std():.4f}")

    # Cosine similarity distribution (should be diverse, not collapsed)
    import torch.nn.functional as F
    normed = F.normalize(train_features, dim=1)
    cos_sim = normed @ normed.t()
    # Exclude diagonal
    mask = ~torch.eye(cos_sim.shape[0], dtype=torch.bool)
    off_diag = cos_sim[mask]
    print(f"  Off-diagonal cosine sim: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
    print(f"  (Good: mean close to 0, std > 0.1. Bad: mean ~1 = collapsed)")

    # Site-based kNN (if site info available from filenames)
    print("\nDone. For kNN/linear probe with labels, use --label_csv.")


if __name__ == "__main__":
    main()
