"""
MRI Preprocessing Utilities.

Standalone preprocessing functions for brain MRI data. These can be:
  1. Run offline via scripts/preprocess_mri.py to save preprocessed volumes
  2. Imported and composed into MONAI transform pipelines

Key differences from CT preprocessing:
  - No HU windowing — MRI has no standardized intensity scale
  - Percentile-based intensity normalization instead
  - Optional N4 bias field correction (slow, best done offline)
  - Isotropic resampling (brain MRI is typically near-isotropic)

For IXI T1 data:
  - Input: ~256×256×130-150, spacing ~0.94×0.94×1.2 mm
  - After preprocessing: 192×224×192 (or similar), spacing 1.0×1.0×1.0 mm
"""
import os
import glob
import argparse
from typing import Tuple, Optional

import numpy as np


def get_preprocess_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    roi_size: Tuple[float, float, float] = (192, 224, 192),
    output_dir: Optional[str] = None,
):
    """Build a MONAI preprocessing pipeline for brain MRI.

    This pipeline performs:
      1. Load NIfTI
      2. Add channel dimension
      3. Reorient to RAS
      4. Resample to isotropic spacing
      5. Percentile intensity clipping (0.5th-99.5th) → normalize to [0, 1]
      6. Center crop to ROI
      7. Spatial pad (if needed)
      8. Optionally save to output_dir

    Args:
        spacing: Target voxel spacing in mm.
        roi_size: Center crop size after resampling.
        output_dir: If provided, save preprocessed volumes here.

    Returns:
        MONAI Compose transform.
    """
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        ScaleIntensityRangePercentilesd,
        CenterSpatialCropd,
        SpatialPadd,
        EnsureTyped,
        SaveImaged,
    )

    transforms = [
        LoadImaged(keys=("image",)),
        EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
        Orientationd(keys=("image",), axcodes="RAS"),
        Spacingd(
            keys=("image",),
            pixdim=spacing,
            mode=("bilinear",),
        ),
        ScaleIntensityRangePercentilesd(
            keys=("image",),
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CenterSpatialCropd(
            keys=("image",),
            roi_size=roi_size,
        ),
        SpatialPadd(
            keys=("image",),
            spatial_size=roi_size,
        ),
        EnsureTyped(keys=("image",), dtype="float32"),
    ]

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        transforms.append(
            SaveImaged(
                keys=("image",),
                output_dir=output_dir,
                output_postfix="preprocessed",
                resample=False,
                separate_folder=False,
            )
        )

    return Compose(transforms)


def compute_dataset_statistics(
    data_dir: str,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_samples: int = 50,
):
    """Compute intensity statistics across a sample of MRI volumes.

    Useful for validating preprocessing and choosing normalization parameters.

    Args:
        data_dir: Directory with NIfTI files.
        spacing: Spacing for resampling before computing stats.
        max_samples: Maximum number of volumes to sample.

    Returns:
        Dict with global mean, std, min, max, percentiles, and per-volume shapes.
    """
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd,
        Orientationd, Spacingd,
    )

    files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*.nii")))
    if not files:
        raise FileNotFoundError(f"No NIfTI files in {data_dir}")

    files = files[:max_samples]
    print(f"Computing statistics on {len(files)} volumes...")

    loader = Compose([
        LoadImaged(keys=("image",)),
        EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
        Orientationd(keys=("image",), axcodes="RAS"),
        Spacingd(keys=("image",), pixdim=spacing, mode=("bilinear",)),
    ])

    all_means = []
    all_stds = []
    shapes = []

    for f in files:
        data = loader({"image": f})
        vol = data["image"].numpy()
        # Compute stats on non-zero region (rough brain mask)
        brain = vol[vol > vol.mean() * 0.1]
        all_means.append(float(brain.mean()))
        all_stds.append(float(brain.std()))
        shapes.append(vol.shape[1:])  # exclude channel dim

    stats = {
        "n_volumes": len(files),
        "global_mean": float(np.mean(all_means)),
        "global_std": float(np.mean(all_stds)),
        "mean_range": (float(np.min(all_means)), float(np.max(all_means))),
        "std_range": (float(np.min(all_stds)), float(np.max(all_stds))),
        "shapes": shapes,
        "shape_min": tuple(int(x) for x in np.min(shapes, axis=0)),
        "shape_max": tuple(int(x) for x in np.max(shapes, axis=0)),
    }

    print(f"  Global mean: {stats['global_mean']:.2f} ± range [{stats['mean_range'][0]:.2f}, {stats['mean_range'][1]:.2f}]")
    print(f"  Global std:  {stats['global_std']:.2f} ± range [{stats['std_range'][0]:.2f}, {stats['std_range'][1]:.2f}]")
    print(f"  Shape range: {stats['shape_min']} to {stats['shape_max']}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess IXI T1 MRI data")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Input directory with IXI T1 NIfTI files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for preprocessed files")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Target voxel spacing in mm")
    parser.add_argument("--roi_size", type=int, nargs=3, default=[192, 224, 192],
                        help="Center crop size")
    parser.add_argument("--stats_only", action="store_true",
                        help="Only compute dataset statistics, don't preprocess")
    args = parser.parse_args()

    if args.stats_only:
        compute_dataset_statistics(args.data_dir, tuple(args.spacing))
    else:
        if args.output_dir is None:
            raise ValueError("--output_dir required when not using --stats_only")

        from monai.data import Dataset, DataLoader

        files = sorted(glob.glob(os.path.join(args.data_dir, "*.nii.gz")))
        if not files:
            files = sorted(glob.glob(os.path.join(args.data_dir, "*.nii")))
        data_list = [{"image": f} for f in files]

        transform = get_preprocess_transforms(
            spacing=tuple(args.spacing),
            roi_size=tuple(args.roi_size),
            output_dir=args.output_dir,
        )
        dataset = Dataset(data=data_list, transform=transform)
        loader = DataLoader(dataset, batch_size=1, num_workers=4)

        print(f"Preprocessing {len(data_list)} volumes...")
        for i, batch in enumerate(loader):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(data_list)}")
        print(f"Done. Saved to {args.output_dir}")
