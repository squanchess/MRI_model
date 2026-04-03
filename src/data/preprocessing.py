"""
MRI 预处理工具。

这些独立的脑 MRI 预处理函数既可以：
  1. 通过离线脚本运行，保存预处理后的体数据；
  2. 直接导入并组合到 MONAI 的变换流水线中。

与 CT 预处理相比，主要差异包括：
  - 不使用 HU 窗宽窗位，MRI 没有统一的强度标尺；
  - 改用基于百分位数的强度归一化；
  - 可选 N4 偏置场校正（速度较慢，更适合离线执行）；
  - 使用各向同性重采样（脑 MRI 通常接近各向同性）。

对于 IXI T1 数据：
  - 输入大约为 256x256x130-150，spacing 约 0.94x0.94x1.2 mm；
  - 预处理后通常变为 192x224x192 左右，spacing 为 1.0x1.0x1.0 mm。
"""
import argparse
import glob
import os
from typing import Optional, Tuple

import numpy as np


def get_preprocess_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    roi_size: Tuple[float, float, float] = (192, 224, 192),
    output_dir: Optional[str] = None,
):
    """构建适用于脑 MRI 的 MONAI 预处理流水线。

    该流水线依次执行：
      1. Load NIfTI
      2. Add channel dimension
      3. Reorient to RAS
      4. Resample to isotropic spacing
      5. 百分位裁剪强度（0.5%-99.5%）并归一化到 [0, 1]
      6. Center crop to ROI
      7. Spatial pad（如有需要）
      8. 可选：保存到 output_dir

    参数：
        spacing: 目标体素间距（单位 mm）。
        roi_size: 重采样后的中心裁剪大小。
        output_dir: 若提供，则将预处理后的体数据保存到该目录。

    返回：
        MONAI Compose 变换对象。
    """
    from monai.transforms import (
        CenterSpatialCropd,
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        Orientationd,
        SaveImaged,
        ScaleIntensityRangePercentilesd,
        Spacingd,
        SpatialPadd,
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
    """统计一批 MRI 体数据的强度分布。

    可用于验证预处理效果，并辅助选择归一化参数。

    参数：
        data_dir: NIfTI 文件所在目录。
        spacing: 计算统计量前所使用的重采样 spacing。
        max_samples: 最多抽样多少个体数据。

    返回：
        包含全局均值、标准差、范围和各体数据形状信息的字典。
    """
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        Orientationd,
        Spacingd,
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
        # 仅在非零区域统计，近似视作脑区掩码
        brain = vol[vol > vol.mean() * 0.1]
        all_means.append(float(brain.mean()))
        all_stds.append(float(brain.std()))
        shapes.append(vol.shape[1:])  # 去掉通道维

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

    print(f"  全局均值: {stats['global_mean']:.2f}，范围 [{stats['mean_range'][0]:.2f}, {stats['mean_range'][1]:.2f}]")
    print(f"  全局标准差: {stats['global_std']:.2f}，范围 [{stats['std_range'][0]:.2f}, {stats['std_range'][1]:.2f}]")
    print(f"  形状范围: {stats['shape_min']} 到 {stats['shape_max']}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理 IXI T1 MRI 数据")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="包含 IXI T1 NIfTI 文件的输入目录")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="预处理结果输出目录")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="目标体素间距，单位 mm")
    parser.add_argument("--roi_size", type=int, nargs=3, default=[192, 224, 192],
                        help="中心裁剪尺寸")
    parser.add_argument("--stats_only", action="store_true",
                        help="仅统计数据集信息，不执行预处理")
    args = parser.parse_args()

    if args.stats_only:
        compute_dataset_statistics(args.data_dir, tuple(args.spacing))
    else:
        if args.output_dir is None:
            raise ValueError("--output_dir required when not using --stats_only")

        from monai.data import DataLoader, Dataset

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

        print(f"开始预处理 {len(data_list)} 个体数据...")
        for i, batch in enumerate(loader):
            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{len(data_list)}")
        print(f"完成，结果已保存到 {args.output_dir}")
