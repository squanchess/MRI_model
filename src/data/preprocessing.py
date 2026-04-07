"""
MRI 预处理工具。

扩展自脑部专用版本，增加全身 MRI 支持：
  - 新增 CropForeground，替代 CenterSpatialCrop，自动去除背景空气
  - 新增 quality_check_volume() 质量检查函数
  - compute_dataset_statistics() 支持递归目录结构
  - roi_size 语义变更：从「中心裁剪目标」变为「最小尺寸保证」

对于 IXI T1 数据：
  - 输入大约为 256x256x130-150，spacing 约 0.94x0.94x1.2 mm
  - 预处理后通常变为 192x224x192 左右，spacing 为 1.0x1.0x1.0 mm

对于全身数据：
  - CropForeground 自动定位组织区域，兼容任意体位和解剖
  - SpatialPad 确保最终体数据不小于 min_spatial_size
"""
import argparse
import glob
import os
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 质量检查
# ---------------------------------------------------------------------------

def quality_check_volume(
    nifti_path: str,
    min_dim: int = 48,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[bool, str]:
    """检查单个 NIfTI 文件是否满足训练要求。

    检查项：
      1. 文件可读、头信息完整
      2. 三维体数据（非 2D 或 4D+）
      3. 体素间距合理且非零
      4. 无 NaN / Inf 值
      5. 强度标准差 > 1e-6（非近常数图像）
      6. 重采样后每个维度 >= min_dim

    参数：
        nifti_path: NIfTI 文件路径。
        min_dim: 重采样后各维度的最小体素数。
        target_spacing: 重采样目标体素间距（mm）。

    返回：
        (通过与否, 原因描述) 的元组。
    """
    import nibabel as nib

    # 1) 文件可读
    try:
        img = nib.load(nifti_path)
        header = img.header
    except Exception as e:
        return False, f"无法加载: {e}"

    # 2) 维度检查
    shape = img.shape
    if len(shape) < 3:
        return False, f"维度不足: shape={shape}，需要 3D"
    if len(shape) > 4:
        return False, f"维度过多: shape={shape}"
    if len(shape) == 4 and shape[3] == 1:
        # 4D 但时间/通道维为 1，可以当作 3D 用
        shape = shape[:3]
    elif len(shape) == 4 and shape[3] > 1:
        # 4D 多帧数据（如心脏 cine），需要先提取单帧
        return False, f"4D 多帧数据: shape={shape}，需先提取单帧"

    spatial_shape = shape[:3]

    # 3) 体素间距检查
    try:
        pixdim = header.get_zooms()[:3]
        if any(s <= 0 or s > 100 for s in pixdim):
            return False, f"体素间距异常: pixdim={pixdim}"
    except Exception:
        return False, "无法读取体素间距"

    # 4) 数据完整性：读取并检查 NaN/Inf
    try:
        data = img.get_fdata(dtype=np.float32)
    except Exception as e:
        return False, f"无法读取体素数据: {e}"

    if np.any(np.isnan(data)):
        return False, "包含 NaN 值"
    if np.any(np.isinf(data)):
        return False, "包含 Inf 值"

    # 5) 强度标准差
    std = float(np.std(data))
    if std < 1e-6:
        return False, f"近常数图像: std={std:.2e}"

    # 6) 重采样后尺寸预估
    for i in range(3):
        estimated_dim = int(spatial_shape[i] * pixdim[i] / target_spacing[i])
        if estimated_dim < min_dim:
            return False, (
                f"重采样后第 {i} 维过小: "
                f"{spatial_shape[i]}×{pixdim[i]:.2f}mm / {target_spacing[i]:.2f}mm "
                f"≈ {estimated_dim} < {min_dim}"
            )

    return True, "通过"


# ---------------------------------------------------------------------------
# 预处理流水线
# ---------------------------------------------------------------------------

def get_preprocess_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_spatial_size: Tuple[int, int, int] = (192, 192, 192),
    output_dir: Optional[str] = None,
    use_foreground_crop: bool = True,
    foreground_margin: int = 10,
):
    """构建适用于通用 MRI 的 MONAI 预处理流水线。

    与脑部专用版本的区别：
      - CenterSpatialCrop -> CropForeground：自动去除背景空气，
        不依赖解剖居中假设
      - roi_size -> min_spatial_size：语义为最小尺寸保证（SpatialPad），
        不再做中心裁剪

    流水线步骤：
      1. Load NIfTI
      2. Add channel dimension
      3. Reorient to RAS
      4. Resample to target spacing
      5. 百分位裁剪强度（0.5%-99.5%）并归一化到 [0, 1]
      6. 前景裁剪（可选）或中心裁剪
      7. Spatial pad（确保最小尺寸）
      8. 可选：保存到 output_dir

    参数：
        spacing: 目标体素间距（mm）。
        min_spatial_size: 最小空间尺寸（不足时补边）。
        output_dir: 若提供，则保存预处理后的体数据。
        use_foreground_crop: 是否使用前景裁剪（True=全身，False=脑部）。
        foreground_margin: 前景裁剪时保留的边缘体素数。

    返回：
        MONAI Compose 变换对象。
    """
    from monai.transforms import (
        CenterSpatialCropd,
        Compose,
        CropForegroundd,
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
    ]

    # 裁剪策略
    if use_foreground_crop:
        transforms.append(
            CropForegroundd(
                keys=("image",),
                source_key="image",
                margin=foreground_margin,
            )
        )
    else:
        # 向后兼容：脑部专用的中心裁剪
        transforms.append(
            CenterSpatialCropd(
                keys=("image",),
                roi_size=min_spatial_size,
            )
        )

    transforms.extend([
        SpatialPadd(
            keys=("image",),
            spatial_size=min_spatial_size,
        ),
        EnsureTyped(keys=("image",), dtype="float32"),
    ])

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


def get_brain_preprocess_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    roi_size: Tuple[int, int, int] = (192, 224, 192),
    output_dir: Optional[str] = None,
):
    """向后兼容：脑部 MRI 预处理（使用 CenterSpatialCrop）。"""
    return get_preprocess_transforms(
        spacing=spacing,
        min_spatial_size=roi_size,
        output_dir=output_dir,
        use_foreground_crop=False,
    )


# ---------------------------------------------------------------------------
# 数据集统计
# ---------------------------------------------------------------------------

def compute_dataset_statistics(
    data_dir: str,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_samples: int = 50,
    recursive: bool = True,
):
    """统计一批 MRI 体数据的强度分布。

    可用于验证预处理效果，并辅助选择归一化参数。

    参数：
        data_dir: NIfTI 文件所在目录。
        spacing: 计算统计量前所使用的重采样 spacing。
        max_samples: 最多抽样多少个体数据。
        recursive: 是否递归搜索子目录。

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

    # 递归或非递归查找
    files = []
    if recursive:
        for root, _dirs, fnames in os.walk(data_dir, followlinks=True):
            for fname in fnames:
                if fname.lower().endswith((".nii.gz", ".nii")):
                    files.append(os.path.join(root, fname))
        files = sorted(files)
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
        if not files:
            files = sorted(glob.glob(os.path.join(data_dir, "*.nii")))

    if not files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到 NIfTI 文件")

    files = files[:max_samples]
    print(f"正在统计 {len(files)} 个体数据（共找到 {len(files)} 个）...")

    loader = Compose([
        LoadImaged(keys=("image",)),
        EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
        Orientationd(keys=("image",), axcodes="RAS"),
        Spacingd(keys=("image",), pixdim=spacing, mode=("bilinear",)),
    ])

    all_means = []
    all_stds = []
    shapes = []
    spacings_raw = []

    for i, f in enumerate(files):
        try:
            data = loader({"image": f})
            vol = data["image"].numpy()
            # 仅在非零区域统计，近似视作组织掩码
            tissue = vol[vol > vol.mean() * 0.1]
            if len(tissue) == 0:
                tissue = vol.flatten()
            all_means.append(float(tissue.mean()))
            all_stds.append(float(tissue.std()))
            shapes.append(vol.shape[1:])  # 去掉通道维

            # 原始体素间距
            import nibabel as nib
            orig = nib.load(f)
            spacings_raw.append(orig.header.get_zooms()[:3])

            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(files)}")
        except Exception as e:
            print(f"  跳过 {os.path.basename(f)}: {e}")

    if not all_means:
        raise RuntimeError("所有文件均处理失败")

    stats = {
        "n_volumes": len(all_means),
        "global_mean": float(np.mean(all_means)),
        "global_std": float(np.mean(all_stds)),
        "mean_range": (float(np.min(all_means)), float(np.max(all_means))),
        "std_range": (float(np.min(all_stds)), float(np.max(all_stds))),
        "shapes": shapes,
        "shape_min": tuple(int(x) for x in np.min(shapes, axis=0)),
        "shape_max": tuple(int(x) for x in np.max(shapes, axis=0)),
        "raw_spacing_min": tuple(float(x) for x in np.min(spacings_raw, axis=0)),
        "raw_spacing_max": tuple(float(x) for x in np.max(spacings_raw, axis=0)),
    }

    print(f"  全局均值: {stats['global_mean']:.2f}，"
          f"范围 [{stats['mean_range'][0]:.2f}, {stats['mean_range'][1]:.2f}]")
    print(f"  全局标准差: {stats['global_std']:.2f}，"
          f"范围 [{stats['std_range'][0]:.2f}, {stats['std_range'][1]:.2f}]")
    print(f"  形状范围: {stats['shape_min']} 到 {stats['shape_max']}")
    print(f"  原始间距范围: {stats['raw_spacing_min']} 到 {stats['raw_spacing_max']}")

    return stats


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理 MRI 数据（通用全身版）")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="包含 NIfTI 文件的输入目录")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="预处理结果输出目录")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="目标体素间距，单位 mm")
    parser.add_argument("--min_spatial_size", type=int, nargs=3,
                        default=[192, 192, 192],
                        help="最小空间尺寸（SpatialPad 目标）")
    parser.add_argument("--use_foreground_crop", action="store_true", default=True,
                        help="使用前景裁剪（全身 MRI 推荐）")
    parser.add_argument("--use_center_crop", action="store_true", default=False,
                        help="使用中心裁剪（脑部 MRI 兼容模式）")
    parser.add_argument("--recursive", action="store_true", default=True,
                        help="递归搜索子目录")
    parser.add_argument("--stats_only", action="store_true",
                        help="仅统计数据集信息，不执行预处理")
    args = parser.parse_args()

    if args.use_center_crop:
        args.use_foreground_crop = False

    if args.stats_only:
        compute_dataset_statistics(
            args.data_dir, tuple(args.spacing), recursive=args.recursive,
        )
    else:
        if args.output_dir is None:
            raise ValueError("--output_dir 为必需参数（非 --stats_only 模式）")

        from monai.data import DataLoader, Dataset

        # 递归查找所有 NIfTI 文件
        files = []
        if args.recursive:
            for root, _dirs, fnames in os.walk(args.data_dir, followlinks=True):
                for fname in fnames:
                    if fname.lower().endswith((".nii.gz", ".nii")):
                        files.append(os.path.join(root, fname))
            files = sorted(files)
        else:
            files = sorted(glob.glob(os.path.join(args.data_dir, "*.nii.gz")))
            if not files:
                files = sorted(glob.glob(os.path.join(args.data_dir, "*.nii")))

        data_list = [{"image": f} for f in files]

        transform = get_preprocess_transforms(
            spacing=tuple(args.spacing),
            min_spatial_size=tuple(args.min_spatial_size),
            output_dir=args.output_dir,
            use_foreground_crop=args.use_foreground_crop,
        )
        dataset = Dataset(data=data_list, transform=transform)
        loader = DataLoader(dataset, batch_size=1, num_workers=4)

        print(f"开始预处理 {len(data_list)} 个体数据...")
        for i, batch in enumerate(loader):
            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{len(data_list)}")
        print(f"完成，结果已保存到 {args.output_dir}")
