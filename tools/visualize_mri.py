# -*- coding: utf-8 -*-
# @Time    : 2026/4/1 16:28
# @Author  : rkj
# @FileName: visualize_mri.py
# @Software: PyCharm

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立 MRI 可视化脚本
支持:
- NIfTI: .nii / .nii.gz
- NumPy: .npy

功能:
- 显示轴状(axial) / 冠状(coronal) / 矢状(sagittal) 三个正交切面
- 默认显示中间层
- 可手动指定切片索引
- 可按百分位裁剪强度，提升显示效果
- 可保存为 PNG

示例:
python tools/visualize_mri.py --input ./sample.nii.gz
python tools/visualize_mri.py --input ./sample.nii.gz --save ./mri_view.png
python tools/visualize_mri.py --input ./sample.nii.gz --axial 80 --coronal 90 --sagittal 70
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None


def load_volume(path: str) -> np.ndarray:
    p = Path(path)
    suffixes = p.suffixes

    if suffixes[-2:] == [".nii", ".gz"] or p.suffix == ".nii":
        if nib is None:
            raise ImportError("读取 .nii/.nii.gz 需要安装 nibabel: pip install nibabel")
        img = nib.load(str(p))
        vol = img.get_fdata()
    elif p.suffix == ".npy":
        vol = np.load(str(p))
    else:
        raise ValueError(f"暂不支持的文件格式: {p.name}")

    vol = np.asarray(vol, dtype=np.float32)

    # 若有多余维度，尽量压缩到 3D
    vol = np.squeeze(vol)
    if vol.ndim != 3:
        raise ValueError(f"当前体数据维度为 {vol.ndim}，期望为 3D 体数据。shape={vol.shape}")

    return vol


def normalize_for_display(
    vol: np.ndarray,
    clip_low: float = 0.5,
    clip_high: float = 99.5,
) -> np.ndarray:
    low, high = np.percentile(vol, [clip_low, clip_high])
    if high <= low:
        low, high = float(vol.min()), float(vol.max())
        if high <= low:
            return np.zeros_like(vol, dtype=np.float32)

    vol = np.clip(vol, low, high)
    vol = (vol - low) / (high - low + 1e-8)
    return vol


def get_slice_indices(
    vol: np.ndarray,
    axial: int | None,
    coronal: int | None,
    sagittal: int | None,
):
    x, y, z = vol.shape

    if sagittal is None:
        sagittal = x // 2
    if coronal is None:
        coronal = y // 2
    if axial is None:
        axial = z // 2

    sagittal = int(np.clip(sagittal, 0, x - 1))
    coronal = int(np.clip(coronal, 0, y - 1))
    axial = int(np.clip(axial, 0, z - 1))

    return axial, coronal, sagittal


def extract_views(vol: np.ndarray, axial: int, coronal: int, sagittal: int):
    # 注意:
    # vol shape = (X, Y, Z)
    # sagittal: 固定 X
    # coronal:  固定 Y
    # axial:    固定 Z
    sagittal_view = vol[sagittal, :, :]
    coronal_view = vol[:, coronal, :]
    axial_view = vol[:, :, axial]

    # 为了更符合肉眼观察习惯，做旋转
    sagittal_view = np.rot90(sagittal_view)
    coronal_view = np.rot90(coronal_view)
    axial_view = np.rot90(axial_view)

    return axial_view, coronal_view, sagittal_view


def visualize(
    vol: np.ndarray,
    axial: int | None = None,
    coronal: int | None = None,
    sagittal: int | None = None,
    cmap: str = "gray",
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    axial, coronal, sagittal = get_slice_indices(vol, axial, coronal, sagittal)
    axial_view, coronal_view, sagittal_view = extract_views(vol, axial, coronal, sagittal)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(axial_view, cmap=cmap)
    axes[0].set_title(f"Axial (z={axial})")
    axes[0].axis("off")

    axes[1].imshow(coronal_view, cmap=cmap)
    axes[1].set_title(f"Coronal (y={coronal})")
    axes[1].axis("off")

    axes[2].imshow(sagittal_view, cmap=cmap)
    axes[2].set_title(f"Sagittal (x={sagittal})")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path is not None:
        save_path = str(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] 可视化结果已保存到: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_argparser():
    parser = argparse.ArgumentParser(description="MRI 文件可视化脚本")
    parser.add_argument("--input", type=str, required=True, help="输入 MRI 文件路径 (.nii/.nii.gz/.npy)")
    parser.add_argument("--save", type=str, default=None, help="保存图片路径，如 ./out/view.png")
    parser.add_argument("--axial", type=int, default=None, help="轴状面索引 z")
    parser.add_argument("--coronal", type=int, default=None, help="冠状面索引 y")
    parser.add_argument("--sagittal", type=int, default=None, help="矢状面索引 x")
    parser.add_argument("--clip_low", type=float, default=0.5, help="显示强度下百分位")
    parser.add_argument("--clip_high", type=float, default=99.5, help="显示强度上百分位")
    parser.add_argument("--cmap", type=str, default="gray", help="matplotlib colormap")
    parser.add_argument("--no_show", action="store_true", help="只保存，不弹出窗口")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    vol = load_volume(str(input_path))
    vol_disp = normalize_for_display(vol, clip_low=args.clip_low, clip_high=args.clip_high)

    print(f"[INFO] 文件: {input_path}")
    print(f"[INFO] shape: {vol.shape}")
    print(f"[INFO] dtype: {vol.dtype}")
    print(f"[INFO] intensity range: ({float(vol.min()):.4f}, {float(vol.max()):.4f})")

    visualize(
        vol=vol_disp,
        axial=args.axial,
        coronal=args.coronal,
        sagittal=args.sagittal,
        cmap=args.cmap,
        title=input_path.name,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()

