#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv3-MRI 全身预训练数据准备脚本（增强版）。

功能：
  1. 从多来源 MRI 数据集中筛选可用的 3D 体数据
  2. 对每个体数据执行真实质量检查（读盘、维度、NaN/Inf、常数、spacing）
  3. 将非 NIfTI 格式（DICOM / .mhd / .npy / .h5）统一转换为 .nii.gz
  4. 按解剖区域组织到统一目录结构
  5. 应用数据集特异性规则（ACDC 取 ED、Amos 排 CT、MSD 过滤等）
  6. 生成完整的 manifest.csv 用于追踪每个样本的来源和质量

用法：
    # 先用 survey_datasets.py 探查数据
    python survey_datasets.py /data/datasets -o survey_report.md

    # dry_run 预览
    python prepare_all_datasets.py \\
        --data_root /data/datasets \\
        --output_root /data/pretrain_ready \\
        --dry_run

    # 正式执行（软链接模式，省磁盘空间）
    python prepare_all_datasets.py \\
        --data_root /data/datasets \\
        --output_root /data/pretrain_ready \\
        --mode symlink

    # 正式执行（复制模式，完全独立）
    python prepare_all_datasets.py \\
        --data_root /data/datasets \\
        --output_root /data/pretrain_ready \\
        --mode copy

输出目录结构：
    /data/pretrain_ready/
    +-- brain/
    +-- cardiac/
    +-- breast/
    +-- spine/
    +-- abdomen/
    +-- prostate/
    +-- knee/
    +-- other/
    +-- _converted/          (DICOM / 特殊格式转换产物)
    +-- manifest.csv         (全部样本的元数据清单)

依赖：
    pip install nibabel numpy pandas
    # 可选（按需安装）：
    pip install SimpleITK h5py
    # DICOM 转换需要：
    conda install -c conda-forge dcm2niix
"""

import os
import re
import sys
import csv
import glob
import shutil
import hashlib
import logging
import argparse
import subprocess
import traceback
from collections import OrderedDict

import numpy as np
import nibabel as nib

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("prepare_data")


# ===========================================================================
#  第一部分：质量检查
# ===========================================================================

# 质量检查的默认阈值
QC_MIN_DIM = 16          # 每个空间维度的最小尺寸（像素）
QC_MAX_DIM = 2048        # 每个空间维度的最大尺寸（防止异常大文件）
QC_MIN_SPACING = 0.1     # 最小合理 spacing（mm）
QC_MAX_SPACING = 20.0    # 最大合理 spacing（mm）
QC_MIN_STD = 1e-6        # 最小标准差（低于此值视为近常数图像）


def quality_check_volume(filepath, min_dim=QC_MIN_DIM):
    """对单个 NIfTI 文件执行全面质量检查。

    会真正读取体素数据到内存（不只是 header），检查以下项目：
      - 文件是否可被 nibabel 正常加载
      - 数据是否为 3D（4D 会在调用方通过数据集规则提前处理）
      - 是否包含 NaN 或 Inf
      - 是否近常数（std <= QC_MIN_STD）
      - 每个空间维度是否 >= min_dim
      - spacing（体素间距）是否存在且在合理范围内

    参数：
        filepath: NIfTI 文件路径
        min_dim:  每个空间维度的最小尺寸

    返回：
        (passed, info_dict, reason)
        - passed: bool, 是否通过质量检查
        - info_dict: dict, 包含 shape / spacing / mean / std / voxel_count
        - reason: str, 未通过时的原因（通过时为空字符串）
    """
    info = {
        "shape": "",
        "spacing": "",
        "voxel_count": 0,
        "mean_intensity": 0.0,
        "std_intensity": 0.0,
    }

    # --- 尝试加载 ---
    try:
        img = nib.load(filepath)
    except Exception as e:
        return False, info, "nibabel_load_failed: {}".format(str(e)[:120])

    # --- 读取 header 信息 ---
    try:
        header = img.header
        shape = img.shape
        info["shape"] = str(shape)
    except Exception as e:
        return False, info, "header_read_failed: {}".format(str(e)[:120])

    # --- 维度检查 ---
    if len(shape) < 3:
        return False, info, "ndim_too_low: ndim={}".format(len(shape))

    # 对于刚好 3D 的情况，直接检查空间维度
    spatial_shape = shape[:3]
    for i, s in enumerate(spatial_shape):
        if s < min_dim:
            return False, info, "dim_too_small: axis{}={} < {}".format(i, s, min_dim)
        if s > QC_MAX_DIM:
            return False, info, "dim_too_large: axis{}={} > {}".format(i, s, QC_MAX_DIM)

    # --- 真正读取体素数据 ---
    try:
        data = np.asarray(img.dataobj, dtype=np.float32)
    except Exception as e:
        return False, info, "data_read_failed: {}".format(str(e)[:120])

    # 如果是 4D，取第一个 volume
    if data.ndim == 4:
        data = data[:, :, :, 0]
    elif data.ndim > 4:
        return False, info, "ndim_too_high: ndim={}".format(data.ndim)

    info["voxel_count"] = int(np.prod(data.shape))

    # --- NaN / Inf 检查 ---
    if np.any(np.isnan(data)):
        return False, info, "contains_nan"
    if np.any(np.isinf(data)):
        return False, info, "contains_inf"

    # --- 常数图像检查 ---
    std_val = float(np.std(data))
    mean_val = float(np.mean(data))
    info["mean_intensity"] = round(mean_val, 4)
    info["std_intensity"] = round(std_val, 4)

    if std_val <= QC_MIN_STD:
        return False, info, "near_constant: std={:.2e}".format(std_val)

    # --- spacing 检查 ---
    try:
        pixdim = header.get_zooms()
        spacing = tuple(float(x) for x in pixdim[:3])
        info["spacing"] = str(spacing)

        for i, sp in enumerate(spacing):
            if sp <= 0:
                return False, info, "invalid_spacing: axis{}={:.4f}".format(i, sp)
            if sp < QC_MIN_SPACING:
                return False, info, "spacing_too_small: axis{}={:.4f}".format(i, sp)
            if sp > QC_MAX_SPACING:
                return False, info, "spacing_too_large: axis{}={:.4f}".format(i, sp)
    except Exception:
        # 有些文件没有有效的 spacing，给一个警告但不拒绝
        info["spacing"] = "unknown"

    return True, info, ""


# ===========================================================================
#  第二部分：格式转换
# ===========================================================================

def convert_dicom_to_nifti(dicom_dir, output_dir, prefix="converted"):
    """使用 dcm2niix 将 DICOM 目录转换为 NIfTI。

    参数：
        dicom_dir:  包含 DICOM 文件的目录
        output_dir: NIfTI 输出目录
        prefix:     输出文件名前缀

    返回：
        转换成功的 .nii.gz 文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "dcm2niix",
        "-z", "y",                          # 压缩输出
        "-f", "{}_{}".format(prefix, "%p_%s"),  # 前缀_协议_序列号
        "-o", output_dir,
        "-b", "n",                          # 不生成 BIDS sidecar
        dicom_dir,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            log.warning("dcm2niix stderr: %s", result.stderr[:300])
    except FileNotFoundError:
        log.error(
            "dcm2niix not installed! "
            "Run: conda install -c conda-forge dcm2niix"
        )
        return []
    except subprocess.TimeoutExpired:
        log.warning("dcm2niix timeout: %s", dicom_dir)
        return []
    except Exception as e:
        log.warning("dcm2niix error: %s", str(e)[:200])
        return []

    return sorted(glob.glob(os.path.join(output_dir, "*.nii.gz")))


def convert_mhd_to_nifti(filepath, output_path):
    """将 .mhd / .mha 转换为 .nii.gz（需要 SimpleITK）。

    返回：
        成功返回 output_path，失败返回 None
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        log.error("SimpleITK not installed: pip install SimpleITK")
        return None

    try:
        img = sitk.ReadImage(filepath)
        if img.GetDimension() < 3:
            log.warning("mhd ndim < 3: %s", filepath)
            return None
        sitk.WriteImage(img, output_path)
        return output_path
    except Exception as e:
        log.warning("mhd conversion failed: %s -> %s", filepath, str(e)[:120])
        return None


def convert_npy_to_nifti(filepath, output_path):
    """将 .npy 数组转换为 .nii.gz。

    仅保留 3D 数组；4D 取第一个 volume。

    返回：
        成功返回 output_path，失败返回 None
    """
    try:
        arr = np.load(filepath)
    except Exception as e:
        log.warning("npy load failed: %s -> %s", filepath, str(e)[:120])
        return None

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        log.warning("npy skip: ndim=%d, shape=%s, file=%s", arr.ndim, arr.shape, filepath)
        return None

    try:
        nii = nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4))
        nib.save(nii, output_path)
        return output_path
    except Exception as e:
        log.warning("npy->nifti save failed: %s", str(e)[:120])
        return None


def convert_h5_to_nifti(filepath, output_path, reconstruction_only=True):
    """将 HDF5 文件转换为 .nii.gz。

    优先读取重建图像（reconstruction_rss 等），排除 k-space 原始数据。

    参数：
        filepath:            .h5 文件路径
        output_path:         输出 .nii.gz 路径
        reconstruction_only: 为 True 时只读 reconstruction 类 key

    返回：
        成功返回 output_path，失败返回 None
    """
    try:
        import h5py
    except ImportError:
        log.error("h5py not installed: pip install h5py")
        return None

    # 定义优先级：先找重建图像，再找通用 key
    if reconstruction_only:
        candidate_keys = [
            "reconstruction_rss", "reconstruction_esc",
            "reconstruction", "image_rss",
        ]
    else:
        candidate_keys = [
            "reconstruction_rss", "reconstruction_esc",
            "reconstruction", "image_rss",
            "image", "data", "volume",
        ]

    try:
        with h5py.File(filepath, "r") as f:
            # 排除 kspace
            for key in candidate_keys:
                if key in f:
                    arr = np.array(f[key], dtype=np.float32)
                    if arr.ndim >= 3:
                        # 多余维度：逐层取第一个直到 3D
                        while arr.ndim > 3:
                            arr = arr[0]
                        nii = nib.Nifti1Image(arr, affine=np.eye(4))
                        nib.save(nii, output_path)
                        return output_path

        log.warning("h5 no valid key: %s (keys=%s)", filepath, "N/A")
        return None
    except Exception as e:
        log.warning("h5 conversion failed: %s -> %s", filepath, str(e)[:120])
        return None


def convert_special_format(filepath, output_path, dataset_name=""):
    """统一入口：根据后缀选择转换方式。

    支持 .mhd / .mha / .npy / .h5 / .hdf5。

    参数：
        filepath:     源文件路径
        output_path:  目标 .nii.gz 路径
        dataset_name: 数据集名称（用于 fastMRI 等特殊逻辑）

    返回：
        成功返回 output_path，失败返回 None
    """
    lower = filepath.lower()

    if lower.endswith((".mhd", ".mha")):
        return convert_mhd_to_nifti(filepath, output_path)

    if lower.endswith(".npy"):
        return convert_npy_to_nifti(filepath, output_path)

    if lower.endswith((".h5", ".hdf5")):
        # fastMRI 只要重建图像，排除 kspace
        recon_only = ("fastmri" in dataset_name.lower())
        return convert_h5_to_nifti(filepath, output_path, reconstruction_only=recon_only)

    log.warning("Unsupported format: %s", filepath)
    return None


# ===========================================================================
#  第三部分：数据集特异性规则
# ===========================================================================

def handle_4d_cardiac(filepath, output_dir, dataset_name, basename_prefix):
    """处理 4D 心脏 MRI：提取 ED 帧（通常是 frame 0）。

    ACDC / M&M 的 4D 文件包含一个完整心动周期的多帧。
    自监督预训练只需要单帧 3D volume。

    策略：
      - 尝试读取同目录下的 Info.cfg / *_Info.cfg 获取 ED 帧编号
      - 找不到 metadata 时，默认取 frame 0 并发出警告

    参数：
        filepath:        原始 4D NIfTI 路径
        output_dir:      转换输出目录
        dataset_name:    数据集名
        basename_prefix: 输出文件名前缀

    返回：
        提取出的 3D .nii.gz 路径，失败返回 None
    """
    try:
        img = nib.load(filepath)
        data = np.asarray(img.dataobj, dtype=np.float32)
    except Exception as e:
        log.warning("[%s] 4D load failed: %s -> %s", dataset_name, filepath, str(e)[:120])
        return None

    if data.ndim != 4:
        # 不是 4D，直接返回原文件
        return filepath

    n_frames = data.shape[3]
    if n_frames < 2:
        # 只有 1 帧，等价于 3D
        return filepath

    # 尝试从 Info.cfg 读取 ED 帧号
    ed_frame = 0
    parent_dir = os.path.dirname(filepath)
    info_candidates = glob.glob(os.path.join(parent_dir, "*Info*.cfg"))
    info_candidates += glob.glob(os.path.join(parent_dir, "*info*.cfg"))

    ed_found = False
    for cfg_path in info_candidates:
        try:
            with open(cfg_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.lower().startswith("ed:") or line.lower().startswith("ed ="):
                        val = line.split(":")[-1].split("=")[-1].strip()
                        ed_frame = int(val)
                        ed_found = True
                        break
            if ed_found:
                break
        except Exception:
            continue

    if not ed_found:
        log.warning(
            "[%s] No ED metadata for 4D file, using frame 0: %s (n_frames=%d)",
            dataset_name, os.path.basename(filepath), n_frames,
        )

    if ed_frame >= n_frames:
        log.warning(
            "[%s] ED frame %d >= n_frames %d, using frame 0: %s",
            dataset_name, ed_frame, n_frames, os.path.basename(filepath),
        )
        ed_frame = 0

    # 提取单帧
    try:
        frame_data = data[:, :, :, ed_frame]
        frame_img = nib.Nifti1Image(frame_data, img.affine, img.header)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(
            output_dir,
            "{}_frame{:02d}.nii.gz".format(basename_prefix, ed_frame),
        )
        nib.save(frame_img, out_path)
        return out_path
    except Exception as e:
        log.warning("[%s] Frame extraction failed: %s", dataset_name, str(e)[:120])
        return None


def filter_amos_mri_only(filepath):
    """Amos 数据集：仅保留 MRI，排除 CT。

    Amos 约定：文件编号 >= 500 为 MRI，< 500 为 CT。
    例如 amos_0501.nii.gz 是 MRI，amos_0001.nii.gz 是 CT。

    返回：
        True = 保留（MRI），False = 排除（CT）
    """
    basename = os.path.basename(filepath)
    match = re.search(r"(\d{4})", basename)
    if match:
        num = int(match.group(1))
        return num >= 500
    # 无法判断编号时，保守保留
    return True


def filter_msd_mri_tasks(filepath):
    """Medical Segmentation Decathlon：仅保留 MRI 相关任务的 imagesTr。

    MSD 中的 MRI 任务：
      - Task01_BrainTumour (多模态脑 MRI)
      - Task02_Heart       (心脏 MRI)
      - Task05_Prostate     (前列腺 MRI)

    CT 任务（排除）：
      - Task03_Liver, Task06_Lung, Task07_Pancreas,
        Task08_HepaticVessel, Task09_Spleen, Task10_Colon

    同时只保留 imagesTr 目录下的文件。

    返回：
        True = 保留，False = 排除
    """
    rel = filepath.replace("\\", "/")

    # 必须在 imagesTr 目录下
    if "/imagesTr/" not in rel and "/imagesTs/" not in rel:
        return False

    # MRI 任务白名单
    mri_tasks = ["Task01", "Task02", "Task05"]
    for task in mri_tasks:
        if task in rel:
            return True

    return False


def filter_prostatex_t2(filepath):
    """PROSTATEx / LLD-MMRI：仅保留 T2 序列。

    通过文件名或目录名中的 t2 关键词判断。

    返回：
        True = 保留（T2），False = 排除
    """
    combined = filepath.lower()
    # 匹配 t2, t2w, T2, T2W, T2_TSE 等
    if re.search(r"(?:^|[_\-./\\])t2", combined):
        return True
    return False


# 全局排除模式：所有数据集共用，排除 mask / seg / label / gt
GLOBAL_EXCLUDE_PATTERN = re.compile(
    r"(mask|seg[^m]|segmentation|label|_gt[_./]|ground.?truth|annotation|contour)",
    re.IGNORECASE,
)


def is_globally_excluded(filepath):
    """检查文件是否匹配全局排除规则（mask / label / segmentation 等）。

    除了文件名本身，也检查上两级目录名。
    """
    basename = os.path.basename(filepath).lower()
    parent = os.path.basename(os.path.dirname(filepath)).lower()
    grandparent = os.path.basename(
        os.path.dirname(os.path.dirname(filepath))
    ).lower()
    combined = "/".join([grandparent, parent, basename])

    return bool(GLOBAL_EXCLUDE_PATTERN.search(combined))


# ===========================================================================
#  第四部分：稳定命名
# ===========================================================================

def stable_output_name(dataset_name, source_path, max_len=180):
    """生成稳定、唯一、可复现的输出文件名。

    格式：{safe_dataset}_{md5_8chars}_{basename}
    - safe_dataset: 数据集名（去掉特殊字符）
    - md5_8chars:   source_path 的 MD5 前 8 位（保证唯一性）
    - basename:     原始文件名

    总长度超过 max_len 时会截断 basename。

    参数：
        dataset_name: 数据集名
        source_path:  源文件的绝对路径
        max_len:      文件名最大长度

    返回：
        稳定的文件名字符串（以 .nii.gz 结尾）
    """
    # 安全的数据集名：替换特殊字符
    safe_ds = re.sub(r"[^a-zA-Z0-9_\-]", "_", dataset_name)
    # 避免连续下划线
    safe_ds = re.sub(r"_+", "_", safe_ds).strip("_")
    # 截断过长的数据集名
    if len(safe_ds) > 40:
        safe_ds = safe_ds[:40]

    # 基于完整路径的 MD5（稳定、可复现）
    path_hash = hashlib.md5(source_path.encode("utf-8")).hexdigest()[:8]

    # 原始文件名
    basename = os.path.basename(source_path)
    # 去掉 .nii.gz 后缀以便重新拼接
    if basename.lower().endswith(".nii.gz"):
        stem = basename[:-7]
        suffix = ".nii.gz"
    elif basename.lower().endswith(".nii"):
        stem = basename[:-4]
        suffix = ".nii.gz"  # 统一压缩
    else:
        stem = os.path.splitext(basename)[0]
        suffix = ".nii.gz"

    # 拼接
    name = "{}_{}_{}{}".format(safe_ds, path_hash, stem, suffix)

    # 超长截断
    if len(name) > max_len:
        avail = max_len - len(safe_ds) - len(path_hash) - len(suffix) - 3  # 3 个下划线
        if avail < 8:
            avail = 8
        stem = stem[:avail]
        name = "{}_{}_{}{}".format(safe_ds, path_hash, stem, suffix)

    return name


# ===========================================================================
#  第五部分：文件发现
# ===========================================================================

# 受支持的特殊格式后缀
SPECIAL_EXTENSIONS = (".mhd", ".mha", ".npy", ".h5", ".hdf5")


def discover_files(data_dir, include_pattern=None, exclude_pattern=None,
                   include_special=False):
    """递归查找可用的医学影像文件。

    参数：
        data_dir:         搜索根目录
        include_pattern:  正则表达式，只保留匹配的文件路径
        exclude_pattern:  正则表达式，排除匹配的文件路径
        include_special:  是否同时搜索特殊格式（.mhd/.npy/.h5 等）

    返回：
        排序后的文件路径列表
    """
    nifti_exts = (".nii.gz", ".nii")
    all_exts = nifti_exts + SPECIAL_EXTENSIONS if include_special else nifti_exts

    files = []
    for root, _, fnames in os.walk(data_dir, followlinks=True):
        for fname in fnames:
            lower = fname.lower()
            matched = False
            for ext in all_exts:
                if lower.endswith(ext):
                    matched = True
                    break
            if not matched:
                continue

            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, data_dir)

            # 应用 include 过滤
            if include_pattern:
                if not re.search(include_pattern, rel, re.IGNORECASE):
                    continue

            # 应用 exclude 过滤
            if exclude_pattern:
                if re.search(exclude_pattern, rel, re.IGNORECASE):
                    continue

            # 全局排除
            if is_globally_excluded(fpath):
                continue

            files.append(fpath)

    return sorted(files)


def discover_dicom_series(data_dir):
    """查找所有可能的 DICOM 系列目录。

    判断标准：目录中超过一半的文件是 .dcm 或无后缀文件。

    返回：
        DICOM 系列目录路径列表
    """
    series_dirs = []
    for root, _dirs, files in os.walk(data_dir, followlinks=True):
        if not files:
            continue
        dcm_count = 0
        file_count = 0
        for f in files:
            full = os.path.join(root, f)
            if not os.path.isfile(full):
                continue
            file_count += 1
            lower = f.lower()
            if lower.endswith(".dcm") or ("." not in f and f != "DICOMDIR"):
                dcm_count += 1
        if dcm_count >= 5 and file_count > 0 and dcm_count > file_count * 0.5:
            series_dirs.append(root)

    return sorted(series_dirs)


# ===========================================================================
#  第六部分：数据集注册表
# ===========================================================================

DATASET_REGISTRY = OrderedDict([

    # ===================== 脑部 =====================
    ("IXI", {
        "region": "brain",
        "format": "nifti",
        "include_pattern": r"T1\.nii\.gz$",
        "exclude_pattern": None,
        "notes": "IXI T1, ~600 volumes",
    }),
    ("OASIS", {
        "region": "brain",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": r"(atlas)",
        "notes": "OASIS brain MRI",
    }),
    ("ABIDE1-MRI", {
        "region": "brain",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "notes": "ABIDE-I brain",
    }),
    ("ABIDE2", {
        "region": "brain",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "notes": "ABIDE-II brain",
    }),
    ("atlas-train-dataset-1.0.1", {
        "region": "brain",
        "format": "nifti",
        "include_pattern": r"T1w.*\.nii(\.gz)?$",
        "exclude_pattern": r"(lesion)",
        "notes": "ATLAS stroke T1",
    }),
    ("MR Brain Segmentation", {
        "region": "brain",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "notes": "MR Brain Seg",
    }),
    ("Openmind", {
        "region": "brain",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "notes": "OpenMind brain",
    }),
    ("brain-tumour-MRI-scan", {
        "region": "brain",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "notes": "Brain tumour MRI",
    }),
    ("PKG-Vestibular-Schwannoma-MC-RC2_Oct2025", {
        "region": "brain",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "notes": "Vestibular schwannoma",
    }),
    ("ISLES", {
        "region": "brain",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": r"(lesion)",
        "notes": "Ischemic stroke",
    }),

    # ===================== 心脏 =====================
    ("ACDC", {
        "region": "cardiac",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "special_rule": "cardiac_4d",  # 4D 提取 ED 帧
        "notes": "ACDC cardiac, 4D->ED frame",
    }),
    ("211230_M&Ms_Dataset_information_diagnosis_opendataset", {
        "region": "cardiac",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "special_rule": "cardiac_4d",
        "notes": "M&Ms cardiac, 4D->ED frame",
    }),
    ("M&M", {
        "region": "cardiac",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "special_rule": "cardiac_4d",
        "notes": "M&M cardiac, 4D->ED frame",
    }),

    # ===================== 乳腺（DICOM） =====================
    ("Duke-Breast-Cancer-MRI", {
        "region": "breast",
        "format": "dicom",
        "notes": "Duke breast cancer DCE-MRI",
    }),
    ("QIN_Breast_DCE-MRI", {
        "region": "breast",
        "format": "dicom",
        "notes": "QIN breast DCE-MRI",
    }),
    ("RIDER_Breast_MRI", {
        "region": "breast",
        "format": "dicom",
        "notes": "RIDER breast MRI",
    }),

    # ===================== 脊柱 =====================
    ("lumbar-spine-mri", {
        "region": "spine",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "notes": "Lumbar spine MRI",
    }),
    ("Lumbar Spine MRI Dataset", {
        "region": "spine",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "notes": "Lumbar spine MRI",
    }),
    ("An open-access lumbosacral spine MRI dataset with enhanced spinal "
     "nerve root structure resolution", {
        "region": "spine",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "notes": "Lumbosacral spine MRI",
    }),
    ("LSpineSMRI A Comprehensive Dataset of Non-Contrast Lumbar Spine "
     "Stenosis MRI Examinations", {
        "region": "spine",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "notes": "Lumbar stenosis MRI",
    }),

    # ===================== 前列腺 =====================
    ("PROMISE12_MICCAI", {
        "region": "prostate",
        "format": "nifti",
        "include_pattern": r"\.(nii(\.gz)?|mhd|mha)$",
        "exclude_pattern": None,
        "include_special": True,  # 搜索 .mhd
        "notes": "PROMISE12 prostate (may have .mhd)",
    }),
    ("PROSTATEx_ClinSig_Strict4ch", {
        "region": "prostate",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "special_rule": "t2_only",
        "notes": "PROSTATEx T2 only",
    }),

    # ===================== 腹部 =====================
    ("Amos", {
        "region": "abdomen",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "special_rule": "amos_mri",  # 排除 CT
        "notes": "AMOS abdomen, MRI only (id >= 500)",
    }),
    ("TCGA-LIHC", {
        "region": "abdomen",
        "format": "dicom",
        "notes": "TCGA liver (DICOM, may contain CT)",
    }),
    ("Medical_Segmentation_Decathlon", {
        "region": "abdomen",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "special_rule": "msd_mri",  # 只保留 MRI 任务的 imagesTr
        "notes": "MSD MRI tasks only (Task01/02/05, imagesTr)",
    }),
    ("LLD-MMRI", {
        "region": "abdomen",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "special_rule": "t2_only",
        "notes": "Liver multi-seq MRI, T2 only",
    }),

    # ===================== 膝关节 =====================
    ("MRNet-v1.0", {
        "region": "knee",
        "format": "nifti",
        "include_pattern": r"\.(nii(\.gz)?|npy)$",
        "exclude_pattern": None,
        "include_special": True,  # 搜索 .npy
        "notes": "MRNet knee (may be .npy)",
    }),
    ("fastMRI", {
        "region": "knee",
        "format": "nifti",
        "include_pattern": r"\.(nii(\.gz)?|h5|hdf5)$",
        "exclude_pattern": None,
        "include_special": True,
        "special_rule": "fastmri",  # 只保留重建图像
        "notes": "fastMRI knee, reconstruction only",
    }),

    # ===================== 其他 =====================
    ("MRI3DVLM", {
        "region": "other",
        "format": "nifti",
        "include_pattern": r"\.nii(\.gz)?$",
        "exclude_pattern": None,
        "notes": "3D VLM data",
    }),

    # ===================== 跳过 =====================
    ("MedFrameQA",      {"format": "skip", "notes": "VQA, not volumetric"}),
    ("MRI_512-VQA",     {"format": "skip", "notes": "VQA dataset"}),
    ("RadImageNet-VQA", {"format": "skip", "notes": "VQA dataset"}),
    ("VQA-RAD",         {"format": "skip", "notes": "VQA dataset"}),
    ("SLAKE",           {"format": "skip", "notes": "VQA dataset"}),
    ("MedPix-2.0",      {"format": "skip", "notes": "Teaching dataset"}),
    ("TrackRAD2025",    {"format": "skip", "notes": "Radiotherapy tracking"}),
    ("BRISC_2025",      {"format": "skip", "notes": "Check manually"}),
    ("archive",         {"format": "skip", "notes": "Check content"}),
    ("_TASK",           {"format": "skip", "notes": "Check content"}),
    ("manifest",        {"format": "skip", "notes": "Metadata only"}),
])


# ===========================================================================
#  第七部分：核心处理流程
# ===========================================================================

def apply_special_rule(filepath, rule, dataset_name, convert_dir):
    """应用数据集特异性规则。

    参数：
        filepath:     当前文件路径
        rule:         规则名称（来自注册表的 special_rule 字段）
        dataset_name: 数据集名
        convert_dir:  转换输出目录

    返回：
        (keep, actual_path)
        - keep: 是否保留该文件
        - actual_path: 实际要使用的文件路径（可能是转换后的新路径）
    """
    if rule == "cardiac_4d":
        # 心脏 4D：检查是否为 4D，若是则提取 ED 帧
        try:
            img = nib.load(filepath)
            if img.ndim == 4 and img.shape[3] > 1:
                prefix = stable_output_name(dataset_name, filepath)[:-7]  # 去掉 .nii.gz
                result = handle_4d_cardiac(
                    filepath, convert_dir, dataset_name, prefix
                )
                if result is None:
                    return False, filepath
                if result != filepath:
                    return True, result
            # 3D 或单帧 4D，直接保留
            return True, filepath
        except Exception:
            return True, filepath

    if rule == "amos_mri":
        return filter_amos_mri_only(filepath), filepath

    if rule == "msd_mri":
        return filter_msd_mri_tasks(filepath), filepath

    if rule == "t2_only":
        return filter_prostatex_t2(filepath), filepath

    if rule == "fastmri":
        # fastMRI 的 NIfTI 文件直接保留；H5 文件在格式转换阶段处理
        # 这里只需排除 kspace 文件名
        lower = os.path.basename(filepath).lower()
        if "kspace" in lower:
            return False, filepath
        return True, filepath

    # 未知规则，默认保留
    return True, filepath


def process_single_file(source_path, dataset_name, config, output_dir,
                        convert_dir, mode, manifest_rows):
    """处理单个文件：特殊规则 -> 格式转换 -> 质量检查 -> 输出。

    这是整个处理流程中最核心的函数，对每个文件依次执行：
      1. 数据集特异性规则过滤
      2. 如果是非 NIfTI 格式，执行格式转换
      3. 质量检查
      4. 写入输出目录（symlink 或 copy）
      5. 记录到 manifest

    参数：
        source_path:   源文件路径
        dataset_name:  数据集名
        config:        该数据集的配置字典
        output_dir:    目标区域目录（如 output_root/brain/）
        convert_dir:   格式转换临时目录
        mode:          "symlink" 或 "copy"
        manifest_rows: manifest 行列表（就地追加）

    返回：
        True = 成功写入，False = 跳过
    """
    region = config.get("region", "other")
    original_format = "nifti"
    lower = source_path.lower()
    for ext in SPECIAL_EXTENSIONS:
        if lower.endswith(ext):
            original_format = ext.lstrip(".")
            break
    if config.get("format") == "dicom":
        original_format = "dicom"

    # --- 1. 数据集特异性规则 ---
    special_rule = config.get("special_rule")
    actual_path = source_path
    if special_rule:
        keep, actual_path = apply_special_rule(
            source_path, special_rule, dataset_name, convert_dir
        )
        if not keep:
            manifest_rows.append({
                "output_path": "",
                "source_path": source_path,
                "dataset_name": dataset_name,
                "region": region,
                "original_format": original_format,
                "final_format": "",
                "shape": "",
                "spacing": "",
                "voxel_count": 0,
                "mean_intensity": 0.0,
                "std_intensity": 0.0,
                "passed_quality_check": False,
                "skip_reason": "filtered_by_rule: {}".format(special_rule),
            })
            return False

    # --- 2. 格式转换（非 NIfTI） ---
    needs_conversion = False
    for ext in SPECIAL_EXTENSIONS:
        if actual_path.lower().endswith(ext):
            needs_conversion = True
            break

    if needs_conversion:
        out_name = stable_output_name(dataset_name, source_path)
        converted_path = os.path.join(convert_dir, out_name)
        os.makedirs(os.path.dirname(converted_path), exist_ok=True)
        result = convert_special_format(actual_path, converted_path, dataset_name)
        if result is None:
            manifest_rows.append({
                "output_path": "",
                "source_path": source_path,
                "dataset_name": dataset_name,
                "region": region,
                "original_format": original_format,
                "final_format": "",
                "shape": "",
                "spacing": "",
                "voxel_count": 0,
                "mean_intensity": 0.0,
                "std_intensity": 0.0,
                "passed_quality_check": False,
                "skip_reason": "conversion_failed",
            })
            return False
        actual_path = result

    # --- 3. 质量检查 ---
    passed, info, reason = quality_check_volume(actual_path)

    if not passed:
        manifest_rows.append({
            "output_path": "",
            "source_path": source_path,
            "dataset_name": dataset_name,
            "region": region,
            "original_format": original_format,
            "final_format": "nii.gz",
            "shape": info["shape"],
            "spacing": info["spacing"],
            "voxel_count": info["voxel_count"],
            "mean_intensity": info["mean_intensity"],
            "std_intensity": info["std_intensity"],
            "passed_quality_check": False,
            "skip_reason": reason,
        })
        return False

    # --- 4. 写入输出目录 ---
    out_name = stable_output_name(dataset_name, source_path)
    dest = os.path.join(output_dir, out_name)

    if os.path.exists(dest):
        # 文件已存在，跳过（幂等）
        return False

    try:
        if mode == "symlink":
            os.symlink(os.path.abspath(actual_path), dest)
        else:
            shutil.copy2(actual_path, dest)
    except Exception as e:
        log.warning("Output write failed: %s -> %s", dest, str(e)[:120])
        manifest_rows.append({
            "output_path": "",
            "source_path": source_path,
            "dataset_name": dataset_name,
            "region": region,
            "original_format": original_format,
            "final_format": "nii.gz",
            "shape": info["shape"],
            "spacing": info["spacing"],
            "voxel_count": info["voxel_count"],
            "mean_intensity": info["mean_intensity"],
            "std_intensity": info["std_intensity"],
            "passed_quality_check": False,
            "skip_reason": "write_failed: {}".format(str(e)[:80]),
        })
        return False

    # --- 5. 记录 manifest ---
    manifest_rows.append({
        "output_path": dest,
        "source_path": source_path,
        "dataset_name": dataset_name,
        "region": region,
        "original_format": original_format,
        "final_format": "nii.gz",
        "shape": info["shape"],
        "spacing": info["spacing"],
        "voxel_count": info["voxel_count"],
        "mean_intensity": info["mean_intensity"],
        "std_intensity": info["std_intensity"],
        "passed_quality_check": True,
        "skip_reason": "",
    })
    return True


def prepare_dataset(dataset_name, config, data_root, output_root,
                    mode, manifest_rows, dry_run=False):
    """处理单个数据集的全部文件。

    参数：
        dataset_name:  数据集名
        config:        注册表配置
        data_root:     数据根目录
        output_root:   输出根目录
        mode:          "symlink" 或 "copy"
        manifest_rows: manifest 行列表
        dry_run:       仅统计不执行

    返回：
        成功写入的文件数
    """
    if config.get("format") == "skip":
        log.info("  SKIP: %s", config.get("notes", ""))
        return 0

    region = config.get("region", "other")
    data_dir = os.path.join(data_root, dataset_name)

    if not os.path.isdir(data_dir):
        log.warning("  Directory not found: %s", data_dir)
        return 0

    output_dir = os.path.join(output_root, region)
    convert_dir = os.path.join(output_root, "_converted", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    count = 0

    if config.get("format") == "dicom":
        # ---- DICOM 模式 ----
        series_dirs = discover_dicom_series(data_dir)
        log.info("  Found %d DICOM series", len(series_dirs))

        if dry_run:
            return len(series_dirs)

        os.makedirs(convert_dir, exist_ok=True)
        for i, sdir in enumerate(series_dirs):
            safe_ds = re.sub(r"[^a-zA-Z0-9_\-]", "_", dataset_name)[:30]
            prefix = "{}_s{:04d}".format(safe_ds, i)
            nifti_files = convert_dicom_to_nifti(sdir, convert_dir, prefix=prefix)

            for nf in nifti_files:
                try:
                    ok = process_single_file(
                        nf, dataset_name, config, output_dir,
                        convert_dir, mode, manifest_rows,
                    )
                    if ok:
                        count += 1
                except Exception as e:
                    log.warning("  Error processing DICOM output %s: %s",
                                nf, str(e)[:120])
    else:
        # ---- NIfTI / 特殊格式 模式 ----
        include_special = config.get("include_special", False)
        files = discover_files(
            data_dir,
            include_pattern=config.get("include_pattern"),
            exclude_pattern=config.get("exclude_pattern"),
            include_special=include_special,
        )
        log.info("  Found %d candidate files", len(files))

        if dry_run:
            return len(files)

        for fpath in files:
            try:
                ok = process_single_file(
                    fpath, dataset_name, config, output_dir,
                    convert_dir, mode, manifest_rows,
                )
                if ok:
                    count += 1
            except Exception as e:
                log.warning("  Error processing %s: %s", fpath, str(e)[:120])
                manifest_rows.append({
                    "output_path": "",
                    "source_path": fpath,
                    "dataset_name": dataset_name,
                    "region": region,
                    "original_format": "unknown",
                    "final_format": "",
                    "shape": "",
                    "spacing": "",
                    "voxel_count": 0,
                    "mean_intensity": 0.0,
                    "std_intensity": 0.0,
                    "passed_quality_check": False,
                    "skip_reason": "unhandled_exception: {}".format(str(e)[:80]),
                })

    return count


# ===========================================================================
#  第八部分：Manifest 输出
# ===========================================================================

MANIFEST_FIELDS = [
    "output_path",
    "source_path",
    "dataset_name",
    "region",
    "original_format",
    "final_format",
    "shape",
    "spacing",
    "voxel_count",
    "mean_intensity",
    "std_intensity",
    "passed_quality_check",
    "skip_reason",
]


def write_manifest(manifest_rows, output_path):
    """将 manifest 写入 CSV 文件。

    使用标准库 csv 而非 pandas，减少依赖。

    参数：
        manifest_rows: 字典列表
        output_path:   输出 CSV 路径
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    log.info("Manifest saved: %s (%d rows)", output_path, len(manifest_rows))


def print_summary(manifest_rows):
    """打印处理结果汇总统计。"""
    total = len(manifest_rows)
    passed = sum(1 for r in manifest_rows if r["passed_quality_check"])
    failed = total - passed

    # 按区域统计
    region_stats = {}
    for r in manifest_rows:
        region = r["region"]
        if region not in region_stats:
            region_stats[region] = {"passed": 0, "failed": 0}
        if r["passed_quality_check"]:
            region_stats[region]["passed"] += 1
        else:
            region_stats[region]["failed"] += 1

    # 按失败原因统计
    reason_stats = {}
    for r in manifest_rows:
        if not r["passed_quality_check"] and r["skip_reason"]:
            # 取冒号前的主原因
            main_reason = r["skip_reason"].split(":")[0]
            reason_stats[main_reason] = reason_stats.get(main_reason, 0) + 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Total samples scanned : {:>6d}".format(total))
    print("Passed quality check  : {:>6d}".format(passed))
    print("Failed / skipped      : {:>6d}".format(failed))
    print()
    print("By region:")
    print("  {:20s} {:>8s} {:>8s}".format("Region", "Passed", "Failed"))
    print("  " + "-" * 38)
    for region in sorted(region_stats.keys()):
        s = region_stats[region]
        print("  {:20s} {:>8d} {:>8d}".format(region, s["passed"], s["failed"]))
    print()

    if reason_stats:
        print("Failure reasons:")
        for reason, cnt in sorted(reason_stats.items(), key=lambda x: -x[1]):
            print("  {:40s} {:>6d}".format(reason, cnt))
    print("=" * 70)


# ===========================================================================
#  第九部分：主入口
# ===========================================================================

def main():
    global QC_MIN_DIM

    parser = argparse.ArgumentParser(
        description="DINOv3-MRI data preparation with quality checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Dry run\n"
            "  python prepare_all_datasets.py \\\n"
            "      --data_root /data/datasets --output_root /data/ready --dry_run\n"
            "\n"
            "  # Symlink mode (saves disk space)\n"
            "  python prepare_all_datasets.py \\\n"
            "      --data_root /data/datasets --output_root /data/ready --mode symlink\n"
            "\n"
            "  # Process specific datasets only\n"
            "  python prepare_all_datasets.py \\\n"
            "      --data_root /data/datasets --output_root /data/ready \\\n"
            "      --datasets IXI OASIS ACDC\n"
        ),
    )
    parser.add_argument(
        "--data_root", default="/home2/Data",
        help="Raw datasets root directory",
    )
    parser.add_argument(
        "--output_root", default="/home2/Data_processed",
        help="Output directory for organized data",
    )
    parser.add_argument(
        "--mode", choices=["symlink", "copy"], default="symlink",
        help="symlink = save disk, copy = fully independent (default: symlink)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Count only, do not write files",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Process only these datasets (default: all in registry)",
    )
    parser.add_argument(
        "--min_dim", type=int, default=QC_MIN_DIM,
        help="Minimum dimension per axis (default: {})".format(QC_MIN_DIM),
    )
    args = parser.parse_args()

    QC_MIN_DIM = args.min_dim

    log.info("Data root   : %s", args.data_root)
    log.info("Output root : %s", args.output_root)
    log.info("Mode        : %s", args.mode)
    log.info("Min dim     : %d", args.min_dim)
    if args.dry_run:
        log.info("*** DRY RUN - no files will be written ***")
    print("=" * 70 + "\n")

    os.makedirs(args.output_root, exist_ok=True)

    manifest_rows = []
    total_count = 0

    names_to_process = args.datasets if args.datasets else list(DATASET_REGISTRY.keys())

    for name in names_to_process:
        if name not in DATASET_REGISTRY:
            log.warning("[%s] Not in registry, skipping", name)
            continue

        config = DATASET_REGISTRY[name]
        log.info("[%s]", name)

        n = prepare_dataset(
            dataset_name=name,
            config=config,
            data_root=args.data_root,
            output_root=args.output_root,
            mode=args.mode,
            manifest_rows=manifest_rows,
            dry_run=args.dry_run,
        )
        total_count += n
        log.info("  -> %d files written\n", n)

    # 写入 manifest
    if not args.dry_run:
        manifest_path = os.path.join(args.output_root, "manifest.csv")
        write_manifest(manifest_rows, manifest_path)

    # 打印汇总
    if manifest_rows:
        print_summary(manifest_rows)
    else:
        if args.dry_run:
            print("\nDry run total: {} candidate files".format(total_count))
        else:
            print("\nNo files processed.")


if __name__ == "__main__":
    main()
