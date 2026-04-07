#!/usr/bin/env python3
"""
数据集准备脚本：筛选、转换、验证、整理多来源 MRI 数据。

将多个原始数据集处理为统一的目录结构，供 DINOv3 全身 MRI 预训练使用。

处理流程：
  1. 遍历每个数据源，按数据集类型执行特定筛选规则
  2. DICOM 数据集自动调用 dcm2niix 转换为 NIfTI
  3. 对每个 NIfTI 文件执行质量检查
  4. 将通过检查的文件按区域组织到统一输出目录
  5. 生成数据清单 CSV 文件

用法：
    python scripts/prepare_datasets.py \\
        --sources \\
            IXI=/home2/Data/IXI \\
            ABIDE1=/home2/Data/ABIDE1-MRI \\
            ABIDE2=/home2/Data/ABIDE2 \\
            OASIS=/home2/Data/OASIS \\
            ACDC=/home2/Data/ACDC \\
            MnM=/home2/Data/M\\&M \\
            Amos=/home2/Data/Amos \\
            MSD=/home2/Data/Medical_Segmentation_Decathlon \\
            LLD_MMRI=/home2/Data/LLD-MMRI \\
            ISLES=/home2/Data/ISLES \\
            PROMISE12=/home2/Data/PROMISE12_MICCAI \\
            PROSTATEx=/home2/Data/PROSTATEx_ClinSig_Strict4ch \\
            MRNet=/home2/Data/MRNet-v1.0 \\
            fastMRI=/home2/Data/fastMRI \\
            LSpine=/home2/Data/lumbar-spine-mri \\
            LSpineSMRI=/home2/Data/LSpineSMRI \\
            Duke_Breast=/home2/Data/Duke-Breast-Cancer-MRI \\
            MR_Brain_Seg=/home2/Data/MR\\ Brain\\ Segmentation \\
        --output_dir /home2/Data/wholebody \\
        --spacing 1.0 1.0 1.0 \\
        --min_dim 48 \\
        --workers 8

    # 仅扫描并统计，不执行复制
    python scripts/prepare_datasets.py \\
        --sources IXI=/home2/Data/IXI \\
        --output_dir /home2/Data/wholebody \\
        --dry_run
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

# 添加项目根目录到 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessing import quality_check_volume


# ---------------------------------------------------------------------------
# 数据集 → 解剖区域映射
# ---------------------------------------------------------------------------

DATASET_REGION_MAP = {
    # 脑部
    "IXI": "brain",
    "ABIDE1": "brain",
    "ABIDE2": "brain",
    "OASIS": "brain",
    "ISLES": "brain",
    "MR_Brain_Seg": "brain",
    "fastMRI_brain": "brain",
    "Openmind": "brain",
    "PKG_VS": "brain",
    # 心脏
    "ACDC": "cardiac",
    "MnM": "cardiac",
    # 腹部 / 肝脏
    "Amos": "abdomen",
    "LLD_MMRI": "abdomen",
    "TCGA_LIHC": "abdomen",
    # 前列腺
    "PROMISE12": "prostate",
    "PROSTATEx": "prostate",
    # 膝关节
    "MRNet": "knee",
    "fastMRI_knee": "knee",
    # 脊柱
    "LSpine": "spine",
    "LSpineSMRI": "spine",
    "Lumbosacral": "spine",
    # 乳腺
    "Duke_Breast": "breast",
    "QIN_Breast": "breast",
    "RIDER_Breast": "breast",
    # MSD 子任务
    "MSD_BrainTumour": "brain",
    "MSD_Heart": "cardiac",
    "MSD_Prostate": "prostate",
    "MSD_Liver": "abdomen",
}

# 通用默认区域（无法识别时使用）
DEFAULT_REGION = "other"


# ---------------------------------------------------------------------------
# 数据集特定筛选器
# ---------------------------------------------------------------------------

def _find_nifti_recursive(data_dir: str) -> List[str]:
    """递归查找所有 NIfTI 文件。"""
    files = []
    for root, _dirs, fnames in os.walk(data_dir, followlinks=True):
        for fname in fnames:
            if fname.lower().endswith((".nii.gz", ".nii")):
                files.append(os.path.join(root, fname))
    return sorted(files)


def filter_generic(data_dir: str, dataset_name: str) -> List[str]:
    """通用筛选器：返回所有 NIfTI 文件。"""
    return _find_nifti_recursive(data_dir)


def filter_acdc(data_dir: str, dataset_name: str) -> List[str]:
    """ACDC 筛选器：仅选择舒张末期（ED）帧。

    ACDC 目录结构：
      patient001/
        patient001_4d.nii.gz     # 完整 4D cine（跳过）
        patient001_frame01.nii.gz  # ED 帧
        patient001_frame12.nii.gz  # ES 帧
        Info.cfg                   # 包含 ED/ES 帧号

    策略：读取 Info.cfg 获取 ED 帧号，选择对应的 frame 文件。
    """
    selected = []
    all_files = _find_nifti_recursive(data_dir)

    # 查找所有 Info.cfg 以确定 ED 帧号
    ed_frames = {}  # patient_dir -> ed_frame_filename
    for root, _dirs, fnames in os.walk(data_dir):
        if "Info.cfg" in fnames:
            cfg_path = os.path.join(root, "Info.cfg")
            try:
                with open(cfg_path, "r") as f:
                    for line in f:
                        if line.strip().startswith("ED:"):
                            ed_num = int(line.strip().split(":")[1].strip())
                            ed_frames[root] = f"frame{ed_num:02d}"
                            break
            except Exception:
                pass

    if ed_frames:
        # 有 Info.cfg，精确选择 ED 帧
        for fpath in all_files:
            fname = os.path.basename(fpath).lower()
            parent = os.path.dirname(fpath)
            # 跳过 4D 文件
            if "4d" in fname:
                continue
            # 跳过分割掩码
            if "_gt" in fname:
                continue
            # 匹配 ED 帧
            if parent in ed_frames and ed_frames[parent] in fname:
                selected.append(fpath)
    else:
        # 无 Info.cfg，选择第一个 frame 文件作为近似 ED
        for fpath in all_files:
            fname = os.path.basename(fpath).lower()
            if "4d" in fname or "_gt" in fname:
                continue
            if "frame01" in fname or "frame00" in fname:
                selected.append(fpath)

    # 如果仍然没有找到任何帧文件，回退到跳过 4D 和 GT 的所有文件
    if not selected:
        selected = [
            f for f in all_files
            if "4d" not in os.path.basename(f).lower()
            and "_gt" not in os.path.basename(f).lower()
        ]

    return selected


def filter_mnm(data_dir: str, dataset_name: str) -> List[str]:
    """M&M 筛选器：与 ACDC 类似，选择 ED 帧。

    M&M 目录结构类似 ACDC，可能包含 _sa（短轴）标记。
    """
    all_files = _find_nifti_recursive(data_dir)
    selected = []

    for fpath in all_files:
        fname = os.path.basename(fpath).lower()
        # 跳过 4D 文件和分割掩码
        if "4d" in fname or "_gt" in fname or "label" in fname:
            continue
        # 选择 ED 帧
        if "ed" in fname or "frame01" in fname or "frame00" in fname:
            selected.append(fpath)

    # 回退：如果没有 ED 标记，选择所有非标签文件
    if not selected:
        selected = [
            f for f in all_files
            if "_gt" not in os.path.basename(f).lower()
            and "label" not in os.path.basename(f).lower()
        ]

    return selected


def filter_amos(data_dir: str, dataset_name: str) -> List[str]:
    """Amos 筛选器：仅保留 MRI 数据（排除 CT）。

    Amos 文件命名规则：amos_XXXX.nii.gz
      - 编号 0001-0200: CT
      - 编号 0201-0300: MRI（labelsTr 中 0501-0600 对应）

    也可能通过 dataset.json 或文件名中的 mri/ct 标记区分。
    """
    all_files = _find_nifti_recursive(data_dir)
    selected = []

    # 检查是否有 dataset.json 提供模态信息
    json_path = os.path.join(data_dir, "dataset.json")
    mri_ids = set()

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
            # 尝试从 modality 或 training 字段推断
            if "training" in meta:
                for item in meta["training"]:
                    # 某些版本的 dataset.json 包含模态标签
                    img = item.get("image", "")
                    if "mri" in img.lower() or "mr" in img.lower():
                        mri_ids.add(os.path.basename(img))
        except Exception:
            pass

    for fpath in all_files:
        fname = os.path.basename(fpath)
        fname_lower = fname.lower()

        # 跳过标签文件
        if "label" in fname_lower:
            continue

        # 方法 1：通过文件名中的 mri/ct 标记
        if "ct" in fname_lower and "mri" not in fname_lower:
            continue

        # 方法 2：通过编号判断（Amos 约定 0201+ 为 MRI）
        try:
            # 提取编号：amos_0201.nii.gz -> 201
            parts = fname.replace(".nii.gz", "").replace(".nii", "")
            num = int("".join(filter(str.isdigit, parts.split("_")[-1])))
            if num < 200:
                # 大概率是 CT
                continue
        except (ValueError, IndexError):
            pass

        # 方法 3：通过 dataset.json
        if mri_ids and fname not in mri_ids:
            continue

        selected.append(fpath)

    # 如果所有方法都无法区分，返回全部（让 quality_check 处理）
    if not selected:
        selected = [
            f for f in all_files
            if "label" not in os.path.basename(f).lower()
        ]

    return selected


def filter_msd(data_dir: str, dataset_name: str) -> List[str]:
    """Medical Segmentation Decathlon 筛选器。

    仅选择 MRI 任务的 imagesTr 文件：
      - Task01_BrainTumour (FLAIR, T1, T1ce, T2)
      - Task02_Heart (mono MRI)
      - Task05_Prostate (T2, ADC)

    其余任务为 CT，跳过。
    """
    mri_tasks = ["Task01", "Task02", "Task05"]
    selected = []

    for task in mri_tasks:
        # 尝试多种目录命名
        for pattern in [task, task.replace("Task", "task")]:
            task_dirs = []
            for entry in os.listdir(data_dir) if os.path.isdir(data_dir) else []:
                if entry.startswith(pattern):
                    task_dirs.append(os.path.join(data_dir, entry))

            for task_dir in task_dirs:
                images_dir = os.path.join(task_dir, "imagesTr")
                if os.path.isdir(images_dir):
                    files = _find_nifti_recursive(images_dir)
                    selected.extend(files)
                else:
                    # 如果没有 imagesTr 子目录，搜索整个任务目录
                    files = _find_nifti_recursive(task_dir)
                    # 排除标签文件
                    files = [
                        f for f in files
                        if "label" not in os.path.basename(f).lower()
                    ]
                    selected.extend(files)

    return selected


def filter_prostatex(data_dir: str, dataset_name: str) -> List[str]:
    """PROSTATEx 筛选器：优先选择 T2 序列。"""
    all_files = _find_nifti_recursive(data_dir)
    t2_files = [
        f for f in all_files
        if "t2" in os.path.basename(f).lower()
    ]
    if t2_files:
        return t2_files
    return all_files


def filter_lld_mmri(data_dir: str, dataset_name: str) -> List[str]:
    """LLD-MMRI 筛选器：优先选择 T2 或单一对比度。"""
    all_files = _find_nifti_recursive(data_dir)
    # 尝试选择 T2 序列
    t2_files = [
        f for f in all_files
        if "t2" in os.path.basename(f).lower()
    ]
    if t2_files:
        return t2_files
    return all_files


def filter_fastmri(data_dir: str, dataset_name: str) -> List[str]:
    """fastMRI 筛选器：仅选择重建后的 NIfTI 文件。

    fastMRI 原始数据为 HDF5 格式（.h5），跳过。
    如果已有重建后的 NIfTI 文件，直接使用。
    """
    all_files = _find_nifti_recursive(data_dir)
    if not all_files:
        # 检查是否有 h5 文件（原始 k-space，需要预先重建）
        h5_count = 0
        for root, _dirs, fnames in os.walk(data_dir):
            h5_count += sum(1 for f in fnames if f.endswith(".h5"))
        if h5_count > 0:
            print(
                f"  [fastMRI] 找到 {h5_count} 个 .h5 文件但无 NIfTI。"
                f"请先运行 fastMRI 重建脚本将 k-space 转换为 NIfTI。"
            )
    return all_files


def filter_duke_breast(data_dir: str, dataset_name: str) -> List[str]:
    """Duke Breast Cancer MRI 筛选器。

    如果是 DICOM 数据，需要先用 dcm2niix 转换。
    """
    nifti_files = _find_nifti_recursive(data_dir)
    if nifti_files:
        return nifti_files

    # 尝试 DICOM 转换
    return _convert_dicom_dir(data_dir, dataset_name)


def _convert_dicom_dir(data_dir: str, dataset_name: str) -> List[str]:
    """将目录下的 DICOM 文件转换为 NIfTI。"""
    # 检查是否有 DICOM 文件
    has_dicom = False
    for root, _dirs, fnames in os.walk(data_dir):
        for fname in fnames:
            if fname.lower().endswith((".dcm", ".ima")) or fname == "DICOMDIR":
                has_dicom = True
                break
        if has_dicom:
            break

    if not has_dicom:
        return []

    # 创建 NIfTI 输出目录
    nifti_dir = os.path.join(data_dir, "_nifti_converted")
    os.makedirs(nifti_dir, exist_ok=True)

    # 检查 dcm2niix 是否可用
    try:
        subprocess.run(
            ["dcm2niix", "--version"],
            capture_output=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            f"  [DICOM] dcm2niix 不可用。请安装: pip install dcm2niix"
        )
        return []

    print(f"  [DICOM] 正在将 {dataset_name} 的 DICOM 转换为 NIfTI...")
    try:
        subprocess.run(
            [
                "dcm2niix",
                "-z", "y",           # 压缩输出
                "-f", "%p_%s_%d",    # 文件名模板
                "-o", nifti_dir,     # 输出目录
                "-b", "y",           # 生成 BIDS sidecar
                data_dir,
            ],
            capture_output=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  [DICOM] 转换失败: {e.stderr.decode()[:200]}")
        return []

    return _find_nifti_recursive(nifti_dir)


# 数据集名称 → 筛选函数映射
DATASET_FILTERS = {
    "ACDC": filter_acdc,
    "MnM": filter_mnm,
    "Amos": filter_amos,
    "MSD": filter_msd,
    "PROSTATEx": filter_prostatex,
    "LLD_MMRI": filter_lld_mmri,
    "fastMRI": filter_fastmri,
    "fastMRI_brain": filter_fastmri,
    "fastMRI_knee": filter_fastmri,
    "Duke_Breast": filter_duke_breast,
    "QIN_Breast": filter_duke_breast,
    "RIDER_Breast": filter_duke_breast,
}


# ---------------------------------------------------------------------------
# 单文件处理
# ---------------------------------------------------------------------------

def process_single_file(
    nifti_path: str,
    dataset_name: str,
    region: str,
    output_dir: str,
    min_dim: int,
    target_spacing: Tuple[float, float, float],
    use_symlink: bool,
    dry_run: bool,
) -> Optional[Dict]:
    """处理单个 NIfTI 文件：质量检查 + 复制/链接到输出目录。

    返回：
        通过检查时返回清单记录字典，失败时返回 None。
    """
    import nibabel as nib

    # 质量检查
    passed, reason = quality_check_volume(
        nifti_path, min_dim=min_dim, target_spacing=target_spacing,
    )

    if not passed:
        return None

    # 读取元信息
    try:
        img = nib.load(nifti_path)
        shape = img.shape[:3]
        pixdim = img.header.get_zooms()[:3]
        data = img.get_fdata(dtype=np.float32)
        mean_val = float(np.mean(data[data > data.mean() * 0.1]))
        std_val = float(np.std(data[data > data.mean() * 0.1]))
        voxel_count = int(np.prod(shape))
    except Exception:
        return None

    # 构建输出路径
    fname = os.path.basename(nifti_path)
    # 添加数据集前缀以避免文件名冲突
    out_fname = f"{dataset_name}__{fname}" if not fname.startswith(dataset_name) else fname
    out_path = os.path.join(output_dir, region, dataset_name, out_fname)

    if not dry_run:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if use_symlink:
            # 使用绝对路径符号链接
            abs_src = os.path.abspath(nifti_path)
            if os.path.exists(out_path):
                os.remove(out_path)
            os.symlink(abs_src, out_path)
        else:
            shutil.copy2(nifti_path, out_path)

    return {
        "path": out_path,
        "source_path": nifti_path,
        "dataset": dataset_name,
        "region": region,
        "spacing_x": f"{pixdim[0]:.4f}",
        "spacing_y": f"{pixdim[1]:.4f}",
        "spacing_z": f"{pixdim[2]:.4f}",
        "shape_x": shape[0],
        "shape_y": shape[1],
        "shape_z": shape[2],
        "voxel_count": voxel_count,
        "mean_intensity": f"{mean_val:.4f}",
        "std_intensity": f"{std_val:.4f}",
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def prepare_datasets(
    sources: Dict[str, str],
    output_dir: str,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_dim: int = 48,
    workers: int = 4,
    use_symlink: bool = True,
    dry_run: bool = False,
):
    """执行完整的数据集准备流程。

    参数：
        sources: 数据源字典 {名称: 目录路径}。
        output_dir: 统一输出目录。
        spacing: 质量检查使用的目标体素间距。
        min_dim: 重采样后各维度最小体素数。
        workers: 并行处理的进程数。
        use_symlink: 使用符号链接而非复制。
        dry_run: 仅扫描和统计，不执行复制。
    """
    print("=" * 70)
    print("DINOv3 全身 MRI 数据集准备工具")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print(f"目标间距: {spacing}")
    print(f"最小维度: {min_dim}")
    print(f"并行进程: {workers}")
    print(f"使用符号链接: {use_symlink}")
    print(f"试运行: {dry_run}")
    print(f"数据源: {len(sources)} 个")
    print()

    all_records = []
    total_passed = 0
    total_failed = 0

    for ds_name, ds_path in sorted(sources.items()):
        print(f"--- [{ds_name}] {ds_path} ---")

        if not os.path.isdir(ds_path):
            print(f"  ⚠ 目录不存在，跳过")
            continue

        # 确定解剖区域
        region = DATASET_REGION_MAP.get(ds_name, DEFAULT_REGION)

        # 获取筛选函数
        filter_fn = DATASET_FILTERS.get(ds_name, filter_generic)

        # 执行数据集特定筛选
        try:
            candidate_files = filter_fn(ds_path, ds_name)
        except Exception as e:
            print(f"  ⚠ 筛选出错: {e}")
            continue

        if not candidate_files:
            print(f"  ⚠ 未找到有效文件")
            continue

        print(f"  找到 {len(candidate_files)} 个候选文件，区域: {region}")

        # 并行处理
        passed = 0
        failed = 0

        if workers > 1 and len(candidate_files) > 10:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_single_file,
                        fpath, ds_name, region, output_dir,
                        min_dim, spacing, use_symlink, dry_run,
                    ): fpath
                    for fpath in candidate_files
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            all_records.append(result)
                            passed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
        else:
            for fpath in candidate_files:
                try:
                    result = process_single_file(
                        fpath, ds_name, region, output_dir,
                        min_dim, spacing, use_symlink, dry_run,
                    )
                    if result is not None:
                        all_records.append(result)
                        passed += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1

        total_passed += passed
        total_failed += failed
        print(f"  ✓ 通过: {passed}, ✗ 未通过: {failed}")

    # 写入清单 CSV
    if all_records and not dry_run:
        manifest_path = os.path.join(output_dir, "manifest.csv")
        os.makedirs(output_dir, exist_ok=True)
        fieldnames = [
            "path", "source_path", "dataset", "region",
            "spacing_x", "spacing_y", "spacing_z",
            "shape_x", "shape_y", "shape_z",
            "voxel_count", "mean_intensity", "std_intensity",
        ]
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\n清单已保存: {manifest_path}")

    # 打印汇总
    print()
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"总计通过: {total_passed}")
    print(f"总计未通过: {total_failed}")
    print()

    # 按区域统计
    region_counts: Dict[str, int] = {}
    dataset_counts: Dict[str, int] = {}
    for rec in all_records:
        r = rec["region"]
        d = rec["dataset"]
        region_counts[r] = region_counts.get(r, 0) + 1
        dataset_counts[d] = dataset_counts.get(d, 0) + 1

    print("按区域:")
    for r, c in sorted(region_counts.items()):
        print(f"  {r:20s}: {c:6d} ({100*c/total_passed:.1f}%)")

    print()
    print("按数据集:")
    for d, c in sorted(dataset_counts.items()):
        print(f"  {d:20s}: {c:6d}")

    return all_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_sources(source_args: List[str]) -> Dict[str, str]:
    """解析 NAME=PATH 格式的数据源参数。"""
    sources = {}
    for arg in source_args:
        if "=" not in arg:
            raise ValueError(
                f"数据源格式错误: '{arg}'，期望 NAME=PATH"
            )
        name, path = arg.split("=", 1)
        sources[name.strip()] = os.path.expanduser(path.strip())
    return sources


def main():
    parser = argparse.ArgumentParser(
        description="DINOv3 全身 MRI 数据集准备工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="数据源列表，格式: NAME=PATH NAME=PATH ...",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="统一输出目录",
    )
    parser.add_argument(
        "--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help="质量检查用的目标体素间距（mm）",
    )
    parser.add_argument(
        "--min_dim", type=int, default=48,
        help="重采样后各维度最小体素数",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="并行处理的进程数",
    )
    parser.add_argument(
        "--copy", action="store_true", default=False,
        help="复制文件而非创建符号链接",
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="仅扫描和统计，不执行文件操作",
    )
    args = parser.parse_args()

    sources = parse_sources(args.sources)

    prepare_datasets(
        sources=sources,
        output_dir=args.output_dir,
        spacing=tuple(args.spacing),
        min_dim=args.min_dim,
        workers=args.workers,
        use_symlink=not args.copy,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
