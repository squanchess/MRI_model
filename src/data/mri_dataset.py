"""
IXI T1 MRI 数据集。

IXI 数据集包含约 600 个 T1 加权脑 MRI 体数据，来自伦敦三家医院，
以 NIfTI 文件形式分发：
  - IXI002-Guys-0828-T1.nii.gz   (Guy's Hospital, Philips 1.5T)
  - IXI012-HH-1211-T1.nii.gz     (Hammersmith Hospital, Philips 3T)
  - IXI015-IOP-0852-T1.nii.gz    (Institute of Psychiatry, GE 1.5T)

典型体数据尺寸约为：256 x 256 x 130-150，体素间距约为 0.94 x 0.94 x 1.2 mm。

用法：
    dataset = IXIDataset(data_dir="/path/to/IXI-T1", transform=my_transform)
    sample = dataset[0]  # {"image": "/path/to/IXI002-Guys-0828-T1.nii.gz"}
"""
import glob
import os
import warnings
from typing import Callable, List, Optional

from monai.data import CacheDataset, Dataset, PersistentDataset


def _discover_ixi_files(
    data_dir: str,
    sites: Optional[List[str]] = None,
    fraction: float = 1.0,
) -> List[dict]:
    """扫描 `data_dir`，查找 IXI T1 的 NIfTI 文件。

    支持两种目录结构：

    结构 A：扁平目录（标准 IXI 下载形式）：
        data_dir/
            IXI002-Guys-0828-T1.nii.gz
            IXI012-HH-1211-T1.nii.gz
            ...

    结构 B：按受试者分目录（例如 FLamby/TorchIO 风格）：
        data_dir/
            IXI002-Guys-0828/
                T1/
                    IXI002-Guys-0828-T1.nii.gz
            ...

    参数：
        data_dir: IXI 数据根目录。
        sites: 按医院站点筛选，可选值为 "Guys"、"HH"、"IOP"；
            为 None 时使用全部站点。
        fraction: 使用的数据比例，范围为 (0, 1]，便于调试。

    返回：
        供 MONAI transform 使用的字典列表，格式为 [{"image": path}, ...]。
    """
    # 先尝试结构 A（扁平目录）
    patterns = [
        os.path.join(data_dir, "*.nii.gz"),
        os.path.join(data_dir, "*.nii"),
    ]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))

    # 筛选 T1 数据
    t1_files = [f for f in files if "T1" in os.path.basename(f).upper()]
    if not t1_files:
        # 如果筛选不到，则默认这些文件全都是 T1
        t1_files = files

    # 如果扁平目录未找到，再尝试结构 B
    if not t1_files:
        patterns_nested = [
            os.path.join(data_dir, "**", "*T1*.nii.gz"),
            os.path.join(data_dir, "**", "*T1*.nii"),
            os.path.join(data_dir, "**", "*t1*.nii.gz"),
        ]
        for p in patterns_nested:
            t1_files.extend(sorted(glob.glob(p, recursive=True)))

    if not t1_files:
        raise FileNotFoundError(
            f"No NIfTI files found in {data_dir}. "
            f"Expected IXI*-T1.nii.gz files or nested T1/ directories."
        )

    # 按医院站点筛选
    if sites is not None:
        site_set = {s.upper() for s in sites}
        filtered = []
        for f in t1_files:
            basename = os.path.basename(f).upper()
            if any(site in basename for site in site_set):
                filtered.append(f)
        if not filtered:
            warnings.warn(
                f"No files matched sites={sites} in {data_dir}. "
                f"Available files: {len(t1_files)}. Using all."
            )
        else:
            t1_files = filtered

    # 按比例截取数据
    n = max(1, int(len(t1_files) * fraction))
    t1_files = t1_files[:n]

    # 构造兼容 MONAI 的数据列表
    data_list = [{"image": f} for f in t1_files]
    return data_list


class IXIDataset(Dataset):
    """基于 MONAI Dataset 的 IXI T1 MRI 数据集。

    每个样本在访问时即时加载。如果希望借助缓存提高训练速度，
    可以使用 IXICacheDataset 或 IXIPersistentDataset。

    参数：
        data_dir: 包含 IXI T1 NIfTI 文件的目录路径。
        transform: MONAI 的变换流水线（例如 DINOTransform）。
        sites: 医院筛选项（"Guys"、"HH"、"IOP"），None 表示全部。
        fraction: 使用的数据集比例，常用于调试。
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        sites: Optional[List[str]] = None,
        fraction: float = 1.0,
    ):
        data_list = _discover_ixi_files(data_dir, sites=sites, fraction=fraction)
        super().__init__(data=data_list, transform=transform)


class IXICacheDataset(CacheDataset):
    """带内存缓存的 IXI T1 数据集（首个 epoch 后会更快）。

    会将确定性变换（加载、重采样、归一化等）的结果缓存在内存中。
    随机增强仍会在每个 epoch 重新执行。

    参数：
        data_dir: IXI T1 NIfTI 文件所在路径。
        transform: 完整的变换流水线。
        sites: 医院筛选条件。
        fraction: 数据使用比例。
        cache_rate: 缓存比例，1.0 表示全部缓存。
        num_workers: 初始化时用于并行构建缓存的 worker 数量。
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        sites: Optional[List[str]] = None,
        fraction: float = 1.0,
        cache_rate: float = 1.0,
        num_workers: int = 4,
    ):
        data_list = _discover_ixi_files(data_dir, sites=sites, fraction=fraction)
        super().__init__(
            data=data_list,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )


class IXIPersistentDataset(PersistentDataset):
    """带磁盘持久化缓存的 IXI T1 数据集。

    会将确定性变换的结果写入磁盘。当数据集过大、无法完全缓存在内存中，
    或者训练需要频繁中断恢复时，这种方式更适合。

    参数：
        data_dir: IXI T1 NIfTI 文件所在路径。
        transform: 完整的变换流水线。
        sites: 医院筛选条件。
        fraction: 数据使用比例。
        cache_dir: 持久化缓存文件目录。
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        sites: Optional[List[str]] = None,
        fraction: float = 1.0,
        cache_dir: str = "./cache/ixi",
    ):
        data_list = _discover_ixi_files(data_dir, sites=sites, fraction=fraction)
        os.makedirs(cache_dir, exist_ok=True)
        super().__init__(
            data=data_list,
            transform=transform,
            cache_dir=cache_dir,
        )
