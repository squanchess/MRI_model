"""
通用 MRI 数据集。

扩展自 IXI 专用实现，支持任意 NIfTI 格式的 MRI 数据集，
并提供多数据源合并、区域均衡采样等功能，用于全身 MRI 预训练。

兼容性：
  - IXIDataset / IXICacheDataset / IXIPersistentDataset 作为别名保留，
    确保原有脑部训练流程无需修改。

用法：
    # 单数据源（兼容旧接口）
    dataset = MRIDataset(data_dir="/path/to/nifti", transform=my_transform)

    # 多数据源
    dataset = MultiSourceMRIDataset(
        data_dirs={"brain": "/data/brain", "cardiac": "/data/cardiac"},
        transform=my_transform,
    )
    sampler = dataset.get_balanced_sampler(weights={"brain": 1.0, "cardiac": 3.0})
"""
import glob
import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from monai.data import CacheDataset, Dataset, PersistentDataset


# ---------------------------------------------------------------------------
# 通用 NIfTI 文件发现
# ---------------------------------------------------------------------------

def _discover_nifti_files(
    data_dir: str,
    recursive: bool = True,
    fraction: float = 1.0,
    extensions: Tuple[str, ...] = (".nii.gz", ".nii"),
) -> List[dict]:
    """扫描 `data_dir`，查找所有 NIfTI 文件。

    与原 _discover_ixi_files 不同，本函数不对文件名做任何过滤
    （如 "T1" 关键词），从而兼容任意数据集的命名规范。

    支持两种扫描模式：
      - recursive=True：递归搜索所有子目录
      - recursive=False：仅搜索 data_dir 顶层

    参数：
        data_dir: 数据根目录。
        recursive: 是否递归搜索子目录。
        fraction: 使用的数据比例，范围为 (0, 1]。
        extensions: 要搜索的文件扩展名。

    返回：
        供 MONAI transform 使用的字典列表 [{"image": path}, ...]。
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    files = []

    if recursive:
        for root, _dirs, fnames in os.walk(data_dir, followlinks=True):
            for fname in fnames:
                if any(fname.lower().endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, fname))
    else:
        for ext in extensions:
            pattern = os.path.join(data_dir, f"*{ext}")
            files.extend(glob.glob(pattern))

    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"在 {data_dir} 中未找到 NIfTI 文件（扩展名: {extensions}）。"
        )

    # 按比例截取
    n = max(1, int(len(files) * fraction))
    files = files[:n]

    data_list = [{"image": f} for f in files]
    return data_list


def _discover_ixi_files(
    data_dir: str,
    sites: Optional[List[str]] = None,
    fraction: float = 1.0,
) -> List[dict]:
    """扫描 `data_dir`，查找 IXI T1 的 NIfTI 文件（向后兼容）。

    保留原有的站点筛选功能和 T1 文件名过滤逻辑。

    参数：
        data_dir: IXI 数据根目录。
        sites: 按医院站点筛选（"Guys"、"HH"、"IOP"），None 表示全部。
        fraction: 使用的数据比例。

    返回：
        供 MONAI transform 使用的字典列表 [{"image": path}, ...]。
    """
    # 先用通用发现
    all_items = _discover_nifti_files(data_dir, recursive=True, fraction=1.0)
    all_files = [d["image"] for d in all_items]

    # 筛选 T1 数据
    t1_files = [f for f in all_files if "T1" in os.path.basename(f).upper()]
    if not t1_files:
        t1_files = all_files

    # 按医院站点筛选
    if sites is not None:
        site_set = {s.upper() for s in sites}
        filtered = [
            f for f in t1_files
            if any(site in os.path.basename(f).upper() for site in site_set)
        ]
        if not filtered:
            warnings.warn(
                f"没有文件匹配 sites={sites}（{data_dir}），使用全部 {len(t1_files)} 个文件。"
            )
        else:
            t1_files = filtered

    # 按比例截取
    n = max(1, int(len(t1_files) * fraction))
    t1_files = t1_files[:n]

    return [{"image": f} for f in t1_files]


# ---------------------------------------------------------------------------
# 通用 MRI 数据集
# ---------------------------------------------------------------------------

class MRIDataset(Dataset):
    """通用 MRI 数据集（基于 MONAI Dataset）。

    与 IXIDataset 的区别：
      - 不限定文件名格式，接受任意 NIfTI 文件
      - 支持递归搜索子目录
      - 不按站点筛选（该功能保留在 IXIDataset 中）

    参数：
        data_dir: 包含 NIfTI 文件的目录路径。
        transform: MONAI 变换流水线。
        fraction: 使用的数据集比例。
        recursive: 是否递归搜索子目录。
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        fraction: float = 1.0,
        recursive: bool = True,
    ):
        data_list = _discover_nifti_files(
            data_dir, recursive=recursive, fraction=fraction,
        )
        super().__init__(data=data_list, transform=transform)
        self.data_dir = data_dir


class MRICacheDataset(CacheDataset):
    """带内存缓存的通用 MRI 数据集。

    参数：
        data_dir: NIfTI 文件所在路径。
        transform: 完整的变换流水线。
        fraction: 数据使用比例。
        recursive: 是否递归搜索子目录。
        cache_rate: 缓存比例。
        num_workers: 缓存构建时的并行 worker 数量。
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        fraction: float = 1.0,
        recursive: bool = True,
        cache_rate: float = 1.0,
        num_workers: int = 4,
    ):
        data_list = _discover_nifti_files(
            data_dir, recursive=recursive, fraction=fraction,
        )
        super().__init__(
            data=data_list,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        self.data_dir = data_dir


class MRIPersistentDataset(PersistentDataset):
    """带磁盘持久化缓存的通用 MRI 数据集。

    参数：
        data_dir: NIfTI 文件所在路径。
        transform: 完整的变换流水线。
        fraction: 数据使用比例。
        recursive: 是否递归搜索子目录。
        cache_dir: 磁盘缓存目录。
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        fraction: float = 1.0,
        recursive: bool = True,
        cache_dir: str = "./cache/mri",
    ):
        data_list = _discover_nifti_files(
            data_dir, recursive=recursive, fraction=fraction,
        )
        os.makedirs(cache_dir, exist_ok=True)
        super().__init__(
            data=data_list,
            transform=transform,
            cache_dir=cache_dir,
        )
        self.data_dir = data_dir


# ---------------------------------------------------------------------------
# 多数据源合并数据集
# ---------------------------------------------------------------------------

class MultiSourceMRIDataset:
    """多数据源 MRI 数据集，支持区域均衡采样。

    将多个目录下的 NIfTI 文件合并为一个 ConcatDataset，
    并提供加权随机采样器以均衡不同解剖区域的训练占比。

    参数：
        data_dirs: 数据源字典 {名称: 目录路径} 或目录列表。
        transform: MONAI 变换流水线，所有数据源共用。
        fraction: 全局数据使用比例。
        recursive: 是否递归搜索子目录。

    示例：
        dataset = MultiSourceMRIDataset(
            data_dirs={
                "brain": "/data/wholebody/brain",
                "cardiac": "/data/wholebody/cardiac",
                "knee": "/data/wholebody/knee",
            },
            transform=my_transform,
        )
        sampler = dataset.get_balanced_sampler(
            weights={"brain": 1.0, "cardiac": 3.0, "knee": 1.5}
        )
        loader = DataLoader(dataset.dataset, sampler=sampler, ...)
    """

    def __init__(
        self,
        data_dirs: Union[Dict[str, str], List[str]],
        transform: Optional[Callable] = None,
        fraction: float = 1.0,
        recursive: bool = True,
        cache_rate: float = 0.0,
        cache_num_workers: int = 4,
    ):
        # 统一为字典形式
        if isinstance(data_dirs, (list, tuple)):
            data_dirs = {
                os.path.basename(d.rstrip("/")): d for d in data_dirs
            }

        self.source_names: List[str] = []
        self.source_sizes: Dict[str, int] = {}
        self.source_offsets: Dict[str, Tuple[int, int]] = {}
        datasets: List[Dataset] = []
        offset = 0

        use_cache = cache_rate > 0

        for name, path in sorted(data_dirs.items()):
            if not os.path.isdir(path):
                warnings.warn(f"数据源 '{name}' 目录不存在，跳过: {path}")
                continue

            try:
                data_list = _discover_nifti_files(
                    path, recursive=recursive, fraction=fraction,
                )
                if use_cache:
                    ds = CacheDataset(
                        data=data_list,
                        transform=transform,
                        cache_rate=cache_rate,
                        num_workers=cache_num_workers,
                    )
                else:
                    ds = MRIDataset(
                        data_dir=path,
                        transform=transform,
                        fraction=fraction,
                        recursive=recursive,
                    )
            except FileNotFoundError as e:
                warnings.warn(f"数据源 '{name}' 无有效文件，跳过: {e}")
                continue

            n = len(ds)
            self.source_names.append(name)
            self.source_sizes[name] = n
            self.source_offsets[name] = (offset, offset + n)
            datasets.append(ds)
            offset += n

        if not datasets:
            raise FileNotFoundError(
                f"所有数据源均无有效文件: {list(data_dirs.keys())}"
            )

        self.dataset = ConcatDataset(datasets)
        self._datasets = datasets

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def summary(self) -> str:
        """返回各数据源的体数据数量摘要。"""
        lines = [f"MultiSourceMRIDataset — 共 {len(self)} 个体数据:"]
        for name in self.source_names:
            n = self.source_sizes[name]
            pct = 100.0 * n / len(self)
            lines.append(f"  {name:20s}: {n:6d} ({pct:5.1f}%)")
        return "\n".join(lines)

    def get_balanced_sampler(
        self,
        weights: Optional[Dict[str, float]] = None,
        num_samples: Optional[int] = None,
    ) -> WeightedRandomSampler:
        """构造加权随机采样器，均衡不同数据源的采样频率。

        参数：
            weights: 各数据源的采样权重 {名称: 权重}，
                     不在字典中的源默认权重为 1.0。
                     设为 None 时按数据量倒数均衡（每个源被采样概率相同）。
            num_samples: 每个 epoch 的总采样数，默认为数据集大小。

        返回：
            torch WeightedRandomSampler 对象。
        """
        total = len(self.dataset)
        sample_weights = np.ones(total, dtype=np.float64)

        if weights is None:
            # 默认策略：按数据量倒数均衡
            for name in self.source_names:
                start, end = self.source_offsets[name]
                n = end - start
                if n > 0:
                    sample_weights[start:end] = 1.0 / n
        else:
            for name in self.source_names:
                start, end = self.source_offsets[name]
                w = weights.get(name, 1.0)
                n = end - start
                if n > 0:
                    # 权重 / 数据量 = 每个样本的采样概率
                    sample_weights[start:end] = w / n

        # 归一化
        sample_weights /= sample_weights.sum()

        if num_samples is None:
            num_samples = total

        return WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=num_samples,
            replacement=True,
        )


# ---------------------------------------------------------------------------
# 向后兼容别名
# ---------------------------------------------------------------------------

class IXIDataset(Dataset):
    """IXI T1 数据集（向后兼容）。

    保留原有的站点筛选功能。新项目建议直接使用 MRIDataset。
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
    """IXI T1 带缓存数据集（向后兼容）。"""

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
    """IXI T1 持久化缓存数据集（向后兼容）。"""

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
