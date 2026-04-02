"""
IXI T1 MRI Dataset.

The IXI dataset contains ~600 T1-weighted brain MRI volumes from three
London hospitals, distributed as NIfTI files:
  - IXI002-Guys-0828-T1.nii.gz   (Guy's Hospital, Philips 1.5T)
  - IXI012-HH-1211-T1.nii.gz     (Hammersmith Hospital, Philips 3T)
  - IXI015-IOP-0852-T1.nii.gz    (Institute of Psychiatry, GE 1.5T)

Typical volume shape: ~256 × 256 × 130-150, voxel size ~0.94 × 0.94 × 1.2 mm.

Usage:
    dataset = IXIDataset(data_dir="/path/to/IXI-T1", transform=my_transform)
    sample = dataset[0]  # {"image": "/path/to/IXI002-Guys-0828-T1.nii.gz"}
"""
import os
import glob
import warnings
from typing import Optional, Callable, List

from monai.data import Dataset, CacheDataset, PersistentDataset


def _discover_ixi_files(
    data_dir: str,
    sites: Optional[List[str]] = None,
    fraction: float = 1.0,
) -> List[dict]:
    """Scan data_dir for IXI T1 NIfTI files.

    Supports two directory layouts:

    Layout A — flat directory (standard IXI download):
        data_dir/
            IXI002-Guys-0828-T1.nii.gz
            IXI012-HH-1211-T1.nii.gz
            ...

    Layout B — per-subject folders (e.g. FLamby/TorchIO style):
        data_dir/
            IXI002-Guys-0828/
                T1/
                    IXI002-Guys-0828-T1.nii.gz
            ...

    Args:
        data_dir: Root directory containing IXI data.
        sites: Filter by hospital site(s). Options: "Guys", "HH", "IOP".
                None = use all sites.
        fraction: Fraction of data to use (0, 1]. Useful for debugging.

    Returns:
        List of dicts [{"image": path}, ...] for MONAI transforms.
    """
    # Try Layout A first (flat)
    patterns = [
        os.path.join(data_dir, "*.nii.gz"),
        os.path.join(data_dir, "*.nii"),
    ]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))

    # Filter for T1 files if mixed modalities are present
    t1_files = [f for f in files if "T1" in os.path.basename(f).upper()]
    if not t1_files:
        # Maybe all files are T1 (user already filtered), use everything
        t1_files = files

    # Try Layout B if flat didn't work
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

    # Filter by hospital site
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

    # Apply fraction
    n = max(1, int(len(t1_files) * fraction))
    t1_files = t1_files[:n]

    # Build MONAI-compatible data list
    data_list = [{"image": f} for f in t1_files]
    return data_list


class IXIDataset(Dataset):
    """IXI T1 MRI dataset using MONAI's Dataset.

    Each sample is loaded on-the-fly. Use IXICacheDataset or
    IXIPersistentDataset for faster training with caching.

    Args:
        data_dir: Path to directory containing IXI T1 NIfTI files.
        transform: MONAI transform pipeline (e.g. DINOTransform).
        sites: Filter by hospital ("Guys", "HH", "IOP"). None = all.
        fraction: Fraction of dataset to use (for debugging).
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
    """IXI T1 dataset with in-memory caching (faster after first epoch).

    Caches the result of the deterministic transforms (loading, resampling,
    normalization) in memory. Random augmentations are still applied fresh
    each epoch.

    Args:
        data_dir: Path to IXI T1 NIfTI files.
        transform: Full transform pipeline.
        sites: Hospital filter.
        fraction: Data fraction.
        cache_rate: Fraction of dataset to cache (1.0 = all).
        num_workers: Workers for parallel caching at init.
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
    """IXI T1 dataset with disk-based persistent caching.

    Saves deterministic transform results to disk. Useful when the dataset
    is too large to cache in memory or when training is resumed frequently.

    Args:
        data_dir: Path to IXI T1 NIfTI files.
        transform: Full transform pipeline.
        sites: Hospital filter.
        fraction: Data fraction.
        cache_dir: Directory for persistent cache files.
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
