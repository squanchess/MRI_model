"""
DINO Multi-Crop Transform for 3D MRI.

Modified from SPECTRE (MIT License) — key changes:

CT (SPECTRE)                          MRI (this file)
─────────────────────────────────────────────────────────────
ScaleIntensityRange(-1000, 1000)  →   Percentile clip + Z-score normalize
Spacing(0.5, 0.5, 1.0)           →   Spacing(1.0, 1.0, 1.0) isotropic
CenterCrop(512, 512, 384)        →   CenterCrop(192, 224, 192) for brain
Global crop (128, 128, 64)       →   Global crop (96, 96, 96) isotropic
Local crop (48, 48, 24)          →   Local crop (48, 48, 48) isotropic
RandScaleIntensityRange (HU)     →   RandScaleIntensity (generic)

The pipeline structure is preserved from SPECTRE:
  Load → Normalize → Reorient → Resample → CenterCrop → Pad →
  RandSpatialCropSamples → DINORandomCropTransform (global/local crops
  with resize + augmentation)
"""
from copy import deepcopy
from typing import Tuple, Mapping, Hashable, Any, List

import torch
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
    Orientationd,
    Spacingd,
    CenterSpatialCropd,
    SpatialPadd,
    EnsureTyped,
    RandSpatialCropSamplesd,
    SelectItemsd,
    RandSpatialCropSamples,
    RandFlip,
    OneOf,
    RandGaussianSharpen,
    RandGaussianSmooth,
    RandGaussianNoise,
    RandAdjustContrast,
    RandScaleIntensity,
    RandBiasField,
    Resize,
    MapTransform,
    Randomizable,
    LazyTransform,
)


class DINOTransform(Compose):
    """DINO multi-crop data augmentation pipeline for 3D MRI.

    Produces global and local crops from each MRI volume for DINO-style
    self-supervised training.

    Data flow:
        1. Load NIfTI → add channel dim
        2. Percentile intensity clip (0.5th-99.5th) → normalize to [0,1]
        3. Reorient to RAS → resample to isotropic 1mm³
        4. Center crop to fixed ROI (brain region)
        5. Spatial pad (ensure minimum size for cropping)
        6. Sample multiple base patches from the volume
        7. For each base patch: create 2 global crops + N local crops
           with random resize + augmentations

    Args:
        num_base_patches: Crops sampled per volume for I/O efficiency.
        global_views_size: Size of global crops (fed to both teacher & student).
        local_views_size: Size of local crops (fed to student only).
        local_views_scale: (min, max) scale range for local crop sampling.
        num_local_views: Number of local crops per base patch.
        roi_size: Center crop size to focus on brain region.
        spacing: Target voxel spacing (mm).
        dtype: "float32" or "float16".
    """
    def __init__(
        self,
        num_base_patches: int = 4,
        global_views_size: Tuple[int, int, int] = (96, 96, 96),
        local_views_size: Tuple[int, int, int] = (48, 48, 48),
        local_views_scale: Tuple[float, float] = (0.25, 0.5),
        num_local_views: int = 6,
        roi_size: Tuple[int, int, int] = (192, 224, 192),
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        dtype: str = "float32",
    ):
        assert dtype in ["float16", "float32"]

        # Base crop: large enough so that the biggest local crop fits inside
        base_crop_size = tuple(
            int(sz * (1 / local_views_scale[0])) for sz in local_views_size
        )  # e.g. (48 / 0.25) = (192, 192, 192)

        super().__init__([
            # --- Deterministic preprocessing ---
            LoadImaged(keys=("image",)),
            EnsureChannelFirstd(
                keys=("image",),
                channel_dim="no_channel",
            ),
            # MRI intensity normalization:
            # Step 1: Clip extreme values using percentiles (remove background/artifact)
            ScaleIntensityRangePercentilesd(
                keys=("image",),
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # Reorient to standard RAS orientation
            Orientationd(keys=("image",), axcodes="RAS"),
            # Resample to isotropic 1mm³ voxels
            Spacingd(
                keys=("image",),
                pixdim=spacing,
                mode=("bilinear",),
            ),
            # Center crop to focus on brain (removes most background)
            CenterSpatialCropd(
                keys=("image",),
                roi_size=roi_size,
            ),
            # Pad if volume is smaller than base_crop_size
            SpatialPadd(
                keys=("image",),
                spatial_size=base_crop_size,
            ),
            EnsureTyped(
                keys=("image",),
                dtype=getattr(torch, dtype),
                device="cpu",
            ),
            # --- Random sampling ---
            # Sample multiple base patches from the volume
            RandSpatialCropSamplesd(
                keys=("image",),
                num_samples=num_base_patches,
                roi_size=base_crop_size,
                random_size=False,
                random_center=True,
            ),
            # DINO multi-crop: global + local views with augmentations
            DINORandomCropTransformd(
                keys=("image",),
                base_crop_size=base_crop_size,
                global_views_size=global_views_size,
                local_views_size=local_views_size,
                local_views_scale=local_views_scale,
                num_local_views=num_local_views,
                dtype=dtype,
            ),
            SelectItemsd(
                keys=("image_global_views", "image_local_views"),
            ),
        ])


class DINORandomCropTransformd(Randomizable, MapTransform, LazyTransform):
    """Create DINO global and local crops with augmentations.

    For each input patch, produces:
      - 2 global crops (random scale 0.5-1.0 of base, resized to global_views_size)
      - N local crops (random scale local_views_scale of base, resized to local_views_size)

    Each crop is independently augmented with flips, blur, noise, contrast, etc.

    Modified from SPECTRE:
      - RandScaleIntensityRange (CT HU-based) replaced with
        RandScaleIntensity + RandBiasField (MRI-appropriate)
      - Gaussian kernel sigma values made isotropic
    """
    def __init__(
        self,
        keys: KeysCollection,
        base_crop_size: Tuple[int, int, int] = (192, 192, 192),
        global_views_size: Tuple[int, int, int] = (96, 96, 96),
        local_views_size: Tuple[int, int, int] = (48, 48, 48),
        local_views_scale: Tuple[float, float] = (0.25, 0.5),
        num_local_views: int = 6,
        dtype: str = "float32",
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys)
        LazyTransform.__init__(self, lazy)
        self.global_views_size = global_views_size
        self.local_views_size = local_views_size
        self.local_views_scale = local_views_scale
        self.num_local_views = num_local_views

        # Global crops: random size between 50%-100% of base
        self.cropper_global = RandSpatialCropSamples(
            roi_size=tuple(int(local_views_scale[1] * sz) for sz in base_crop_size),
            num_samples=2,
            max_roi_size=base_crop_size,
            random_center=True,
            random_size=True,
            lazy=lazy,
        )
        # Local crops: random size between local_views_scale[0]-[1] of base
        self.cropper_local = RandSpatialCropSamples(
            roi_size=tuple(int(self.local_views_scale[0] * sz) for sz in base_crop_size),
            num_samples=num_local_views,
            max_roi_size=tuple(int(self.local_views_scale[1] * sz) for sz in base_crop_size),
            random_center=True,
            random_size=True,
            lazy=lazy,
        )

        self.resize_global = Resize(
            spatial_size=global_views_size,
            mode="trilinear",
            dtype=getattr(torch, dtype),
            anti_aliasing=True,
            lazy=lazy,
        )
        self.resize_local = Resize(
            spatial_size=local_views_size,
            mode="trilinear",
            dtype=getattr(torch, dtype),
            anti_aliasing=True,
            lazy=lazy,
        )

        # MRI-appropriate augmentations
        self.augmentor = Compose([
            # Random flips (brain is roughly symmetric in L-R)
            RandFlip(spatial_axis=0, prob=0.5),
            RandFlip(spatial_axis=1, prob=0.5),
            RandFlip(spatial_axis=2, prob=0.5),
            # Blur / sharpen (isotropic sigmas for MRI)
            OneOf([
                RandGaussianSharpen(
                    sigma1_x=(1.0, 2.0), sigma1_y=(1.0, 2.0), sigma1_z=(1.0, 2.0),
                    sigma2_x=(0.25, 0.75), sigma2_y=(0.25, 0.75), sigma2_z=(0.25, 0.75),
                    prob=0.25,
                ),
                RandGaussianSmooth(
                    sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5),
                    prob=0.25,
                ),
            ]),
            # Contrast jitter
            RandAdjustContrast(gamma=(0.8, 1.2), prob=0.3),
            # Additive Gaussian noise (simulate acquisition noise)
            RandGaussianNoise(std=0.05, sample_std=True, prob=0.25),
            # Random intensity scaling (simulate scanner intensity drift)
            RandScaleIntensity(factors=0.15, prob=0.25),
            # Simulate MRI bias field inhomogeneity
            RandBiasField(
                degree=3,
                coeff_range=(0.0, 0.3),
                prob=0.2,
            ),
        ], lazy=lazy)

    def randomize(self, data: Any = None) -> None:
        self.sub_seed = self.R.randint(0, 2**32 // 2 - 1)
        self.cropper_global.set_random_state(seed=self.sub_seed)
        self.cropper_local.set_random_state(seed=self.sub_seed)
        self.augmentor.set_random_state(seed=self.sub_seed)

    def __call__(
        self,
        data: Mapping[Hashable, Any] | List[Mapping[Hashable, Any]],
        lazy: bool | None = None,
    ) -> dict[Hashable, Any]:

        # Support list of dicts (from RandSpatialCropSamplesd)
        if isinstance(data, list):
            return [self.__call__(d, lazy=lazy) for d in data]

        ret = dict()
        for key in set(data.keys()).difference(set(self.keys)):
            ret[key] = deepcopy(data[key])

        self.randomize()
        lazy_ = self.lazy if lazy is None else lazy

        for key in self.key_iterator(dict(data)):
            image = data[key]
            global_views = list(self.cropper_global(image, lazy=lazy_))
            local_views = list(self.cropper_local(image, lazy=lazy_))

            global_views = [self.resize_global(gv, lazy=lazy_) for gv in global_views]
            local_views = [self.resize_local(lv, lazy=lazy_) for lv in local_views]

            global_views = [self.augmentor(gv, lazy=lazy_) for gv in global_views]
            local_views = [self.augmentor(lv, lazy=lazy_) for lv in local_views]

            ret[f"{key}_global_views"] = global_views
            ret[f"{key}_local_views"] = local_views

        return ret
