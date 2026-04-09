"""
适用于 3D MRI 的 DINO 多裁剪变换（容错版）。

容错机制：
  - SafeDINOTransform 包裹 DINOTransform，捕获所有异常
  - 损坏/异常文件返回 None，由 collate_dino 过滤
  - 训练不会因单个坏文件中断
"""
from copy import deepcopy
from typing import Any, Hashable, List, Mapping, Optional, Tuple
import logging

import torch
from monai.config import KeysCollection
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LazyTransform,
    LoadImaged,
    MapTransform,
    OneOf,
    Orientationd,
    RandAdjustContrast,
    RandBiasField,
    RandFlip,
    RandGaussianNoise,
    RandGaussianSharpen,
    RandGaussianSmooth,
    RandScaleIntensity,
    RandSpatialCropSamples,
    RandSpatialCropSamplesd,
    Randomizable,
    Resize,
    ScaleIntensityRangePercentilesd,
    SelectItemsd,
    Spacingd,
    SpatialPadd,
)

log = logging.getLogger("transforms")


class Squeeze4Dto3Dd(MapTransform):
    """将 4D 数据压缩为 3D：取第一个时间帧。"""

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.ndim == 5:
                d[key] = img[:, :, :, :, 0]
            elif img.ndim > 5:
                while img.ndim > 4:
                    img = img[..., 0]
                d[key] = img
        return d


class SafeDINOTransform:
    """容错包装器：捕获 DINOTransform 中的所有异常。

    当某个样本的 transform 失败（文件损坏、格式异常、维度错误等），
    不抛异常，而是返回 None。配合 collate_dino 中的 None 过滤，
    实现训练对坏样本的自动跳过。
    """

    def __init__(self, transform):
        self.transform = transform
        self.error_count = 0
        self.max_log = 50

    def __call__(self, data):
        try:
            return self.transform(data)
        except Exception as e:
            self.error_count += 1
            filepath = "unknown"
            if isinstance(data, dict):
                filepath = data.get("image", "unknown")
                if not isinstance(filepath, str):
                    filepath = str(getattr(filepath, "meta", {}).get(
                        "filename_or_obj", "unknown"))

            if self.error_count <= self.max_log:
                log.warning(
                    "Skipping corrupt sample #%d: %s -> %s",
                    self.error_count, filepath, str(e)[:200],
                )
                if self.error_count == self.max_log:
                    log.warning("(further error logs suppressed)")

            return None


class DINOTransform(Compose):
    """用于 3D MRI 的 DINO 多裁剪数据增强流水线。"""

    def __init__(
        self,
        num_base_patches: int = 4,
        global_views_size: Tuple[int, int, int] = (96, 96, 96),
        local_views_size: Tuple[int, int, int] = (48, 48, 48),
        local_views_scale: Tuple[float, float] = (0.25, 0.5),
        num_local_views: int = 6,
        roi_size: Optional[Tuple[int, int, int]] = (192, 224, 192),
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        dtype: str = "float32",
        use_foreground_crop: bool = False,
        foreground_margin: int = 10,
    ):
        assert dtype in ["float16", "float32"]

        base_crop_size = tuple(
            int(sz * (1 / local_views_scale[0])) for sz in local_views_size
        )

        deterministic_transforms = [
            LoadImaged(keys=("image",)),
            EnsureChannelFirstd(
                keys=("image",),
                channel_dim="no_channel",
            ),
            Squeeze4Dto3Dd(keys=("image",)),
            ScaleIntensityRangePercentilesd(
                keys=("image",),
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=("image",), axcodes="RAS"),
            Spacingd(
                keys=("image",),
                pixdim=spacing,
                mode=("bilinear",),
            ),
        ]

        if use_foreground_crop:
            deterministic_transforms.append(
                CropForegroundd(
                    keys=("image",),
                    source_key="image",
                    margin=foreground_margin,
                )
            )
        elif roi_size is not None:
            deterministic_transforms.append(
                CenterSpatialCropd(
                    keys=("image",),
                    roi_size=roi_size,
                )
            )

        deterministic_transforms.extend([
            SpatialPadd(
                keys=("image",),
                spatial_size=base_crop_size,
            ),
            EnsureTyped(
                keys=("image",),
                dtype=getattr(torch, dtype),
                device="cpu",
            ),
        ])

        random_transforms = [
            RandSpatialCropSamplesd(
                keys=("image",),
                num_samples=num_base_patches,
                roi_size=base_crop_size,
                random_size=False,
                random_center=True,
            ),
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
        ]

        super().__init__(deterministic_transforms + random_transforms)


class DINORandomCropTransformd(Randomizable, MapTransform, LazyTransform):
    """生成带增强的 DINO 全局与局部裁剪。"""

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

        self.cropper_global = RandSpatialCropSamples(
            roi_size=tuple(int(local_views_scale[1] * sz) for sz in base_crop_size),
            num_samples=2,
            max_roi_size=base_crop_size,
            random_center=True,
            random_size=True,
            lazy=lazy,
        )
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

        self.augmentor = Compose([
            RandFlip(spatial_axis=0, prob=0.5),
            RandFlip(spatial_axis=1, prob=0.5),
            RandFlip(spatial_axis=2, prob=0.5),
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
            RandAdjustContrast(gamma=(0.8, 1.2), prob=0.3),
            RandGaussianNoise(std=0.05, sample_std=True, prob=0.25),
            RandScaleIntensity(factors=0.15, prob=0.25),
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
