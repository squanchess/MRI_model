"""
适用于 3D MRI 的 DINO 多裁剪变换。

基于 SPECTRE（MIT License）修改，关键差异如下：

CT (SPECTRE)                          MRI（本文件）
------------------------------------------------------------
ScaleIntensityRange(-1000, 1000)   ->  百分位裁剪 + Z-score 归一化
Spacing(0.5, 0.5, 1.0)             ->  Spacing(1.0, 1.0, 1.0) 各向同性
CenterCrop(512, 512, 384)          ->  CropForeground / CenterCrop 可选
Global crop (128, 128, 64)         ->  Global crop (96, 96, 96) 各向同性
Local crop (48, 48, 24)            ->  Local crop (48, 48, 48) 各向同性
RandScaleIntensityRange (HU)       ->  RandScaleIntensity（通用）

全身 MRI 扩展：
  - 新增 use_foreground_crop 选项，替代 CenterSpatialCrop
  - roi_size 为可选参数；设为 None 时跳过裁剪步骤
  - 所有增强操作解剖无关，无需按体部定制

整体流程沿用了 SPECTRE 的结构：
  Load -> Normalize -> Reorient -> Resample -> Crop -> Pad ->
  RandSpatialCropSamples -> DINORandomCropTransform
  （生成全局/局部裁剪，并执行 resize 与增强）
"""
from copy import deepcopy
from typing import Any, Hashable, List, Mapping, Optional, Tuple

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


class DINOTransform(Compose):
    """用于 3D MRI 的 DINO 多裁剪数据增强流水线。

    会从每个 MRI 体数据中生成全局视图和局部视图，
    用于 DINO 风格的自监督训练。

    数据流：
        1. 加载 NIfTI，并补充通道维
        2. 执行百分位强度裁剪（0.5%-99.5%），归一化到 [0, 1]
        3. 重定向到 RAS 坐标系，并重采样到目标体素间距
        4. 裁剪策略：
           - use_foreground_crop=True: CropForeground（全身 MRI）
           - use_foreground_crop=False: CenterSpatialCrop（脑部 MRI 兼容）
        5. 空间补边，确保后续裁剪最小尺寸足够
        6. 从体数据中随机采样多个基础 patch
        7. 针对每个基础 patch，生成 2 个全局裁剪和 N 个局部裁剪，
           并附加随机 resize 与增强

    参数：
        num_base_patches: 每个体数据采样多少个基础 patch。
        global_views_size: 全局视图尺寸。
        local_views_size: 局部视图尺寸。
        local_views_scale: 局部裁剪的随机尺度范围。
        num_local_views: 每个基础 patch 生成多少个局部视图。
        roi_size: 裁剪目标尺寸（CenterSpatialCrop 模式）或最小尺寸保证。
                  设为 None 时跳过裁剪，仅做 SpatialPad。
        spacing: 目标体素间距（mm）。
        dtype: "float32" 或 "float16"。
        use_foreground_crop: 使用前景裁剪替代中心裁剪。
        foreground_margin: 前景裁剪保留的边缘体素数。
    """

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

        # 基础裁剪尺寸需要足够大，以容纳最大的局部裁剪
        base_crop_size = tuple(
            int(sz * (1 / local_views_scale[0])) for sz in local_views_size
        )  # 例如：(48 / 0.25) = (192, 192, 192)

        # --- 构建确定性预处理步骤 ---
        deterministic_transforms = [
            LoadImaged(keys=("image",)),
            EnsureChannelFirstd(
                keys=("image",),
                channel_dim="no_channel",
            ),
            # MRI 强度归一化：百分位裁剪极端值
            ScaleIntensityRangePercentilesd(
                keys=("image",),
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # 重定向到标准 RAS 朝向
            Orientationd(keys=("image",), axcodes="RAS"),
            # 重采样到目标体素间距
            Spacingd(
                keys=("image",),
                pixdim=spacing,
                mode=("bilinear",),
            ),
        ]

        # --- 裁剪策略 ---
        if use_foreground_crop:
            # 全身 MRI：前景裁剪，自动去除背景
            deterministic_transforms.append(
                CropForegroundd(
                    keys=("image",),
                    source_key="image",
                    margin=foreground_margin,
                )
            )
        elif roi_size is not None:
            # 脑部 MRI 兼容模式：中心裁剪
            deterministic_transforms.append(
                CenterSpatialCropd(
                    keys=("image",),
                    roi_size=roi_size,
                )
            )

        # 补边以确保基础裁剪尺寸足够
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

        # --- 随机采样 + DINO 多裁剪 ---
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
    """生成带增强的 DINO 全局与局部裁剪。

    对每个输入 patch，会生成：
      - 2 个全局裁剪（从基础尺寸的 50%-100% 随机采样，再缩放到 global_views_size）
      - N 个局部裁剪（按 local_views_scale 随机采样，再缩放到 local_views_size）

    每个裁剪都会独立施加翻转、模糊、噪声、对比度等增强。

    相比 SPECTRE 的改动：
      - 将面向 CT HU 的 RandScaleIntensityRange 替换为
        更适合 MRI 的 RandScaleIntensity + RandBiasField
      - 将高斯核的 sigma 调整为各向同性
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

        # 全局裁剪：随机尺寸范围为基础 patch 的 50%-100%
        self.cropper_global = RandSpatialCropSamples(
            roi_size=tuple(int(local_views_scale[1] * sz) for sz in base_crop_size),
            num_samples=2,
            max_roi_size=base_crop_size,
            random_center=True,
            random_size=True,
            lazy=lazy,
        )
        # 局部裁剪：随机尺寸范围由 local_views_scale 控制
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

        # 更适合 MRI 的增强方式
        self.augmentor = Compose([
            # 随机翻转
            RandFlip(spatial_axis=0, prob=0.5),
            RandFlip(spatial_axis=1, prob=0.5),
            RandFlip(spatial_axis=2, prob=0.5),
            # 模糊 / 锐化（各向同性 sigma）
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
            # 对比度抖动
            RandAdjustContrast(gamma=(0.8, 1.2), prob=0.3),
            # 叠加高斯噪声，模拟采集噪声
            RandGaussianNoise(std=0.05, sample_std=True, prob=0.25),
            # 随机强度缩放，模拟扫描仪强度漂移
            RandScaleIntensity(factors=0.15, prob=0.25),
            # 模拟 MRI 偏置场不均匀
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

        # 兼容 RandSpatialCropSamplesd 返回的字典列表
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
