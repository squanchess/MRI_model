"""
面向 MRI 体数据的 3D 图像到 Patch Embedding 模块。

基于 SPECTRE（MIT License）修改：
  - 默认 `img_size` 从 `(128, 128, 64)` 调整为 `(96, 96, 96)`，适配各向同性 MRI
  - 默认 `patch_size` 从 `(16, 16, 8)` 调整为 `(16, 16, 16)`，使用各向同性 patch
  - 导入路径切换为 `src.*` 命名空间
"""
import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.misc import Format, nchwd_to, to_3tuple
from src.utils.modeling import resample_patch_embed


class PatchEmbed(nn.Module):
    """将 3D 图像转换为 patch embedding。

    通过单个 `Conv3d`，并令 `kernel_size = stride = patch_size`，
    将 5D 输入 `(B, C, H, W, D)` 转换为 patch 序列。

    对于各向同性 MRI（例如 1mm^3 的 T1w），通常使用 `(16, 16, 16)` 这样的等边 patch；
    如果数据是各向异性的，则应根据体素纵横比调整 `patch_size`。
    """

    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            # ---- MRI 默认配置：96^3 各向同性体数据，16^3 patch ----
            img_size: Optional[Union[int, Tuple[int, int, int]]] = (96, 96, 96),
            patch_size: Union[int, Tuple[int, int, int]] = (16, 16, 16),
            in_chans: int = 1,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            self.flatten = flatten
            self.output_fmt = Format.NCHWD
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_3tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        return img_size, grid_size, num_patches

    def set_input_size(
            self,
            img_size: Optional[Union[int, Tuple[int, int, int]]] = None,
            patch_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_3tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            with torch.no_grad():
                new_proj = nn.Conv3d(
                    self.proj.in_channels,
                    self.proj.out_channels,
                    kernel_size=new_patch_size,
                    stride=new_patch_size,
                    bias=self.proj.bias is not None,
                )
                new_proj.weight.copy_(resample_patch_embed(self.proj.weight, new_patch_size, verbose=True))
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """根据输入图像大小计算特征网格尺寸，并考虑动态 padding。"""
        if self.dynamic_img_pad:
            return (
                math.ceil(img_size[0] / self.patch_size[0]),
                math.ceil(img_size[1] / self.patch_size[1]),
                math.ceil(img_size[2] / self.patch_size[2]),
            )
        else:
            return (
                img_size[0] // self.patch_size[0],
                img_size[1] // self.patch_size[1],
                img_size[2] // self.patch_size[2],
            )

    def forward(self, x):
        _, _, H, W, D = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
                assert D == self.img_size[2], f"Input depth ({D}) doesn't match model ({self.img_size[2]})."
            elif not self.dynamic_img_pad:
                assert H % self.patch_size[0] == 0, \
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                assert W % self.patch_size[1] == 0, \
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                assert D % self.patch_size[2] == 0, \
                    f"Input depth ({D}) should be divisible by patch size ({self.patch_size[2]})."
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            pad_d = (self.patch_size[2] - D % self.patch_size[2]) % self.patch_size[2]
            x = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHWD -> NLC
        elif self.output_fmt != Format.NCHWD:
            x = nchwd_to(x, self.output_fmt)
        x = self.norm(x)
        return x
