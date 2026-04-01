"""
LayerNorm for channels of 3D spatial NCHWD tensors.

Copied from SPECTRE (MIT License) — no modifications needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.fast_norm import is_fast_norm, fast_layer_norm


class LayerNorm3d(nn.LayerNorm):
    """LayerNorm for channels of '3D' spatial NCHWD tensors."""
    _fast_norm: torch.jit.Final[bool]

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x
