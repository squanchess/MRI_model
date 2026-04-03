"""
DINO 投影头。

结构为 3 层 MLP + L2 归一化 + weight normalization 最后一层。
同时用于 DINO 与 DINOv2。

代码改自 SPECTRE（MIT License），核心逻辑未作修改。
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOProjectionHead(nn.Module):
    """DINO / DINOv2 使用的投影头。

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: SwAV, 2020, https://arxiv.org/abs/2006.09882

    Attributes:
        input_dim: 头部输入维度。
        hidden_dim: 隐层维度。
        bottleneck_dim: 最后一层前的瓶颈维度。
        output_dim: 输出维度。
        freeze_last_layer: 输出层冻结的 epoch 数。
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        batch_norm: bool = False,
        freeze_last_layer: int = -1,
        norm_last_layer: bool = True,
    ):
        super().__init__()

        blocks = [
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim) if batch_norm else None, nn.GELU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim) if batch_norm else None, nn.GELU()),
            (hidden_dim, bottleneck_dim, None, None),
        ]

        layers: List[nn.Module] = []
        for block in blocks:
            in_dim, out_dim, bn, non_linearity, *bias = block
            use_bias = bias[0] if bias else not bool(bn)
            layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
            if bn:
                layers.append(bn)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.freeze_last_layer = freeze_last_layer
        self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(self.last_layer)
        self.last_layer.weight_g.data.fill_(1)  # type: ignore

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """在训练早期屏蔽最后一层梯度，以提升稳定性。"""
        if current_epoch >= self.freeze_last_layer:
            return
        for param in self.last_layer.parameters():
            param.grad = None

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init._no_grad_trunc_normal_(module.weight, mean=0, std=0.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x
