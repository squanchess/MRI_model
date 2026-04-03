"""
模型相关工具函数。

代码改自 SPECTRE（MIT License），仅保留 DINO 与 ViT 所需部分，
并移除了 SigLIP 相关辅助函数（如 `last_token_pool`、`cat_keep_shapes` 等）。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Teacher-student 辅助函数（DINO EMA）
# ---------------------------------------------------------------------------

def deactivate_requires_grad_and_to_eval(model: nn.Module):
    """冻结所有参数，并将模型切换到 eval 模式。"""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def activate_requires_grad_and_to_train(model: nn.Module):
    """解冻所有参数，并将模型切换到 train 模式。"""
    for param in model.parameters():
        param.requires_grad = True
    model.train()


@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """执行 EMA 更新：`model_ema = m * model_ema + (1 - m) * model`。"""
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)


def update_drop_path_rate(
    model: "VisionTransformer",
    drop_path_rate: float,
    mode: str = "linear",
) -> None:
    """更新所有 block 上的 drop path 概率。"""
    from timm.layers import DropPath

    total_depth = len(model.blocks)

    if mode == "linear":
        drop_probabilities = np.linspace(0, drop_path_rate, total_depth)
    elif mode == "uniform":
        drop_probabilities = [drop_path_rate for _ in range(total_depth)]
    else:
        raise ValueError(f"Unknown mode: '{mode}'")

    for block, drop_prob in zip(model.blocks, drop_probabilities):
        if drop_prob > 0.0:
            block.drop_path1 = DropPath(drop_prob=drop_prob)
            block.drop_path2 = DropPath(drop_prob=drop_prob)
        else:
            block.drop_path1 = nn.Identity()
            block.drop_path2 = nn.Identity()


# ---------------------------------------------------------------------------
# Token 操作辅助函数
# ---------------------------------------------------------------------------

def repeat_token(token: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    batch_size, sequence_length = size
    return token.repeat(batch_size, sequence_length, 1)


def expand_index_like(index: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    dim = tokens.shape[-1]
    index = index.unsqueeze(-1).expand(-1, -1, dim)
    return index


def get_at_index(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    index = expand_index_like(index, tokens)
    return torch.gather(tokens, 1, index)


def set_at_index(
    tokens: torch.Tensor, index: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    index = expand_index_like(index, tokens)
    return torch.scatter(tokens, 1, index, value)


def mask_at_index(
    tokens: torch.Tensor, index: torch.Tensor, mask_token: torch.Tensor
) -> torch.Tensor:
    mask = tokens.new_zeros(tokens.shape)
    mask = set_at_index(mask, index, 1)
    return (1 - mask) * tokens + mask * mask_token


def mask_bool(
    tokens: torch.Tensor, mask: torch.Tensor, mask_token: torch.Tensor
) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(torch.bool).to(torch.int)
    return (1 - mask) * tokens + mask * mask_token


def patchify(images: torch.Tensor, patch_size: Tuple[int, int, int]) -> torch.Tensor:
    """将一批图像切分为 patch 序列。

    Args:
        images: (B, C, H, W, D)
        patch_size: (ph, pw, pd)

    Returns:
        (B, num_patches, C * prod(patch_size))
    """
    N, C, H, W, D = images.shape
    assert (
        H % patch_size[0] == 0
        and W % patch_size[1] == 0
        and D % patch_size[2] == 0
    ), "Image dims must be multiples of patch size."

    patch_h = H // patch_size[0]
    patch_w = W // patch_size[1]
    patch_d = D // patch_size[2]

    num_patches = patch_h * patch_w * patch_d
    patches = images.reshape(shape=(
        N, C,
        patch_h, patch_size[0],
        patch_w, patch_size[1],
        patch_d, patch_size[2],
    ))
    patches = torch.einsum("nchpwqdr->nhwdpqrc", patches)
    patches = patches.reshape(shape=(N, num_patches, math.prod(patch_size) * C))
    return patches


def random_token_mask(
    size: Tuple[int, int],
    mask_ratio: float = 0.6,
    mask_class_token: bool = False,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    batch_size, sequence_length = size
    num_keep = int(sequence_length * (1 - mask_ratio))

    noise = torch.rand(batch_size, sequence_length, device=device)
    if not mask_class_token and sequence_length > 0:
        noise[:, 0] = -1
        num_keep = max(1, num_keep)

    indices = torch.argsort(noise, dim=1)
    idx_keep = indices[:, :num_keep]
    idx_mask = indices[:, num_keep:]
    return idx_keep, idx_mask


# ---------------------------------------------------------------------------
# 位置编码重采样（3D 三线性插值）
# ---------------------------------------------------------------------------

def resample_abs_pos_embed(
        posemb: torch.Tensor,
        new_size: List[int],
        old_size: List[int],
        num_prefix_tokens: int = 1,
        interpolation: str = "trilinear",
):
    """通过 3D 插值重采样绝对位置编码。"""
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] * new_size[2] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()
    posemb = posemb.reshape(1, old_size[0], old_size[1], old_size[2], -1).permute(0, 4, 1, 2, 3)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation)
    posemb = posemb.permute(0, 2, 3, 4, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)
    return posemb


def resample_abs_pos_embed_nhwdc(
        posemb: torch.Tensor,
        new_size: List[int],
        interpolation: str = "trilinear",
):
    if new_size[0] == posemb.shape[-4] and new_size[1] == posemb.shape[-3] and new_size[2] == posemb.shape[-2]:
        return posemb

    orig_dtype = posemb.dtype
    posemb = posemb.float()
    posemb = posemb.reshape(
        1, posemb.shape[-4], posemb.shape[-3], posemb.shape[-2], posemb.shape[-1]
    ).permute(0, 4, 1, 2, 3)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation)
    posemb = posemb.permute(0, 2, 3, 4, 1).to(orig_dtype)
    return posemb


def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = "trilinear",
):
    """将 patch embedding 卷积核重采样到新的 patch 尺寸。"""
    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    assert len(patch_embed.shape) == 5, "Five dimensions expected"
    assert len(new_size) == 3, "New shape should only be (height, width, depth)"
    old_size = patch_embed.shape[-3:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.0
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.tensor(np.linalg.pinv(resize_mat.T), device=patch_embed.device)

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    orig_dtype = patch_embed.dtype
    patch_embed = patch_embed.float()
    patch_embed = v_resample_kernel(patch_embed)
    patch_embed = patch_embed.to(orig_dtype)
    return patch_embed


# ---------------------------------------------------------------------------
# 特征索引选择与池化
# ---------------------------------------------------------------------------

def feature_take_indices(
        num_features: int,
        indices: Optional[Union[int, List[int]]] = None,
        as_set: bool = False,
) -> Tuple[List[int], int]:
    if indices is None:
        indices = num_features

    if isinstance(indices, int):
        assert 0 < indices <= num_features
        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices: List[int] = []
        for i in indices:
            idx = num_features + i if i < 0 else i
            assert 0 <= idx < num_features
            take_indices.append(idx)

    if not torch.jit.is_scripting() and as_set:
        return set(take_indices), max(take_indices)

    return take_indices, max(take_indices)


def global_pool_nlc(
        x: torch.Tensor,
        pool_type: str = "token",
        num_prefix_tokens: int = 1,
        reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == "token":
        x = x[:, 0]
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == "avg":
            x = x.mean(dim=1)
        elif pool_type == "avgmax":
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == "max":
            x = x.amax(dim=1)
        else:
            assert not pool_type, f"Unknown pool type {pool_type}"

    return x
