"""
Multi-Head / Multi-Query / Multi-Latent Attention with 3D RoPE support.

Copied from SPECTRE (MIT License) — import path updated.
"""
from typing import Type, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.layers import use_fused_attn

from src.models.layers.rotary_pos_embed import rope_apply


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mode: str = "mha",
            q_proj_dim: Optional[int] = None,
            kv_proj_dim: Optional[int] = None,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.mode = mode.lower()
        assert self.mode in ["mha", "mqa", "mla"], "Attention mode must be 'mha', 'mqa', or 'mla'"
        assert not (self.mode == "mla" and kv_proj_dim is None), "kv_proj_dim must be provided for 'mla' mode"
        assert not (self.mode == "mla" and q_proj_dim is None), "q_proj_dim must be provided for 'mla' mode"
        
        if self.mode == "mha":
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
            self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        elif self.mode == "mqa":
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, 2 * self.head_dim, bias=qkv_bias)
            self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        elif self.mode == "mla":
            self.q_proj = nn.Linear(dim, q_proj_dim, bias=qkv_bias)
            self.kv_proj = nn.Linear(dim, kv_proj_dim, bias=qkv_bias)
            self.q_norm = norm_layer(q_proj_dim) if qk_norm else nn.Identity()
            self.kv_norm = norm_layer(kv_proj_dim) if qk_norm else nn.Identity()
            self.q = nn.Linear(q_proj_dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(kv_proj_dim, 2 * dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rotary_pos_emb(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_dtype, k_dtype = q.dtype, k.dtype
        sin, cos = rope

        if sin.ndim == 2:
            sin = sin.unsqueeze(0).unsqueeze(0)
            cos = cos.unsqueeze(0).unsqueeze(0)
        elif sin.ndim == 3:
            sin = sin.unsqueeze(1)
            cos = cos.unsqueeze(1)
        else:
            raise ValueError("RoPE sin/cos must be of shape [N, head_dim] or [B, N, head_dim]")
        
        rope_dtype = sin.dtype

        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)

        N = q.shape[-2]
        N_spatial = sin.shape[-2]
        prefix = N - N_spatial
        assert prefix >= 0, "RoPE sin/cos length exceeds sequence length"

        if prefix > 0:
            q_prefix = q[:, :, :prefix, :]
            k_prefix = k[:, :, :prefix, :]
            q_spatial = q[:, :, prefix:, :]
            k_spatial = k[:, :, prefix:, :]
        else:
            q_prefix = k_prefix = None
            q_spatial, k_spatial = q, k

        q_spatial = rope_apply(q_spatial, sin, cos)
        k_spatial = rope_apply(k_spatial, sin, cos)

        if prefix > 0:
            q = torch.cat((q_prefix, q_spatial), dim=-2)
            k = torch.cat((k_prefix, k_spatial), dim=-2)
        else:
            q, k = q_spatial, k_spatial

        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)

        return q, k

    def compute_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        if self.mode == "mha":
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
        elif self.mode == "mqa":
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, N, 2, 1, self.head_dim).permute(2, 0, 3, 1, 4)
            kv = kv.expand(-1, -1, self.num_heads, -1, -1)
            k, v = kv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
        elif self.mode == "mla":
            q = self.q_proj(x)
            kv = self.kv_proj(x)
            q, kv = self.q_norm(q), self.kv_norm(kv)
            q = self.q(q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(kv).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
        return q, k, v
        
    def compute_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
    ) -> torch.Tensor:
        B, _, N, _ = q.shape
        C = self.num_heads * self.head_dim

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        return x.transpose(1, 2).reshape(B, N, C)

    def forward(
        self, 
        x: torch.Tensor, 
        rope=None,
    ) -> torch.Tensor:

        q, k, v = self.compute_qkv(x)

        if rope is not None:
            if isinstance(rope, list):
                rope = tuple(torch.stack([r[i] for r in rope], dim=0) for i in range(2))
            q, k = self.apply_rotary_pos_emb(q, k, rope)
        
        x = self.compute_attention(q, k, v)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
