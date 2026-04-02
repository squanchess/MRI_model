"""
DINO and DINOv2 self-supervised learning frameworks.

Modified from SPECTRE (MIT License):
  - contrastive.py: DINO / DINOv2 teacher-student wrappers
  - ssl/models/masked_vit.py: MaskedVisionTransformer (inlined here for DINOv2)
  - Import paths updated to src.* namespace

Both DINO (Stage 1, no masking) and DINOv2 (Stage 1+, with iBOT masking)
are included. For IXI T1 initial experiments, use DINO.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn

from src.models.dino_head import DINOProjectionHead
from src.utils.modeling import (
    deactivate_requires_grad_and_to_eval, 
    update_drop_path_rate,
    mask_bool,
    get_at_index,
    mask_at_index,
    resample_abs_pos_embed,
)
from src.utils.misc import to_3tuple


# ---------------------------------------------------------------------------
# DINO (no masking — recommended for initial IXI experiments)
# ---------------------------------------------------------------------------

class DINO(nn.Module):
    """DINO self-supervised framework.

    Teacher-student architecture with:
      - Student: backbone + projection head (trained via backprop)
      - Teacher: backbone + projection head (updated via EMA)

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294

    Args:
        backbone: ViT backbone (will be deepcopied for teacher).
        input_dim: Embedding dim of the backbone output.
        hidden_dim: Hidden dim of the projection MLP.
        bottleneck_dim: Bottleneck dim before the last linear.
        output_dim: Number of prototypes (K in the paper).
        freeze_last_layer: Freeze last layer for first N epochs.
    """
    def __init__(
        self, 
        backbone: nn.Module, 
        input_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        freeze_last_layer: int = -1,
    ):
        super().__init__()

        self.backbone_student = backbone
        self.head_student = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, 
            freeze_last_layer=freeze_last_layer,
        )

        self.backbone_teacher = deepcopy(backbone)
        self.head_teacher = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim,
        )
        deactivate_requires_grad_and_to_eval(self.backbone_teacher)
        deactivate_requires_grad_and_to_eval(self.head_teacher)

    @torch.no_grad()
    def forward_teacher(
        self, 
        global_views: torch.Tensor
    ) -> torch.Tensor:
        teacher_global_cls_token = self.backbone_teacher(global_views).flatten(start_dim=1)
        teacher_global_cls_out = self.head_teacher(teacher_global_cls_token)
        return teacher_global_cls_out

    def forward_student(
        self,
        global_views: torch.Tensor,
        local_views: torch.Tensor,
    ) -> torch.Tensor:
        student_global_cls_token = self.backbone_student(global_views).flatten(start_dim=1)
        student_global_cls_out = self.head_student(student_global_cls_token)

        student_local_cls_token = self.backbone_student(local_views).flatten(start_dim=1)
        student_local_cls_out = self.head_student(student_local_cls_token)

        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out], dim=0)
        return student_cls_out
    
    def forward(
        self, 
        global_views: torch.Tensor,
        local_views: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        teacher_out = self.forward_teacher(global_views)
        student_out = self.forward_student(global_views, local_views)
        return teacher_out, student_out


# ---------------------------------------------------------------------------
# MaskedVisionTransformer (used internally by DINOv2)
# ---------------------------------------------------------------------------

class MaskedVisionTransformer(nn.Module):
    """ViT wrapper that supports token masking for iBOT.

    Handles patch embedding → prefix tokens → positional encoding → masking
    → transformer blocks → normalization.
    """

    def __init__(
        self,
        vit: "VisionTransformer",
        mask_token: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.mask_token = (
            mask_token if mask_token is not None
            else nn.Parameter(torch.zeros(1, 1, self.vit.embed_dim))
        )
        self._initialize_weights()

    @property
    def sequence_length(self) -> int:
        return self.vit.patch_embed.num_patches + self.vit.num_prefix_tokens

    def forward(
        self,
        images: torch.Tensor,
        idx_mask: Optional[torch.Tensor] = None,
        idx_keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.encode(images, idx_mask=idx_mask, idx_keep=idx_keep)
        if self.vit.attn_pool is not None:
            x = self.vit.attn_pool(x)
        elif self.vit.global_pool == "avg":
            x = x[:, self.vit.num_prefix_tokens:].mean(dim=1)
        elif self.vit.global_pool:
            x = x[:, 0]
        return x

    def encode(
        self,
        images: torch.Tensor,
        idx_mask: Optional[torch.Tensor] = None,
        idx_keep: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens, rope = self.preprocess(
            images=images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask
        )
        tokens = self.vit.norm_pre(tokens)
        for blk in self.vit.blocks:
            tokens = blk(tokens, rope=rope)
        tokens = self.vit.norm(tokens)
        return tokens

    def preprocess(
        self,
        images: torch.Tensor,
        idx_mask: Optional[torch.Tensor] = None,
        idx_keep: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if idx_mask is not None and mask is not None:
            raise ValueError("idx_mask and mask cannot both be set.")
        
        _, _, H, W, D = images.shape

        # Convert images to patch tokens
        tokens = self.vit.patch_embed(images)
        if self.vit.dynamic_img_size:
            tokens = tokens.permute(0, 4, 1, 2, 3)  # NHWDC -> NCHWD
            tokens = tokens.flatten(2).transpose(1, 2)  # -> NLC

        # Add prefix tokens
        prefix_tokens = []
        if self.vit.cls_token is not None:
            prefix_tokens.append(self.vit.cls_token.expand(tokens.shape[0], -1, -1))
        if self.vit.reg_token is not None:
            prefix_tokens.append(self.vit.reg_token.expand(tokens.shape[0], -1, -1))
        if prefix_tokens:
            tokens = torch.cat(prefix_tokens + [tokens], dim=1)

        # Apply masking
        if idx_mask is not None:
            tokens = mask_at_index(tokens=tokens, index=idx_mask, mask_token=self.mask_token)
        elif mask is not None:
            tokens = mask_bool(tokens=tokens, mask=mask, mask_token=self.mask_token)

        # Add positional encoding
        tokens, rope = self._add_pos_embed(tokens, img_size=(H, W, D))

        if idx_keep is not None:
            tokens = get_at_index(tokens, idx_keep)

        return tokens, rope

    def _add_pos_embed(
        self, x: torch.Tensor, img_size: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self.vit.pos_embed is None and self.vit.rope is None:
            return x, None
        
        if self.vit.pos_embed is not None:
            if self.vit.dynamic_img_size:
                H, W, D = to_3tuple(img_size)
                prev_grid_size = self.vit.patch_embed.grid_size
                new_size = self.vit.patch_embed.dynamic_feat_size((H, W, D))
                pos_embed = resample_abs_pos_embed(
                    self.vit.pos_embed,
                    new_size=new_size,
                    old_size=prev_grid_size,
                    num_prefix_tokens=(0 if self.vit.no_embed_class else self.vit.num_prefix_tokens),
                )
            else:
                pos_embed = self.vit.pos_embed
            
            if self.vit.no_embed_class:
                if self.vit.num_prefix_tokens:
                    prefix = x[:, :self.vit.num_prefix_tokens, :]
                    spatial = x[:, self.vit.num_prefix_tokens:, :]
                    spatial = spatial + pos_embed
                    x = torch.cat((prefix, spatial), dim=1)
                else:
                    x = x + pos_embed
            else:
                x = x + pos_embed

            x = self.vit.pos_drop(x)
            return x, None
        else:
            B = x.shape[0]
            H, W, D = to_3tuple(img_size)
            feat_h, feat_w, feat_d = self.vit.patch_embed.dynamic_feat_size((H, W, D))
            if self.vit.requires_per_sample_rope:
                rope = [self.vit.rope(H=feat_h, W=feat_w, D=feat_d) for _ in range(B)]
            else:
                rope = self.vit.rope(H=feat_h, W=feat_w, D=feat_d)
            return x, rope

    def _initialize_weights(self) -> None:
        w = self.vit.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.vit.has_class_token:
            nn.init.normal_(self.vit.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


# ---------------------------------------------------------------------------
# DINOv2 (with iBOT masking — for future upgrade)
# ---------------------------------------------------------------------------

class DINOv2(nn.Module):
    """DINOv2 self-supervised framework.

    Extends DINO with:
      - iBOT patch-level masking and loss
      - Optional separate iBOT head
      - Student-specific stochastic depth

    - [0]: DINOv2, 2023, https://arxiv.org/abs/2304.07193

    Args:
        backbone: ViT backbone.
        input_dim: Embedding dim.
        hidden_dim: Projection head hidden dim.
        bottleneck_dim: Projection head bottleneck dim.
        output_dim: Number of prototypes.
        ibot_seperate_head: Use separate projection head for iBOT.
        student_drop_path_rate: Drop path rate for student.
        freeze_last_layer: Freeze last layer for first N epochs.
    """
    def __init__(
        self, 
        backbone: "VisionTransformer", 
        input_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        ibot_seperate_head: bool = False,
        student_drop_path_rate: float = 0.1,
        freeze_last_layer: int = -1,
    ):
        super().__init__()

        self.backbone_student = MaskedVisionTransformer(vit=backbone)
        update_drop_path_rate(
            self.backbone_student.vit, 
            drop_path_rate=student_drop_path_rate
        )
        self.head_student_dino = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, 
            freeze_last_layer=freeze_last_layer,
        )
        if ibot_seperate_head:
            self.head_student_ibot = DINOProjectionHead(
                input_dim, hidden_dim, bottleneck_dim, output_dim,
                freeze_last_layer=freeze_last_layer,
            )
        else:
            self.head_student_ibot = self.head_student_dino

        self.backbone_teacher = deepcopy(self.backbone_student)
        deactivate_requires_grad_and_to_eval(self.backbone_teacher)
        self.head_teacher_dino = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim,
        )
        deactivate_requires_grad_and_to_eval(self.head_teacher_dino)
        if ibot_seperate_head:
            self.head_teacher_ibot = DINOProjectionHead(
                input_dim, hidden_dim, bottleneck_dim, output_dim,
            )
            deactivate_requires_grad_and_to_eval(self.head_teacher_ibot)
        else:
            self.head_teacher_ibot = self.head_teacher_dino
    
    @torch.no_grad()
    def forward_teacher(
        self, 
        global_views: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        teacher_features = self.backbone_teacher.encode(global_views, mask=None)
        teacher_global_cls_token = teacher_features[:, 0]
        teacher_global_cls_out = self.head_teacher_dino(teacher_global_cls_token)
        teacher_global_masked_out = self.head_teacher_ibot(teacher_features[mask])
        return teacher_global_cls_out, teacher_global_masked_out

    def forward_student(
        self, 
        global_views: torch.Tensor, 
        local_views: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Global views (with masking)
        student_features = self.backbone_student.encode(global_views, mask=mask)
        student_global_cls_token = student_features[:, 0]
        student_global_masked_features = student_features[mask]

        student_global_cls_out = self.head_student_dino(student_global_cls_token)
        student_global_masked_out = self.head_student_ibot(student_global_masked_features)

        # Local views (no masking)
        student_local_cls_token = self.backbone_student.encode(local_views, mask=None)[:, 0]
        student_local_cls_out = self.head_student_dino(student_local_cls_token)

        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out], dim=0)
        return student_cls_out, student_global_masked_out

    def forward(
        self, 
        global_views: torch.Tensor,
        local_views: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher_cls_out, teacher_masked_out = self.forward_teacher(global_views, mask)
        student_cls_out, student_masked_out = self.forward_student(global_views, local_views, mask)
        return teacher_cls_out, teacher_masked_out, student_cls_out, student_masked_out


# ---------------------------------------------------------------------------
# DINOv3 Framework (DINOv2 + Gram Anchoring)
# ---------------------------------------------------------------------------

class DINOv3(DINOv2):
    """DINOv3 framework — extends DINOv2 with Gram Anchoring.

    Gram Anchoring prevents dense patch features from degrading during long
    training schedules. It maintains a frozen "Gram Teacher" — a periodic
    snapshot of the EMA teacher — and adds an MSE loss between the Gram
    matrices (patch-token similarity structure) of student and Gram teacher.

    The Gram Teacher is updated by copying the current EMA teacher weights
    every `gram_update_freq` steps, starting after `gram_first_update_step`.
    Between updates, the Gram Teacher is completely frozen.

    Args:
        backbone: ViT backbone.
        input_dim: Embedding dimension.
        hidden_dim / bottleneck_dim / output_dim: Projection head dims.
        ibot_seperate_head: Separate heads for DINO and iBOT objectives.
        student_drop_path_rate: Drop path for student.
        freeze_last_layer: Epochs to freeze projection last layer.
        gram_update_freq: Steps between Gram Teacher updates.
        gram_first_update_step: First step to snapshot the Gram Teacher.
        gram_max_updates: Max number of Gram Teacher updates (None = unlimited).
        reg_tokens: Number of register tokens (DINOv3 trains with these from start).
    """

    def __init__(
        self,
        backbone: "VisionTransformer",
        input_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        ibot_seperate_head: bool = False,
        student_drop_path_rate: float = 0.1,
        freeze_last_layer: int = -1,
        gram_update_freq: int = 10000,
        gram_first_update_step: int = 0,
        gram_max_updates: int | None = None,
    ):
        super().__init__(
            backbone=backbone,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=output_dim,
            ibot_seperate_head=ibot_seperate_head,
            student_drop_path_rate=student_drop_path_rate,
            freeze_last_layer=freeze_last_layer,
        )

        # Gram Teacher: an independent frozen copy, periodically refreshed
        self.gram_teacher = deepcopy(self.backbone_teacher)
        deactivate_requires_grad_and_to_eval(self.gram_teacher)

        self.gram_update_freq = gram_update_freq
        self.gram_first_update_step = gram_first_update_step
        self.gram_max_updates = gram_max_updates
        self._gram_update_count = 0
        self._gram_initialized = False

    @torch.no_grad()
    def maybe_update_gram_teacher(self, global_step: int) -> bool:
        """Snapshot EMA teacher into Gram Teacher if conditions are met.

        Called every training step. Returns True if an update happened.
        """
        if global_step < self.gram_first_update_step:
            return False
        if self.gram_max_updates is not None and self._gram_update_count >= self.gram_max_updates:
            return False

        do_update = False
        if not self._gram_initialized:
            # First snapshot: always update
            do_update = True
            self._gram_initialized = True
        elif (global_step - self.gram_first_update_step) % self.gram_update_freq == 0:
            do_update = True

        if do_update:
            # Copy EMA teacher weights into Gram teacher
            for gt_param, t_param in zip(
                self.gram_teacher.parameters(), self.backbone_teacher.parameters()
            ):
                gt_param.data.copy_(t_param.data)
            self.gram_teacher.eval()
            self._gram_update_count += 1
            return True

        return False

    @torch.no_grad()
    def get_gram_features(
        self,
        global_views: torch.Tensor,
        student_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract patch tokens from student and Gram teacher for Gram loss.

        Args:
            global_views: (B, C, H, W, D) input images.
            student_features: (B, seq_len, D) full student output (cls + patches).

        Returns:
            student_patches: (B, num_patches, D) student patch tokens.
            gram_teacher_patches: (B, num_patches, D) Gram teacher patch tokens.
        """
        # Student: strip CLS and register tokens, keep only patch tokens
        n_prefix = self.backbone_student.vit.num_prefix_tokens
        student_patches = student_features[:, n_prefix:]  # (B, P, D)

        # Gram teacher: forward pass with frozen weights
        gram_features = self.gram_teacher.encode(global_views, mask=None)
        gram_teacher_patches = gram_features[:, n_prefix:]  # (B, P, D)

        return student_patches, gram_teacher_patches

    def forward(
        self,
        global_views: torch.Tensor,
        local_views: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """Forward pass returning all outputs needed for DINOv3 losses.

        Returns a dict with:
            teacher_cls_out: Teacher CLS token projections.
            teacher_masked_out: Teacher masked patch projections.
            student_cls_out: Student CLS token projections (global + local).
            student_masked_out: Student masked patch projections.
            student_patches: Student patch tokens (pre-head) for Gram loss.
            gram_teacher_patches: Gram teacher patch tokens for Gram loss.
        """
        # Standard DINOv2 outputs
        teacher_cls_out, teacher_masked_out = self.forward_teacher(global_views, mask)

        # Student forward — need raw features for Gram loss
        student_features = self.backbone_student.encode(global_views, mask=mask)
        student_global_cls = student_features[:, 0]
        student_global_masked = student_features[mask]

        student_global_cls_out = self.head_student_dino(student_global_cls)
        student_global_masked_out = self.head_student_ibot(student_global_masked)

        student_local_cls = self.backbone_student.encode(local_views, mask=None)[:, 0]
        student_local_cls_out = self.head_student_dino(student_local_cls)

        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out], dim=0)

        # Gram features — student patches vs Gram teacher patches
        student_patches, gram_teacher_patches = self.get_gram_features(
            global_views, student_features
        )

        return {
            "teacher_cls_out": teacher_cls_out,
            "teacher_masked_out": teacher_masked_out,
            "student_cls_out": student_cls_out,
            "student_masked_out": student_global_masked_out,
            "student_patches": student_patches,
            "gram_teacher_patches": gram_teacher_patches,
        }
