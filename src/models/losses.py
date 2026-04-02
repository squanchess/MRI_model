"""
Self-supervised losses for DINO / DINOv2.

Merged from SPECTRE (MIT License):
  - _center.py: Center module with EMA update (supports distributed all_reduce)
  - dino_loss.py: DINO cross-entropy loss with teacher temperature warmup
  - ibot_loss.py: iBOT patch-level loss (for DINOv2)
  - koleo_loss.py: KoLeo regularizer (for DINOv2)

All losses are modality-agnostic — no modifications needed for MRI.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Center (EMA-tracked running mean for teacher centering)
# ---------------------------------------------------------------------------

class Center(nn.Module):
    """Center module for teacher output centering in DINO.

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294

    Attributes:
        size: Shape of the center tensor. Dims to reduce should be 1.
        mode: Center computation mode. Only 'mean' is supported.
        momentum: EMA momentum for center updates.
    """

    def __init__(
        self,
        size: Tuple[int, ...],
        mode: str = "mean",
        momentum: float = 0.9,
    ) -> None:
        super().__init__()

        center_fn = _CENTER_MODE_TO_FUNCTION.get(mode)
        if center_fn is None:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes: {sorted(_CENTER_MODE_TO_FUNCTION.keys())}."
            )
        self._center_fn = center_fn

        self.size = size
        self.dim = tuple(i for i, s in enumerate(size) if s == 1)
        self.register_buffer("center", torch.zeros(self.size))
        self.momentum = momentum

    @property
    def value(self) -> torch.Tensor:
        return self.center

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_center = self._center_fn(x=x, dim=self.dim)
        self.center = _center_momentum(
            center=self.center, batch_center=batch_center, momentum=self.momentum
        )


@torch.no_grad()
def _center_mean(x: torch.Tensor, dim: Tuple[int, ...]) -> torch.Tensor:
    batch_center = torch.mean(x, dim=dim, keepdim=True)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(batch_center)
        batch_center = batch_center / dist.get_world_size()
    return batch_center


@torch.no_grad()
def _center_momentum(
    center: torch.Tensor, batch_center: torch.Tensor, momentum: float
) -> torch.Tensor:
    return center * momentum + batch_center * (1 - momentum)


_CENTER_MODE_TO_FUNCTION = {
    "mean": _center_mean,
}


# ---------------------------------------------------------------------------
# DINO Loss
# ---------------------------------------------------------------------------

class DINOLoss(nn.Module):
    """Implementation of the DINO loss.

    Cross-entropy between softmax outputs of teacher and student networks
    with teacher centering and temperature warmup.

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294

    Attributes:
        output_dim: Dimension of the projection head output.
        teacher_temp: Target temperature for the teacher network.
        student_temp: Temperature for the student network.
        warmup_teacher_temp_epochs: Number of warmup epochs for teacher temperature.
    """

    def __init__(
        self,
        output_dim: int = 65536,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        center_mode: str = "mean",
    ) -> None:
        super().__init__()

        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

        self.center = Center(
            size=(1, 1, output_dim),
            mode=center_mode,
            momentum=center_momentum,
        )

        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp_schedule = torch.linspace(
            start=warmup_teacher_temp,
            end=teacher_temp,
            steps=warmup_teacher_temp_epochs,
        )

    def forward(
        self,
        teacher_out: list[torch.Tensor],
        student_out: list[torch.Tensor],
        teacher_temp: float | None = None,
        epoch: int | None = None,
    ) -> torch.Tensor:
        """Cross-entropy between teacher and student softmax outputs.

        Args:
            teacher_out: List of tensors (batch_size, output_dim), one per view.
            student_out: List of tensors (batch_size, output_dim), one per view.
            teacher_temp: Override temperature. If None, uses schedule or default.
            epoch: Current epoch for temperature warmup schedule.

        Returns:
            Average cross-entropy loss.
        """
        # Get teacher temperature
        if teacher_temp is not None:
            teacher_temperature = torch.tensor(teacher_temp)
        elif epoch is not None:
            if epoch < self.warmup_teacher_temp_epochs:
                teacher_temperature = self.teacher_temp_schedule[epoch]
            else:
                teacher_temperature = torch.tensor(self.teacher_temp)
        else:
            teacher_temperature = torch.tensor(self.teacher_temp)
        teacher_temperature = teacher_temperature.to(teacher_out[0].device)

        # Calculate cross-entropy loss
        teacher_out_stacked = torch.stack(teacher_out)
        t_out: torch.Tensor = F.softmax(
            (teacher_out_stacked - self.center.value) / teacher_temperature, dim=-1
        )
        student_out_stacked = torch.stack(student_out)
        s_out = F.log_softmax(student_out_stacked / self.student_temp, dim=-1)

        # Feature similarities, ignoring the diagonal (same-view pairs)
        # b = batch_size, t = n_views_teacher, s = n_views_student, d = output_dim
        loss = -torch.einsum("tbd,sbd->ts", t_out, s_out)
        loss.fill_diagonal_(0)

        n_terms = loss.numel() - loss.diagonal().numel()
        batch_size = teacher_out_stacked.shape[1]

        loss = loss.sum() / (n_terms * batch_size)

        # Update the center
        self.center.update(teacher_out_stacked)

        return loss


# ---------------------------------------------------------------------------
# iBOT Patch Loss (for DINOv2)
# ---------------------------------------------------------------------------

class iBOTPatchLoss(nn.Module):
    """iBOT patch-level loss as used in DINOv2.

    - [0]: iBOT, 2021, https://arxiv.org/abs/2111.07832
    - [1]: DINOv2, 2023, https://arxiv.org/abs/2304.07193
    """

    def __init__(
        self,
        output_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_mode: str = "mean",
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center = Center(
            size=(1, output_dim),
            mode=center_mode,
            momentum=center_momentum,
        )

    def forward(
        self,
        teacher_out: torch.Tensor,
        student_out: torch.Tensor,
        mask: torch.Tensor,
        teacher_temp: float | None = None,
    ) -> torch.Tensor:
        """
        Args:
            teacher_out: (B*N, D) teacher features of masked tokens.
            student_out: (B*N, D) student features of masked tokens.
            mask: (B, H, W, D) boolean mask.
            teacher_temp: Optional override temperature.
        """
        teacher_temperature = torch.tensor(
            teacher_temp if teacher_temp is not None else self.teacher_temp
        )

        teacher_softmax = F.softmax(
            (teacher_out - self.center.value) / teacher_temperature, dim=-1
        )
        student_log_softmax = F.log_softmax(student_out / self.student_temp, dim=-1)

        loss = -torch.sum(teacher_softmax * student_log_softmax, dim=-1)

        # Weight by inverse of masked tokens per image
        num_masked_per_image = mask.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        weight = (1.0 / num_masked_per_image).expand_as(mask)[mask]

        B = mask.shape[0]
        loss = (loss * weight).sum() / B

        self.center.update(teacher_out)
        return loss


# ---------------------------------------------------------------------------
# KoLeo Loss (for DINOv2 regularization)
# ---------------------------------------------------------------------------

class KoLeoLoss(nn.Module):
    """KoLeo loss — encourages uniform span of features in a batch.

    - [0]: Spreading vectors for similarity search, 2019, https://arxiv.org/abs/1806.03198
    """

    def __init__(self, p: float = 2, eps: float = 1e-8):
        super().__init__()
        self.p = p
        self.eps = eps
        self.pairwise_distance = nn.PairwiseDistance(p=p, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, embedding_size)
        Returns:
            Scalar loss.
        """
        x = F.normalize(x, p=2, dim=-1, eps=self.eps)
        cos_sim = torch.mm(x, x.t())
        cos_sim.fill_diagonal_(-2)
        nn_idx = cos_sim.argmax(dim=1)
        nn_dist: torch.Tensor = self.pairwise_distance(x, x[nn_idx])
        loss = -(nn_dist + self.eps).log().mean()
        return loss


# ---------------------------------------------------------------------------
# Gram Anchoring Loss (DINOv3)
# ---------------------------------------------------------------------------

class GramLoss(nn.Module):
    """Gram Anchoring loss from DINOv3.

    Prevents dense feature degradation during long training by anchoring the
    patch-token similarity structure (Gram matrix) to a frozen reference.

    Computes MSE between the Gram matrices (pairwise cosine similarity) of
    student and teacher patch tokens.

    From: facebookresearch/dinov3 (DINOv3 License)
    Paper: https://arxiv.org/abs/2508.10104

    Args:
        apply_norm: L2-normalize features before computing Gram matrix.
        img_level: If True, compute Gram per image (B, N, D) → (B, N, N).
                   If False, flatten to (B*N, D) and compute one big Gram.
        remove_neg: Zero out negative cosine similarities in both student and teacher.
        remove_only_teacher_neg: Zero out negatives only in teacher (and matching positions in student).
    """

    def __init__(
        self,
        apply_norm: bool = True,
        img_level: bool = True,
        remove_neg: bool = False,
        remove_only_teacher_neg: bool = False,
    ):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

        if self.remove_neg and self.remove_only_teacher_neg:
            raise ValueError("remove_neg and remove_only_teacher_neg are mutually exclusive")

    def forward(
        self,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
        img_level: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            student_feats: (B, N, D) patch token features from student backbone.
            teacher_feats: (B, N, D) patch token features from Gram teacher (frozen reference).
            img_level: If True, Gram is per-image. If False, flatten across batch.

        Returns:
            Scalar MSE loss between student and teacher Gram matrices.
        """
        if img_level:
            assert student_feats.ndim == 3 and teacher_feats.ndim == 3

        student_feats = student_feats.float()
        teacher_feats = teacher_feats.float()

        # Normalize
        if self.apply_norm:
            teacher_feats = F.normalize(teacher_feats, dim=-1)
        if not img_level and teacher_feats.ndim == 3:
            teacher_feats = teacher_feats.flatten(0, 1)

        # Teacher Gram matrix
        teacher_sim = torch.matmul(teacher_feats, teacher_feats.transpose(-1, -2))

        if self.apply_norm:
            student_feats = F.normalize(student_feats, dim=-1)
        if not img_level and student_feats.ndim == 3:
            student_feats = student_feats.flatten(0, 1)

        # Student Gram matrix
        student_sim = torch.matmul(student_feats, student_feats.transpose(-1, -2))

        # Optional: remove negative similarities
        if self.remove_neg:
            teacher_sim = teacher_sim.clamp(min=0.0)
            student_sim = student_sim.clamp(min=0.0)
        elif self.remove_only_teacher_neg:
            neg_mask = teacher_sim < 0
            teacher_sim[neg_mask] = 0.0
            student_sim[neg_mask] = 0.0

        return self.mse_loss(student_sim, teacher_sim)
