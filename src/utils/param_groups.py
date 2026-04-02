"""
Parameter group construction with layer-wise learning rate decay (LLRD).

Copied from SPECTRE (MIT License) — removed SigLIP/LoRA-specific branches.
"""
import torch.nn as nn


def get_vit_lr_decay_rate(
    name: str,
    llrd_factor: float = 1.0,
    num_layers: int = 12,
    force_is_backbone: bool = False,
    shift: int = 0,
) -> float:
    """Get LLRD multiplier for a named parameter."""
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".reg_token" in name
        ):
            layer_id = 0
        elif ".blocks." in name:
            layer_id = int(name[name.find(".blocks."):].split(".")[2]) + 1 + shift
    return llrd_factor ** (num_layers + 1 - layer_id)


def get_param_groups_with_decay(
    model: nn.Module,
    llrd_factor: float = 1.0,
    patch_embed_lr_mult: float = 1.0,
    projection_head_wd_mult: float = 1.0,
    num_layers: int | None = None,
):
    """Build optimizer parameter groups with LLRD and selective weight decay.

    - Layer-wise LR decay: deeper layers get higher LR.
    - No weight decay on biases, norms, learned tokens/embeddings.
    - Separate multipliers for patch_embed and projection heads.
    """
    force_is_backbone = False
    if num_layers is not None:
        num_layers = num_layers
    elif hasattr(model, "blocks"):
        num_layers = len(model.blocks)
        force_is_backbone = True
    elif hasattr(model, "backbone_student") and hasattr(model.backbone_student, "blocks"):
        num_layers = len(model.backbone_student.blocks)
    elif (
        hasattr(model, "backbone_student")
        and hasattr(model.backbone_student, "vit")
        and hasattr(model.backbone_student.vit, "blocks")
    ):
        num_layers = len(model.backbone_student.vit.blocks)
    else:
        num_layers = 0

    all_param_groups = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        llrd_rate = get_vit_lr_decay_rate(
            n, llrd_factor, num_layers, force_is_backbone,
        )

        d = {
            "name": n,
            "params": p,
            "lr_mult": llrd_rate,
            "wd_mult": 1.0,
        }

        if "head" in n or "projection" in n:
            d["wd_mult"] = projection_head_wd_mult

        # No weight decay on biases, norms, layer scale, learned tokens
        if n.endswith("bias") or "norm" in n or "gamma" in n:
            d["wd_mult"] = 0.0

        if "patch_embed" in n:
            d["lr_mult"] *= patch_embed_lr_mult

        all_param_groups.append(d)

    return all_param_groups
