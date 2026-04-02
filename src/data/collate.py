"""
DINO collate function.

Takes the list of per-sample dicts produced by DINOTransform and
stacks global/local views into batched tensors.

Copied from SPECTRE (MIT License) — removed SigLIP collate.
"""
from typing import List

import torch
from monai.data import list_data_collate


def collate_dino(samples_list: List) -> dict:
    """Collate DINO multi-crop samples into batched tensors.

    Input: list of dicts, each with keys:
        - "image_global_views": list of 2 tensors (C, H, W, D)
        - "image_local_views":  list of N tensors (C, h, w, d)

    Output: dict with:
        - "global_views": (2*B, C, H, W, D)
        - "local_views":  (N*B, C, h, w, d)
    """
    collated_data = list_data_collate(samples_list)

    global_views = torch.cat(collated_data["image_global_views"], dim=0)
    local_views = torch.cat(collated_data["image_local_views"], dim=0)

    return {
        "global_views": global_views,
        "local_views": local_views,
    }
