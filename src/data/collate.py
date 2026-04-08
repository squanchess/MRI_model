"""
DINO 的批处理拼接函数。

接收由 DINOTransform 为每个样本生成的字典列表，
并将全局视图与局部视图分别堆叠成 batch 张量。

代码改自 SPECTRE（MIT License），已移除 SigLIP 的 collate 逻辑。
"""
from typing import List

import torch
from monai.data import list_data_collate


def collate_dino(samples_list: List) -> dict:
    """将 DINO 多裁剪样本整理为批量张量。

    输入：由字典组成的列表，每个字典包含：
        - "image_global_views": list of 2 tensors (C, H, W, D)
        - "image_local_views":  list of N tensors (C, h, w, d)

    输出：包含以下字段的字典：
        - "global_views": (2*B, C, H, W, D)
        - "local_views":  (N*B, C, h, w, d)
    """
    collated_data = list_data_collate(samples_list)

    # 所有样本在第0维拼接
    global_views = torch.cat(collated_data["image_global_views"], dim=0)
    local_views = torch.cat(collated_data["image_local_views"], dim=0)

    return {
        "global_views": global_views,
        "local_views": local_views,
    }
