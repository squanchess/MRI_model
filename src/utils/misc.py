"""
通用工具函数。

从 SPECTRE（MIT License）中整理合并而来：
  - `_utils.py`：`fix_random_seeds`、`to_ntuple` 等辅助函数
  - `modeling.py`：`Format` 枚举、`nchwd_to` / `nhwdc_to` 格式转换工具
"""
import random
from enum import Enum
from itertools import repeat
from typing import Iterable

import numpy as np
import torch


def fix_random_seeds(seed: int = 31):
    """固定随机种子，提升实验可复现性。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import monai
        monai.utils.set_determinism(seed=seed)
    except ImportError:
        pass


def _ntuple(n: int):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class Format(str, Enum):
    NCHWD = "NCHWD"
    NHWDC = "NHWDC"
    NCL = "NCL"
    NLC = "NLC"


def nchwd_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWDC:
        x = x.permute(0, 2, 3, 4, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


def nhwdc_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NCHWD:
        x = x.permute(0, 4, 1, 2, 3)
    elif fmt == Format.NLC:
        x = x.flatten(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(1, 2).transpose(1, 2)
    return x
