"""
支持分布式 RNG 状态保存与恢复的 checkpoint 工具。

代码改自 SPECTRE（MIT License），已去除 CT 相关的特定提取逻辑。
"""
import os
import random
import warnings
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist


def _get_local_rng_state() -> dict:
    state = {
        "torch": torch.get_rng_state().cpu(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        cuda_states = [s.cpu() for s in torch.cuda.get_rng_state_all()]
        state["cuda"] = cuda_states
    else:
        state["cuda"] = None
    return state


def _set_local_rng_state(state: dict) -> None:
    if state is None:
        return
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and state["cuda"] is not None and torch.cuda.is_available():
        try:
            cuda_states = [s.cuda() for s in state["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception:
            for i, s in enumerate(state["cuda"]):
                try:
                    torch.cuda.set_rng_state(s.cuda(), device=i)
                except Exception:
                    pass
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    if "random" in state and state["random"] is not None:
        random.setstate(state["random"])


def save_state(ckpt_path: str, epoch: Optional[int] = None, **named_objects: Any) -> None:
    """保存 checkpoint，包括 epoch、各对象 state_dict，以及各 rank 的 RNG 状态。

    在分布式模式下，会先收集所有 rank 的 RNG 状态，最终仅由 rank 0 写盘。
    """
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    local_rng = _get_local_rng_state()

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        all_states = [None] * world_size
        dist.all_gather_object(all_states, local_rng)

        if rank == 0:
            checkpoint = {}
            if epoch is not None:
                checkpoint["epoch"] = epoch
            checkpoint["rng_states"] = all_states
            for name, obj in named_objects.items():
                checkpoint[name] = obj.state_dict()
            torch.save(checkpoint, ckpt_path)
        dist.barrier()
    else:
        checkpoint = {}
        if epoch is not None:
            checkpoint["epoch"] = epoch
        checkpoint["rng_states"] = [local_rng]
        for name, obj in named_objects.items():
            checkpoint[name] = obj.state_dict()
        torch.save(checkpoint, ckpt_path)


def load_state(ckpt_path: str, **named_objects: Any) -> int:
    """加载 checkpoint。若存在 epoch 字段则返回其值，否则返回 0。"""
    if not os.path.isfile(ckpt_path):
        warnings.warn(f"Checkpoint file not found: {ckpt_path}")
        return 0

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch", 0)

    for name, obj in named_objects.items():
        if name in checkpoint:
            try:
                obj.load_state_dict(checkpoint[name])
            except Exception as e:
                warnings.warn(f"Failed to load state_dict for '{name}': {e}")
        else:
            warnings.warn(f"No state_dict found for '{name}' in checkpoint.")

    rng_states = checkpoint.get("rng_states", None)
    if rng_states is not None:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            my_state = rng_states[rank] if rank < len(rng_states) else None
        else:
            my_state = rng_states[0] if len(rng_states) > 0 else None
        try:
            _set_local_rng_state(my_state)
        except Exception as e:
            warnings.warn(f"Failed to restore RNG state: {e}")
    else:
        warnings.warn("No 'rng_states' found in checkpoint; RNGs not restored.")

    return epoch
