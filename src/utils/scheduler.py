"""
Learning rate and parameter schedulers.

Copied from SPECTRE (MIT License) — no modifications needed.
"""
import warnings
from typing import Optional

import numpy as np
import torch


def linear_warmup_schedule(
    step: int,
    warmup_steps: int,
    start_value: float,
    end_value: float,
) -> float:
    if warmup_steps < 0:
        raise ValueError(f"Warmup steps {warmup_steps} can't be negative.")
    if step < 0:
        raise ValueError(f"Current step number {step} can't be negative.")
    if step < warmup_steps:
        return start_value + step / warmup_steps * (end_value - start_value)
    else:
        return end_value


def cosine_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    period: Optional[int] = None,
) -> float:
    """Cosine decay from start_value to end_value."""
    if step < 0:
        raise ValueError(f"Current step number {step} can't be negative.")
    if max_steps < 1:
        raise ValueError(f"Total step number {max_steps} must be >= 1.")
    if period is None and step > max_steps:
        warnings.warn(
            f"Current step number {step} exceeds max_steps {max_steps}.",
            category=RuntimeWarning,
        )
    if period is not None and period <= 0:
        raise ValueError(f"Period {period} must be >= 1")

    decay: float
    if period is not None:
        decay = (
            end_value
            - (end_value - start_value) * (np.cos(2 * np.pi * step / period) + 1) / 2
        )
    elif max_steps == 1:
        decay = end_value
    elif step == max_steps:
        decay = end_value
    else:
        decay = (
            end_value
            - (end_value - start_value)
            * (np.cos(np.pi * step / (max_steps - 1)) + 1)
            / 2
        )
    return decay


def cosine_warmup_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    warmup_steps: int,
    warmup_start_value: float,
    warmup_end_value: Optional[float] = None,
    period: Optional[int] = None,
) -> float:
    """Cosine decay with linear warmup."""
    if warmup_steps < 0:
        raise ValueError(f"Warmup steps {warmup_steps} can't be negative.")
    if warmup_steps > max_steps:
        raise ValueError(f"Warmup steps {warmup_steps} must be <= max_steps.")

    if warmup_end_value is None:
        warmup_end_value = start_value

    if step < warmup_steps:
        return (
            warmup_start_value
            + (warmup_end_value - warmup_start_value) * (step + 1) / warmup_steps
        )
    else:
        max_steps = max_steps - warmup_steps if period is None else 1
        return cosine_schedule(
            step=step - warmup_steps,
            max_steps=max_steps,
            start_value=start_value,
            end_value=end_value,
            period=period,
        )


class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Cosine warmup scheduler for learning rate."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        last_epoch: int = -1,
        start_value: float = 1.0,
        end_value: float = 0.001,
        period: Optional[int] = None,
        verbose: bool = False,
        warmup_start_value: float = 0.0,
        warmup_end_value: Optional[float] = None,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.start_value = start_value
        self.end_value = end_value
        self.period = period
        self.warmup_start_value = warmup_start_value
        self.warmup_end_value = warmup_end_value

        super().__init__(
            optimizer=optimizer,
            lr_lambda=self.scale_lr,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def scale_lr(self, epoch: int) -> float:
        return cosine_warmup_schedule(
            step=epoch,
            max_steps=self.max_epochs,
            start_value=self.start_value,
            end_value=self.end_value,
            warmup_steps=self.warmup_epochs,
            warmup_start_value=self.warmup_start_value,
            warmup_end_value=self.warmup_end_value,
            period=self.period,
        )
