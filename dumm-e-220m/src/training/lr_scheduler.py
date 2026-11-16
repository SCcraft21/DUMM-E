"""Cosine Annealing Learning Rate Scheduler with Linear Warmup."""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupLRScheduler(_LRScheduler):
    """Cosine Annealing Learning Rate Scheduler with Linear Warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max(self.warmup_steps, max_steps)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = max(0, self.last_epoch)

        if step < self.warmup_steps:
            progress = step / self.warmup_steps
            return [
                self.min_lr + (base_lr - self.min_lr) * progress
                for base_lr in self.base_lrs
            ]

        if step >= self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        decay_steps = self.max_steps - self.warmup_steps
        current_step = step - self.warmup_steps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * current_step / decay_steps))

        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_factor
            for base_lr in self.base_lrs
        ]

