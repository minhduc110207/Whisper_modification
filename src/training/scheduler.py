"""
Learning rate scheduler with warmup.
"""
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.
    LR increases linearly during warmup, then decays following cosine schedule.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 500,
        total_steps: int = 10000,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return [
            max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs
        ]
