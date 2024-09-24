import math
from abc import ABC

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class BaseSchedule(ABC):

    def __init__(self):
        pass

    def __call__(self, cur_step):
        return self.get_lr(cur_step)

    def get_lr(self, cur_step):
        raise NotImplementedError


class FixedSchedule(BaseSchedule):

    def __init__(self, value: float):
        super().__init__()

        self.value = value

    def get_lr(self, cur_step):
        return self.value


class TransformerSchedule(BaseSchedule):

    def __init__(self, d_model: int, warmup_steps: int, factor: float, world_size: int):
        super().__init__()

        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.world_size = world_size

    def get_lr(self, cur_step):
        if cur_step == 0:
            cur_step = 1
        else:
            cur_step *= self.world_size

        return (self.d_model ** -0.5 * min(cur_step ** -0.5, cur_step * self.warmup_steps ** -1.5)) * self.factor


class CosineSchedule(BaseSchedule):
    """Take from https://huggingface.co/docs/transformers/main_classes/optimizer_schedules"""

    def __init__(self, d_model: int, warmup_steps: int, max_steps: int, factor: float, world_size: int, num_cycles=0.5):
        super().__init__()

        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.num_cycles = num_cycles
        self.factor = factor
        self.world_size = world_size

    def get_lr(self, cur_step) -> float:
        if cur_step == 0:
            cur_step = 1
        else:
            cur_step *= self.world_size

        if cur_step < self.warmup_steps:
            return float(cur_step) / float(max(1, self.warmup_steps))
        progress = float(cur_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))


def get_learning_rate_scheduler(optimiser: Optimizer, learning_rate: BaseSchedule, last_epoch: int = -1):
    return LambdaLR(optimiser, learning_rate, last_epoch)
