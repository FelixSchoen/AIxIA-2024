from abc import ABC
from pathlib import Path

import torch
from torch import nn

from paul3.main.distributed import get_environment_variables
from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings

LOGGER = get_logger(__name__)


class BaseManager(ABC):

    def __init__(self, model: nn.Module, model_general_settings: dict, model_hyperparameters: dict, model_settings: dict,
                 inference_settings: dict,
                 device: torch.device):
        super().__init__()

        self.model = model
        self.model_general_settings = model_general_settings
        self.model_hyperparameters = model_hyperparameters
        self.model_settings = model_settings
        self.inference_settings = inference_settings
        self.device = device
        self.device_properties = torch.cuda.get_device_properties(device)

        global_rank, local_rank, world_size = get_environment_variables()
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size

    def save_model(self, path: Path, i_step, optimiser, scheduler, identifier):
        output_path = path.joinpath("weights")
        output_path.mkdir(parents=True, exist_ok=True)

        state_dict = {
            "model": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "i_step": i_step
        }

        torch.save(state_dict, output_path.joinpath(f"{identifier}.pth"))

    def load_model(self, path: Path, local_rank: int):
        state_dict = torch.load(path, map_location=torch.device(f"cuda:{local_rank}"))

        self.model.load_state_dict(state_dict["model"])

        if "optimiser" in state_dict.keys() and hasattr(self, "optimiser"):
            self.optimiser.load_state_dict(state_dict["optimiser"])
        if "scheduler" in state_dict.keys() and hasattr(self, "lr_scheduler"):
            self.lr_scheduler.load_state_dict(state_dict["scheduler"])

    @staticmethod
    def map_arguments(batch, mapper):
        output_dict = dict()

        for b_key in batch.keys():
            if isinstance(batch[b_key], torch.Tensor):
                if mapper[b_key] is not None:
                    output_dict[mapper[b_key]] = batch[b_key]
            elif isinstance(batch[b_key], dict):
                rec_dict = BaseManager.map_arguments(batch[b_key], mapper[b_key])
                for rec_key in rec_dict.keys():
                    output_dict[rec_key] = rec_dict[rec_key]
            else:
                raise ValueError(f"Unknown type: {type(batch[b_key])}")

        return output_dict

    @staticmethod
    def move_to_device(arguments, device):
        for key, entry in arguments.items():
            arguments[key] = entry.to(device)

        return arguments

    class AverageMeter:
        # Taken from https://github.com/pytorch/examples

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
