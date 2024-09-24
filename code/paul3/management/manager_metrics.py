from pathlib import Path

import torch
from torch import nn, GradScaler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torcheval.metrics.text import Perplexity

from paul3.data.base_dataset import BaseDataset
from paul3.management.manager_base import BaseManager
from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings

SETTINGS = Settings()
LOGGER = get_logger(__name__)


class MetricsManager(BaseManager):

    def __init__(self, model: nn.Module, model_hyperparameters: dict, model_settings: dict,
                 num_workers: int, device: torch.device):
        super().__init__(model, model_hyperparameters, model_settings, {}, device)

        self.num_workers = num_workers

        self.d_model = self.model_hyperparameters["d_model"]
        self.batch_size = self.model_settings["batch_size"]
        self.accumulation_steps = self.model_settings["accumulation_steps"]
        self.checkpoints_per_epoch = self.model_settings["checkpoints_per_epoch"]
        self.max_norm = self.model_settings["clip_max_norm"]

        self.criterion = CrossEntropyLoss(ignore_index=0,
                                          label_smoothing=self.model_settings["label_smoothing_epsilon"])

    def metrics(self,
                dataset: BaseDataset,
                argument_mapper: dict):
        metric_perplexity = Perplexity(ignore_index=0, device=self.device)

        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                collate_fn=dataset.collate_fn,
                                num_workers=self.num_workers,
                                drop_last=True)

        with torch.no_grad():
            for i_batch, batch in enumerate(dataloader):
                logits, loss = self.model_pass(batch, argument_mapper)
                print(loss)

                arguments_dict = BaseManager.map_arguments(batch, argument_mapper)
                model_input = BaseManager.move_to_device(arguments_dict, self.device)
                model_input = model_input["state"][:, :-1]
                metric_perplexity.update(logits, model_input)
            perplexity = metric_perplexity.compute()
            print(f"Perplexity: {perplexity}")

    def model_pass(self, batch, argument_mapper):
        arguments_dict = BaseManager.map_arguments(batch, argument_mapper)
        model_input = BaseManager.move_to_device(arguments_dict, self.device)
        labels = model_input["state"][:, 1:]
        model_input["state"] = model_input["state"][:, :-1]

        # Forward pass
        logits = self.model(**model_input)
        logits_t = logits.permute(0, 2, 1)
        loss = self.criterion(logits_t, labels)

        return logits, loss
