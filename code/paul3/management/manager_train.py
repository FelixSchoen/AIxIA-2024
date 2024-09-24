import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn, GradScaler
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from paul3.data.base_dataset import BaseDataset
from paul3.main.distributed import is_main_node
from paul3.management.manager_base import BaseManager
from paul3.network.learning_rates import get_learning_rate_scheduler, TransformerSchedule
from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings
from paul3.utils.utility import get_paths_files, get_time_hours_minutes_seconds

SETTINGS = Settings()
LOGGER = get_logger(__name__)


class TrainManager(BaseManager):

    def __init__(self, model: nn.Module,
                 model_general_settings: dict,
                 model_hyperparameters: dict,
                 model_settings: dict,
                 output_path: Path,
                 num_workers: int,
                 device: torch.device,
                 optimiser_params: dict = None,
                 lr_params: dict = None,
                 metrics_writer=None,
                 flag_store: bool = False,
                 flag_validation: bool = False,
                 flag_distributed: bool = False):
        super().__init__(model, model_general_settings, model_hyperparameters, model_settings, {}, device)

        self.output_path = output_path
        self.num_workers = num_workers
        self.optimiser_params = optimiser_params
        self.lr_params = lr_params
        self.metrics_writer = metrics_writer
        self.flag_store = flag_store
        self.flag_validation = flag_validation
        self.flag_distributed = flag_distributed

        self.d_model = self.model_hyperparameters["d_model"]
        self.batch_size = self.model_settings["batch_size"]
        self.accumulation_steps = self.model_settings["accumulation_steps"]
        self.checkpoints_per_epoch = self.model_settings["checkpoints_per_epoch"]
        self.max_norm = self.model_settings["clip_max_norm"]

        self.criterion = CrossEntropyLoss(ignore_index=0,
                                          label_smoothing=self.model_settings["label_smoothing_epsilon"])
        self.optimiser = None
        self.scheduler = None
        self.scaler = GradScaler()

    def train(self,
              n_epochs: int,
              dataset_train: BaseDataset,
              dataset_val: BaseDataset,
              argument_mapper: dict,
              local_dataset_train_len: int,
              local_dataset_val_len: int,
              prefetch_factor: int = 2,
              start_epoch: int = 0,
              model_weights_path: Path = None):
        # ---------------------
        # ----- Essential -----
        # ---------------------

        self.criterion = CrossEntropyLoss(ignore_index=0)
        self.optimiser = AdamW(self.model.parameters(), lr=1, betas=self.optimiser_params["betas"],
                               eps=self.optimiser_params["eps"], weight_decay=self.optimiser_params["weight_decay"])

        # Load model if continue training
        if model_weights_path is not None:
            self.load_model(model_weights_path, self.local_rank)

        schedule = TransformerSchedule(d_model=self.d_model, warmup_steps=self.lr_params["warmup_steps"],
                                       factor=self.lr_params["factor"], world_size=self.world_size)
        self.scheduler = get_learning_rate_scheduler(self.optimiser, schedule, start_epoch - 1)

        # Setup distributed environment
        if self.flag_distributed:
            LOGGER.info("Enabling distributed training...")
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])
            torch.cuda.set_device(self.device)

        # Setup dataloaders
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=self.batch_size,
                                      collate_fn=dataset_train.collate_fn,
                                      num_workers=self.num_workers,
                                      prefetch_factor=prefetch_factor,
                                      drop_last=True)
        dataloader_val = DataLoader(dataset_val,
                                    batch_size=self.batch_size,
                                    collate_fn=dataset_val.collate_fn,
                                    num_workers=self.num_workers,
                                    prefetch_factor=prefetch_factor,
                                    drop_last=True)

        # ----------------
        # ----- Meta -----
        # ----------------

        # Calculate dataset lengths
        dataset_train_len = local_dataset_train_len
        dataset_val_len = local_dataset_val_len
        if self.flag_distributed:
            min_dataset_train_len = torch.tensor([dataset_train_len], device=self.device)
            dist.all_reduce(min_dataset_train_len, op=dist.ReduceOp.MAX)
            dataset_train_len = min_dataset_train_len.item()

            min_dataset_val_len = torch.tensor([dataset_val_len], device=self.device)
            dist.all_reduce(min_dataset_val_len, op=dist.ReduceOp.MAX)
            dataset_val_len = min_dataset_val_len.item()
        dataset_train_batches = ((dataset_train_len // self.batch_size) // self.num_workers) * self.num_workers
        dataset_val_batches = ((dataset_val_len // self.batch_size) // self.num_workers) * self.num_workers
        dataset_train.max_entries = min(dataset_train.max_entries,
                                        dataset_train_len) if dataset_train.max_entries != -1 else dataset_train_len
        dataset_val.max_entries = min(dataset_val.max_entries,
                                      dataset_val_len) if dataset_val.max_entries != -1 else dataset_val_len

        # Calculate checkpoint indices
        checkpoint_indices = []
        n_steps_per_epoch = dataset_train_batches // self.accumulation_steps
        step_size = math.ceil(n_steps_per_epoch / self.checkpoints_per_epoch)
        for i in range(self.checkpoints_per_epoch):
            checkpoint_indices.append(min(step_size * (i + 1), dataset_train_batches))
        checkpoint_indices.pop(-1)

        # Setup metrics
        metrics_step_loss = TrainManager.AverageMeter()
        metrics_step_accuracy = TrainManager.AverageMeter()
        metrics_smooth_loss = TrainManager.AverageMeter()
        metrics_smooth_accuracy = TrainManager.AverageMeter()
        metrics_epoch_loss = TrainManager.AverageMeter()
        metrics_epoch_accuracy = TrainManager.AverageMeter()
        metrics_val_loss = TrainManager.AverageMeter()
        metrics_val_accuracy = TrainManager.AverageMeter()
        metrics_time_step = TrainManager.AverageMeter()
        metrics_time_data = TrainManager.AverageMeter()
        metrics_step = 0
        metrics = {"step": metrics_step,
                   "step_loss": metrics_step_loss, "step_accuracy": metrics_step_accuracy,
                   "smooth_loss": metrics_smooth_loss, "smooth_accuracy": metrics_smooth_accuracy,
                   "epoch_loss": metrics_epoch_loss, "epoch_accuracy": metrics_epoch_accuracy,
                   "val_loss": metrics_val_loss, "val_accuracy": metrics_val_accuracy,
                   "time_step": metrics_time_step, "time_data": metrics_time_data}

        # Information
        if is_main_node():
            LOGGER.info("Model information:")
            LOGGER.info(f"{'-' * 50}")
            LOGGER.info(f"Start Epoch: {start_epoch}")
            LOGGER.info(f"Epochs: {n_epochs}")
            LOGGER.info(f"Tokeniser: {self.model_general_settings['tokeniser']}")
            LOGGER.info(f"Train Entries: {dataset_train_len}")
            LOGGER.info(f"Train Batches: {dataset_train_batches}")
            LOGGER.info(f"Val Entries: {dataset_val_len}")
            LOGGER.info(f"Val Batches: {dataset_val_batches}")
            LOGGER.info(f"Batch Size: {self.batch_size}")
            LOGGER.info(f"Accumulation Steps: {self.accumulation_steps}")
            LOGGER.info(f"Checkpoint Indices: {checkpoint_indices + [n_steps_per_epoch]}")
            LOGGER.info(f"{'-' * 50}")
            LOGGER.info(f"{self.model_hyperparameters}")
            LOGGER.info(f"{self.model_settings}")
            LOGGER.info(f"{'-' * 50}")

        # Timers
        time_step = time.time()
        time_epoch = time.time()

        # --------------------
        # ----- Training -----
        # --------------------

        LOGGER.info("Starting training...")
        self.model.train()

        for i_epoch in range(n_epochs):
            for i_batch, batch in enumerate(dataloader_train):
                # ----- Model Pass -----

                loss, accuracy = self.model_pass(batch, argument_mapper)

                # Update metrics
                metrics["step_loss"].update(loss.item())
                metrics["smooth_loss"].update(loss.item())
                metrics["epoch_loss"].update(loss.item())
                metrics["step_accuracy"].update(accuracy.item())
                metrics["smooth_accuracy"].update(accuracy.item())
                metrics["epoch_accuracy"].update(accuracy.item())

                # ----- Accumulation -----

                if (i_batch + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimiser)
                    clip_grad_norm_(self.model.parameters(), self.max_norm)
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                    self.optimiser.zero_grad()
                    self.scheduler.step()

                    # Update metrics
                    metrics["step"] = metrics["step"] + 1
                    metrics["time_step"].update(time.time() - time_step)
                    i_step = metrics["step"]

                    # Metrics writer
                    if self.metrics_writer is not None and is_main_node():
                        self.metrics_writer.add_scalar("step_loss", metrics["step_loss"].avg, i_step)
                        self.metrics_writer.add_scalar("step_accuracy", metrics["step_accuracy"].avg, i_step)
                        self.metrics_writer.add_scalar("smooth_loss", metrics["step_loss"].avg, i_step)
                        self.metrics_writer.add_scalar("smooth_accuracy", metrics["step_accuracy"].avg, i_step)
                        self.metrics_writer.add_scalar("epoch_loss", metrics["epoch_loss"].avg, i_step)
                        self.metrics_writer.add_scalar("epoch_accuracy", metrics["epoch_accuracy"].avg, i_step)
                        self.metrics_writer.add_scalar("lr", self.scheduler.get_last_lr()[0], i_step)

                    # Logging
                    if i_step % SETTINGS.TRAIN_LOG_FREQUENCY == 0 or i_step == 0:
                        self._log("| "
                                  f"Loss: {metrics['smooth_loss'].avg:07.4f} | "
                                  f"Accuracy: {metrics['smooth_accuracy'].avg:07.4f} | "
                                  f"LR: {self.optimiser.param_groups[0]['lr']:010.8f} | "
                                  f"Time Step: {metrics['time_step'].sum:05.2f} | "
                                  f"GPU Memory: {(torch.cuda.memory_reserved(self.device) / (1024 ** 3)):05.2f} / "
                                  f"{(self.device_properties.total_memory / (1024 ** 3)):05.2f} GB",
                                  i_epoch, i_batch, i_step)
                        metrics["smooth_loss"].reset()
                        metrics["smooth_accuracy"].reset()

                    # Reset metrics
                    metrics["step_loss"].reset()
                    metrics["step_accuracy"].reset()
                    metrics["time_step"].reset()
                    metrics["time_data"].reset()
                    time_step = time.time()

                    # Checkpointing
                    if any([i_step % n_steps_per_epoch == checkpoint_index for checkpoint_index in checkpoint_indices]):
                        self.checkpoint(metrics, dataloader_val, argument_mapper, i_epoch, i_batch, i_step,
                                        checkpoint_indices.index(i_step % n_steps_per_epoch))

            self.checkpoint(metrics, dataloader_val, argument_mapper, i_epoch, i_batch, i_step,
                            len(checkpoint_indices))

            hours, minutes, seconds = get_time_hours_minutes_seconds(time.time() - time_epoch)
            self._log("| "
                      f"Finished epoch {i_epoch + 1} in {hours:02}:{minutes:02}:{seconds:02} | "
                      f"Loss: {metrics['epoch_loss'].avg:07.4f} |"
                      f"Accuracy: {metrics['epoch_accuracy'].avg:07.4f}", i_epoch, i_batch, i_step)

            # Reset metrics
            metrics["epoch_loss"].reset()
            metrics["epoch_accuracy"].reset()
            time_epoch = time.time()

    def model_pass(self, batch, argument_mapper):
        arguments_dict = BaseManager.map_arguments(batch, argument_mapper)
        model_input = BaseManager.move_to_device(arguments_dict, self.device)
        labels = model_input["state"][:, 1:]
        model_input["state"] = model_input["state"][:, :-1]

        # Forward pass
        logits = self.model(**model_input)
        logits_t = logits.permute(0, 2, 1)
        loss = self.criterion(logits_t, labels)

        # Backward pass
        if torch.is_grad_enabled():
            self.scaler.scale(loss / self.accumulation_steps).backward()

        # Accuracy
        with torch.no_grad():
            mask = labels != 0

            predictions = logits.argmax(dim=-1)
            predictions_masked = predictions[mask]
            labels_masked = labels[mask]

            correct_predictions = torch.eq(predictions_masked, labels_masked)
            accuracy = correct_predictions.float().mean()

        return loss, accuracy

    def checkpoint(self, metrics, dataloader_val, argument_mapper, i_epoch, i_batch, i_step, i_checkpoint):
        self._log("| "
                  f"Checkpoint {i_checkpoint + 1}", i_epoch, i_batch, i_step)

        # Store model and remove old checkpoints
        if self.flag_store and is_main_node():
            self.save_model(self.output_path, i_step, self.optimiser, self.scheduler,
                            f"e{i_epoch + 1:02d}_c{i_checkpoint + 1:02d}")
            if SETTINGS.TRAIN_CHECKPOINTS_TO_KEEP > 0:
                checkpoint_paths = sorted(get_paths_files(self.output_path, ["pth"]))
                for checkpoint_path in checkpoint_paths[:-SETTINGS.TRAIN_CHECKPOINTS_TO_KEEP]:
                    checkpoint_path.unlink()

        if self.flag_distributed:
            dist.barrier()

        if self.flag_validation:
            self.validate(metrics, dataloader_val, argument_mapper, i_epoch, i_step)

    def validate(self, metrics, dataloader_val, argument_mapper, i_epoch, i_step):
        self.model.eval()

        with torch.no_grad():
            for i_batch, val_batch in enumerate(dataloader_val):
                v_loss, v_accuracy = self.model_pass(val_batch, argument_mapper)

            # Update metrics
            metrics["val_loss"].update(v_loss.item())
            metrics["val_accuracy"].update(v_accuracy.item())

            if self.metrics_writer is not None and is_main_node():
                self.metrics_writer.add_scalar("val_loss", metrics["val_loss"].avg, i_step)
                self.metrics_writer.add_scalar("val_accuracy", metrics["val_accuracy"].avg, i_step)

            self._log("| "
                      f"Validation Loss: {metrics['val_loss'].avg:07.4f} | "
                      f"Validation Accuracy: {metrics['val_accuracy'].avg:07.4f}", i_epoch, i_batch, i_step)

            # Reset metrics
            metrics["val_loss"].reset()
            metrics["val_accuracy"].reset()

        self.model.train()

    @staticmethod
    def _log(message, i_epoch, i_batch, i_step):
        if is_main_node():
            LOGGER.info(f"[E{i_epoch + 1:02d}|B{i_batch + 1:05d}|S{i_step:08d}] {message}")

    @staticmethod
    def _count_overall_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
