import datetime
import os

import torch
import torch.distributed as dist

from paul3.utils import paul_logging
from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings

SETTINGS = Settings()
LOGGER = get_logger(__name__)


def is_main_node():
    rank = int(os.environ.get("RANK", -1))
    return True if rank == 0 or rank == -1 else False


def get_environment_variables():
    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    return global_rank, local_rank, world_size


def setup_distributed():
    if os.environ.get("WORLD_SIZE", None) is None:
        LOGGER.info("Using default environment variables")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29656")

    if torch.distributed.is_nccl_available():
        LOGGER.info("Using NCCL")
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=SETTINGS.TRAIN_TIMEOUT))
    else:
        LOGGER.info("Using Gloo")
        dist.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=SETTINGS.TRAIN_TIMEOUT))

    paul_logging.setup()


def cleanup_distributed():
    LOGGER.info("Cleaning up distributed environment...")
    dist.destroy_process_group()
