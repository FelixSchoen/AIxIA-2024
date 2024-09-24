import json
from pathlib import Path

import torch

from paul3.data.music.music_dataset import MusicDataset
from paul3.data.text.text_dataset import TextDataset

from paul3.enumerations.model_type import ModelType
from paul3.exceptions.data_exception import DataException
from paul3.exceptions.model_exception import ModelException
from paul3.main.distributed import setup_distributed, get_environment_variables, is_main_node
from paul3.models.pytorch_hacked_single_stream_transformer import PytorchHackedSingleStreamTransformer
from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings

settings = Settings()
LOGGER = get_logger(__name__)


def shared_setup_environment(skip_distributed: bool = False):
    LOGGER.info("Setting up environment...")
    if not skip_distributed:
        setup_distributed()
    global_rank, local_rank, world_size = get_environment_variables()
    LOGGER.info(
        f"[Main Node: {is_main_node()}] Global Rank: {global_rank} Local Rank: {local_rank} World size: {world_size}")

    return global_rank, local_rank, world_size


def shared_setup_device(local_rank: int):
    LOGGER.info("Loading CUDA context...")
    if torch.cuda.is_available():
        LOGGER.info("CUDA device available")
        device = torch.device(f"cuda:{local_rank}")
    else:
        LOGGER.info("No CUDA device found")
        device = torch.device("cpu")

    return device


def shared_load_model(model_identifier, model_hyperparameters, device):
    if model_identifier == "todo":
        raise ModelException(f"Invalid model: {model_identifier}")
    elif model_identifier == ModelType.PYTORCH_HACKED_SINGLE_STREAM_TRANSFORMER.value:
        model = PytorchHackedSingleStreamTransformer(**model_hyperparameters, device=device)
    else:
        raise ModelException(f"Invalid model: {model_identifier}")

    return model


def shared_load_model_config(model_config_identifier):
    model_config_file_path = Path(__file__).parent.parent.parent.joinpath(
        f"cfg/model_configs/{model_config_identifier}.json")
    with open(model_config_file_path) as model_config_file:
        model_config_json = json.load(model_config_file)

    return model_config_json


def shared_load_model_info(model_instance_identifier):
    model_info_file_path = Path(settings.MODELS_BASE_PATH).joinpath(
        model_instance_identifier, "info.json")
    with open(model_info_file_path) as model_info_file:
        model_info_json = json.load(model_info_file)

    return model_info_json


def shared_get_model_weights_path(model_instance_identifier, weight_identifier):
    model_weights_path = Path(settings.MODELS_BASE_PATH).joinpath(
        model_instance_identifier, "weights", f"{weight_identifier}.pth")
    return model_weights_path


def shared_configure_argument_mapper(dataset_type, dataset_class):
    if dataset_type == "music" and dataset_class == "single":
        argument_mapper = {"track_00": {"sequence": "state"}}
    elif dataset_type == "music" and dataset_class == "dual":
        argument_mapper = {"source": "context", "target": "state"}
    elif dataset_type == "text" and dataset_class == "dual":
        argument_mapper = {"source": "context", "target": "state"}
    elif dataset_type == "text" and dataset_class == "single":
        argument_mapper = {"sentence": "state"}
    else:
        raise DataException("Invalid dataset")

    return argument_mapper


def shared_load_music_datasets(dataset_identifier: str, tokeniser_identifier: str, model_hyperparameters: dict,
                               entries_limit_train: int, entries_limit_val: int, invalid_tokens: list):
    max_len = model_hyperparameters["max_len"]
    block_size = 1
    if "block_size" in model_hyperparameters:
        block_size = model_hyperparameters["block_size"]

    dataset_base_path = Path(settings.DATA_SETS_BASE_PATH)
    dataset_path = dataset_base_path.joinpath(settings.DATA_SETS[dataset_identifier]["identifier"], "database",
                                              tokeniser_identifier)

    dataset_train_path = dataset_path.joinpath("train.tar")
    dataset_train = MusicDataset(dataset_train_path, resampled=True, max_len=max_len, max_entries=entries_limit_train,
                                 compatible_divisor=block_size, invalid_tokens=invalid_tokens)

    dataset_val_path = dataset_path.joinpath("val.tar")
    dataset_val = MusicDataset(dataset_val_path, resampled=True, max_len=max_len, max_entries=entries_limit_val,
                               compatible_divisor=block_size, invalid_tokens=invalid_tokens)

    return dataset_train, dataset_val


def shared_load_text_datasets(dataset_identifier: str, model_hyperparameters: dict,
                              entries_limit_train: int, entries_limit_val: int):
    max_len = model_hyperparameters["max_len"]
    block_size = 1
    if "block_size" in model_hyperparameters:
        block_size = model_hyperparameters["block_size"]

    dataset_base_path = Path(settings.DATA_SETS_BASE_PATH)
    dataset_path = dataset_base_path.joinpath(settings.DATA_SETS[dataset_identifier]["identifier"], "database")

    dataset_train_path = dataset_path.joinpath("train.tar")
    dataset_train = TextDataset(dataset_train_path, resampled=True, max_len=max_len, max_entries=entries_limit_train,
                                compatible_divisor=block_size)

    dataset_val_path = dataset_path.joinpath("val.tar")
    dataset_val = TextDataset(dataset_val_path, resampled=True, max_len=max_len, max_entries=entries_limit_val,
                              compatible_divisor=block_size)

    return dataset_train, dataset_val
