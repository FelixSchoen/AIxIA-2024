import json
from datetime import datetime
from pathlib import Path

from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter

from paul3.commands.command_shared import shared_load_model, shared_load_model_config, shared_load_model_info, \
    shared_get_model_weights_path, shared_setup_device, shared_setup_environment, shared_configure_argument_mapper, \
    shared_load_music_datasets, shared_load_text_datasets
from paul3.exceptions.data_exception import DataException
from paul3.main.distributed import cleanup_distributed, is_main_node
from paul3.management.manager_train import TrainManager
from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings

settings = Settings()
LOGGER = get_logger(__name__)


@record
def handle_command_train(model_config_identifier, model_instance_identifier, step_identifier, num_workers,
                         entries_limit_train, entries_limit_val, skip_store=False, skip_validation=False,
                         skip_distributed=True):
    # Setup environment
    global_rank, local_rank, world_size = shared_setup_environment(skip_distributed)

    try:
        # Setup device
        device = shared_setup_device(local_rank)

        # Load base path
        models_base_path = Path(settings.MODELS_BASE_PATH)

        # Load config
        LOGGER.info("Loading config values...")
        if model_instance_identifier is not None:
            model_config_json = shared_load_model_info(model_instance_identifier)
        else:
            model_config_json = shared_load_model_config(model_config_identifier)

        # Load settings and strings
        model_general_settings = model_config_json["general"]
        model_hyperparameters = model_config_json["hyperparameters"]
        model_training_settings = model_config_json["training"]
        model_inference_settings = model_config_json["inference"]
        model_identifier = model_general_settings["model"]
        dataset_identifier = model_general_settings["dataset"]
        tokeniser_identifier = model_general_settings["tokeniser"]
        invalid_tokens = model_training_settings["invalid_tokens"]
        time_string = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Setup output path
        if model_instance_identifier is not None:
            output_path = models_base_path.joinpath(model_instance_identifier)
        else:
            output_path = models_base_path.joinpath(f"{time_string}-{model_identifier}-{dataset_identifier}")
            if not skip_store:
                output_path.mkdir(parents=True, exist_ok=True)

        # Set model
        LOGGER.info("Loading model...")
        model = shared_load_model(model_identifier, model_hyperparameters, device)

        # Set argument mapper and label identifier
        dataset_type = settings.DATA_SETS[dataset_identifier]["dataset_type"]
        dataset_class = settings.DATA_SETS[dataset_identifier]["dataset_class"]
        argument_mapper = shared_configure_argument_mapper(dataset_type, dataset_class)

        # Load datasets
        LOGGER.info("Loading datasets...")
        if dataset_type == "music":
            dataset_train, dataset_val = shared_load_music_datasets(dataset_identifier,
                                                                    tokeniser_identifier,
                                                                    model_hyperparameters,
                                                                    entries_limit_train,
                                                                    entries_limit_val,
                                                                    invalid_tokens)
        elif dataset_type == "text":
            dataset_train, dataset_val = shared_load_text_datasets(dataset_identifier,
                                                                   model_hyperparameters,
                                                                   entries_limit_train,
                                                                   entries_limit_val)
        else:
            raise DataException("Invalid dataset type")

        # Count dataset length
        LOGGER.info("Counting lengths of datasets...")
        dataset_lengths = []
        for dataset in [dataset_train, dataset_val]:
            resampled = dataset.resampled
            dataset.resampled = False
            dataset_len = 0
            for _ in dataset:
                dataset_len += 1
            dataset.resampled = resampled
            dataset_lengths.append(dataset_len)
        dataset_train_len = dataset_lengths[0]
        dataset_val_len = dataset_lengths[1]

        # Setup output path
        if not skip_store and is_main_node():
            if model_instance_identifier is None:
                # Setup info dict
                file_dict = dict()
                info_dict = dict()
                file_dict["general"] = model_general_settings
                file_dict["info"] = info_dict
                info_dict["time"] = time_string
                info_dict["model_config_identifier"] = model_config_identifier
                info_dict["dataset_identifier"] = dataset_identifier
                info_dict["tokeniser_identifier"] = tokeniser_identifier
                info_dict["world_size"] = world_size
                info_dict["dataset_train_entries"] = dataset_train_len
                info_dict["dataset_val_entries"] = dataset_train_len
                file_dict["hyperparameters"] = model_hyperparameters
                file_dict["training"] = model_training_settings
                file_dict["inference"] = model_inference_settings

                # Write info dict to file
                info_path = output_path.joinpath("info.json")
                with open(info_path, "w") as file:
                    json.dump(file_dict, file)

            # Setup tensorboard writer
            logs_path = output_path.joinpath("logs")
            logs_path.mkdir(exist_ok=True)
            tensorboard_writer = SummaryWriter(str(logs_path))
        else:
            output_path = None
            tensorboard_writer = None

        # Training process
        train_manager = TrainManager(model, model_general_settings, model_hyperparameters, model_training_settings,
                                     output_path, num_workers,
                                     device, model_training_settings["optimiser_params"],
                                     model_training_settings["lr_params"], metrics_writer=tensorboard_writer,
                                     flag_store=not skip_store, flag_validation=not skip_validation,
                                     flag_distributed=not skip_distributed)

        # Load model weights
        model_weights_path = None
        start_epoch = 0
        if model_instance_identifier is not None:
            LOGGER.info("Loading model weights...")
            model_weights_path = shared_get_model_weights_path(model_instance_identifier,
                                                               step_identifier)
            start_epoch = int(step_identifier[1:3])

        LOGGER.info("Starting training process...")
        train_manager.train(n_epochs=model_training_settings["epochs"],
                            dataset_train=dataset_train,
                            dataset_val=dataset_val,
                            argument_mapper=argument_mapper,
                            local_dataset_train_len=dataset_train_len,
                            local_dataset_val_len=dataset_val_len,
                            prefetch_factor=settings.DATA_BATCH_BUFFER_SIZE * settings.DATA_PREFETCH_FACTOR,
                            start_epoch=start_epoch,
                            model_weights_path=model_weights_path)

        if not skip_store and is_main_node():
            tensorboard_writer.close()

        LOGGER.info("Training process complete")
    except Exception as e:
        LOGGER.error(f"Exception occurred: {e}")
        raise e
    finally:
        if not skip_distributed:
            cleanup_distributed()

    LOGGER.info("Program stopped")
