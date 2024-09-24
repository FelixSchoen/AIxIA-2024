from paul3.commands.command_shared import shared_load_model, shared_load_model_info, \
    shared_setup_device, shared_setup_environment, shared_get_model_weights_path, shared_configure_argument_mapper, \
    shared_load_music_datasets, shared_load_text_datasets
from paul3.exceptions.data_exception import DataException
from paul3.management.manager_metrics import MetricsManager
from paul3.utils.inference_constrainer import MinimumLengthConstrainer
from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings

settings = Settings()
LOGGER = get_logger(__name__)


def handle_command_metrics(model_instance_identifier, step_identifier, num_workers, entries_limit):
    # Setup environment
    global_rank, local_rank, world_size = shared_setup_environment()

    # Setup device
    device = shared_setup_device(local_rank)

    # Load model info and configuration
    LOGGER.info("Loading model information...")
    model_info_json = shared_load_model_info(model_instance_identifier)
    model_general_settings = model_info_json["general"]
    model_hyperparameters = model_info_json["hyperparameters"]
    model_training_settings = model_info_json["training"]
    model_inference_settings = model_info_json["inference"]
    model_identifier = model_general_settings["model"]
    dataset_identifier = model_general_settings["dataset"]
    tokeniser_identifier = model_general_settings["tokeniser"]
    invalid_tokens = model_training_settings["invalid_tokens"]
    constrainer_settings = model_inference_settings["constrainer"]

    # Load model
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
                                                                entries_limit,
                                                                entries_limit,
                                                                invalid_tokens)
        dataset_train.resampled = False
        dataset_val.resampled = False
    else:
        raise DataException("Invalid dataset type")

    # Load constrainer
    constrainer = MinimumLengthConstrainer(vocab_size=model_hyperparameters["vocab_size"]["target"][0],
                                           **constrainer_settings)

    # Setup manager
    LOGGER.info("Setting up manager...")
    metrics_manager = MetricsManager(model, model_hyperparameters, model_training_settings, num_workers, device)

    # Load model weights
    LOGGER.info("Loading model weights...")
    model_weights_path = shared_get_model_weights_path(model_instance_identifier, step_identifier)
    metrics_manager.load_model(model_weights_path, local_rank)

    # Metrics
    LOGGER.info("Computing metrics...")
    metrics_manager.metrics(dataset_train, argument_mapper)
