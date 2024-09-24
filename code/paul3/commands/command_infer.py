import json
from datetime import datetime
from pathlib import Path

import torch
from scoda.enumerations.tokeniser_type import TokeniserType

from paul3.commands.command_shared import shared_load_model, shared_load_model_info, shared_get_model_weights_path, \
    shared_setup_device, shared_setup_environment
from paul3.management.manager_inference import InferenceManager
from paul3.utils.inference_constrainer import MinimumLengthConstrainer, MinimumBarsConstrainer
from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings
from paul3.utils.utility import get_music_tokeniser

settings = Settings()
LOGGER = get_logger(__name__)


def handle_command_infer_music(model_instance_identifier, step_identifier, iterations, input_identifier=None):
    # Setup environment
    global_rank, local_rank, world_size = shared_setup_environment()

    # Setup device
    device = shared_setup_device(local_rank)

    # Load model info and configuration
    LOGGER.info("Loading model information...")
    model_info_json = shared_load_model_info(model_instance_identifier)
    model_general_settings = model_info_json["general"]
    model_hyperparameters = model_info_json["hyperparameters"]
    model_inference_settings = model_info_json["inference"]
    model_identifier = model_general_settings["model"]
    tokeniser_identifier = model_general_settings["tokeniser"]
    constrainer_settings = model_inference_settings["constrainer"]
    approach_settings = model_inference_settings["approach"]

    # Load model
    LOGGER.info("Loading model...")
    model = shared_load_model(model_identifier, model_hyperparameters, device)

    # Load tokeniser
    tokeniser = get_music_tokeniser(tokeniser_identifier)

    # Load constrainer
    constrainer_type = "minimum_length_constrainer"
    if constrainer_type == "minimum_length_constrainer":
        constrainer = MinimumLengthConstrainer(vocab_size=model_hyperparameters["vocab_size"]["target"][0],
                                               **constrainer_settings)
    elif constrainer_type == "minimum_bars_constrainer":
        constrainer = MinimumBarsConstrainer(vocab_size=model_hyperparameters["vocab_size"]["target"][0],
                                             tokeniser=tokeniser, **constrainer_settings)

    # Setup manager
    LOGGER.info("Setting up manager...")
    inference_manager = InferenceManager(model, model_general_settings, model_hyperparameters, model_inference_settings,
                                         tokeniser, device,
                                         constrainer=constrainer, approach_settings=approach_settings)

    # Load model weights
    LOGGER.info("Loading model weights...")
    model_weights_path = shared_get_model_weights_path(model_instance_identifier, step_identifier)
    inference_manager.load_model(model_weights_path, local_rank)

    # Setup output path
    time_string = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = Path(settings.MODELS_BASE_PATH).joinpath(model_instance_identifier, "output", time_string)
    output_path.mkdir(parents=True, exist_ok=True)

    # Inference
    LOGGER.info("Running inference...")

    output_dict = dict()

    for i in range(iterations):
        LOGGER.info(f"Iteration {i + 1}...")

        # Load input
        input_dict = inference_load_input(model_instance_identifier, input_identifier, i)

        outputs, scores = inference_manager.inference(input_dict)

        for j, output in enumerate(outputs):
            output_dict[f"output_{i:02d}_{j:02d}"] = output

            sequence = tokeniser.detokenise(output)
            sequence.save(output_path.joinpath(f"output_{i:02d}_{j:02d}.mid"))

    with open(output_path.joinpath(f"output_tokens.json"), "w") as f:
        json.dump(output_dict, f)


def inference_load_input(model_instance_identifier, input_identifier, iteration):
    input_file_path_stem = Path(settings.MODELS_BASE_PATH).joinpath(model_instance_identifier, "input",
                                                                    input_identifier)
    input_file_path_iter = input_file_path_stem.with_name(f"{input_file_path_stem.name}_{iteration:02d}").with_suffix(
        ".json")
    input_file_path_base = input_file_path_stem.with_suffix(".json")

    if input_file_path_iter.exists():
        with open(input_file_path_iter) as input_file:
            input_json = json.load(input_file)
    elif input_file_path_base.exists():
        with open(input_file_path_base) as input_file:
            input_json = json.load(input_file)
    else:
        input_json = {"state": [[1]]}

    tensor_dict = dict()

    _convert_dict_to_tensor(input_json, tensor_dict)

    return tensor_dict


def _convert_dict_to_tensor(input_dict, output_dict):
    for key in input_dict:
        if isinstance(input_dict[key], dict):
            if key not in output_dict:
                output_dict[key] = dict()
            _convert_dict_to_tensor(input_dict[key], output_dict[key])
        elif isinstance(input_dict[key], list):
            output_dict[key] = torch.tensor(input_dict[key])
        else:
            raise ValueError("Unknown input type")
