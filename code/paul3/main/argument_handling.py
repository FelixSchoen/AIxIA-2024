import argparse

from scoda.enumerations.tokeniser_type import TokeniserType

from paul3.commands.command_data import handle_command_data_music, handle_command_data_text
from paul3.commands.command_infer import handle_command_infer_music
from paul3.commands.command_metrics import handle_command_metrics
from paul3.enumerations.info_type import InfoType
from paul3.utils.settings import Settings
from paul3.utils.utility import str_to_bool

settings = Settings()


def _parse_arguments():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")

    # Data
    parser_data = subparser.add_parser("data")
    parser_data_subparser = parser_data.add_subparsers(dest="data_type")

    # - Music
    parser_data_music = parser_data_subparser.add_parser("music")
    parser_data_music.add_argument("--dataset-identifier", required=True)

    parser_data_music.add_argument("--tokeniser-identifier",
                                   help="Which tokeniser to use")
    parser_data_music.add_argument("--info-flags", nargs="+", type=str,
                                   help="Which additional information to store")

    parser_data_music.add_argument("--augment-transpose", type=str_to_bool, default=True,
                                   help="Whether to use transpose augmentation")

    parser_data_music.add_argument("--skip-sanitisation", action="store_true", help="Skips the sanitisation procedure")
    parser_data_music.add_argument("--skip-processing", action="store_true", help="Skips the processing procedure")
    parser_data_music.add_argument("--skip-store", action="store_true", help="Skips the storing procedure")
    parser_data_music.add_argument("--skip-difficulty", action="store_true", help="Skips the difficulty calculation")

    parser_data_music.add_argument("--set-split-train", type=int, help="Sets the split of the train dataset")
    parser_data_music.add_argument("--set-split-val", type=int, help="Sets the split of the validation dataset")

    # - Text
    parser_data_text = parser_data_subparser.add_parser("text")
    parser_data_text.add_argument("--dataset-identifier", required=True)

    # Train
    parser_train = subparser.add_parser("train")
    parser_train.add_argument("--model-config-identifier", required=True, help="The model config to use")
    parser_train.add_argument("--model-instance-identifier", required=False,
                              help="The name of the directory to load weights to continue training from")
    parser_train.add_argument("--step-identifier", required=False,
                              help="The name of the set of weights to load")
    parser_train.add_argument("--num-workers", type=int, default=8, required=False,
                              help="The number of workers for data loading")
    parser_train.add_argument("--entries-limit-train", type=int, default=-1, required=False,
                              help="The number of entries to use for training")
    parser_train.add_argument("--entries-limit-val", type=int, default=-1, required=False,
                              help="The number of entries to use for validation")
    parser_train.add_argument("--skip-store", action="store_true", required=False, help="Skips the storing procedure")
    parser_train.add_argument("--skip-validation", action="store_true", required=False,
                              help="Skips the validation procedure")
    parser_train.add_argument("--skip-distributed", action="store_true", required=False,
                              help="Disables distributed context loading")

    # Infer
    parser_infer = subparser.add_parser("infer")
    parser_infer_subparser = parser_infer.add_subparsers(dest="data_type")

    # - Music
    parser_infer_music = parser_infer_subparser.add_parser("music")
    parser_infer_music.add_argument("--model-instance-identifier", required=True,
                                    help="The name of the directory to load weights from")
    parser_infer_music.add_argument("--step-identifier", type=str, required=True)
    parser_infer_music.add_argument("--iterations", type=int, default=1, required=False,
                                    help="Number of iterations to run inference for")
    parser_infer_music.add_argument("--input-identifier", type=str, required=False,
                                    help="Number of iterations to run inference for")
    parser_infer_music.add_argument("--info-flags", nargs="*", type=str,
                                    help="Additional information flags")

    # Metrics
    parser_metrics = subparser.add_parser("metrics")
    parser_metrics.add_argument("--model-instance-identifier", required=True)
    parser_metrics.add_argument("--step-identifier", type=str, required=True)
    parser_metrics.add_argument("--num-workers", type=int, default=8, required=False)
    parser_metrics.add_argument("--entries-limit", type=int, default=-1, required=False)

    return parser.parse_args()


def _handle_arguments(args):
    if args.command == "data":
        if args.data_type == "music":
            dataset_identifier = args.dataset_identifier
            tokeniser_identifier = args.tokeniser_identifier

            info_flags = []
            if args.info_flags is not None:
                for info_flag in args.info_flags:
                    info_flags.append(InfoType[info_flag.upper()])

            # Settings
            if args.set_split_train is not None:
                settings.DATA_SPLIT_TRAIN = args.set_split_train
            if args.set_split_val is not None:
                settings.DATA_SPLIT_VAL = args.set_split_val

            handle_command_data_music(dataset_identifier=dataset_identifier,
                                      tokeniser_identifier=tokeniser_identifier,
                                      info_flags=info_flags,
                                      skip_sanitisation=args.skip_sanitisation,
                                      skip_processing=args.skip_processing,
                                      skip_store=args.skip_store,
                                      augment_transpose=args.augment_transpose)
        elif args.data_type == "text":
            dataset_identifier = args.dataset_identifier

            handle_command_data_text(dataset_identifier)
    elif args.command == "train":
        model_config_identifier = args.model_config_identifier
        model_instance_identifier = args.model_instance_identifier
        step_identifier = args.step_identifier
        num_workers = args.num_workers
        entries_limit_train = args.entries_limit_train
        entries_limit_val = args.entries_limit_val
        skip_store = args.skip_store
        skip_validation = args.skip_validation
        skip_distributed = args.skip_distributed
        from paul3.commands.command_train import handle_command_train
        handle_command_train(model_config_identifier, model_instance_identifier, step_identifier, num_workers,
                             entries_limit_train, entries_limit_val,
                             skip_store, skip_validation, skip_distributed)
    elif args.command == "infer":
        model_instance_identifier = args.model_instance_identifier
        step_identifier = args.step_identifier
        iterations = args.iterations
        input_identifier = args.input_identifier
        if args.data_type == "music":
            handle_command_infer_music(model_instance_identifier, step_identifier, iterations, input_identifier)
        else:
            raise NotImplementedError()
    elif args.command == "metrics":
        model_instance_identifier = args.model_instance_identifier
        step_identifier = args.step_identifier
        num_workers = args.num_workers
        entries_limit = args.entries_limit
        handle_command_metrics(model_instance_identifier, step_identifier, num_workers, entries_limit)
    else:
        raise ValueError("Unknown command")
