from pathlib import Path

from scoda.enumerations.tokeniser_type import TokeniserType

from paul3.data.music.music_processing import process_midi_files
from paul3.data.music.music_sanitisation import sanitise_midi_files
from paul3.data.music.music_storage import music_store
from paul3.data.text.text_processing_storage import text_process_store_single
from paul3.enumerations.info_type import InfoType
from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings
from paul3.utils.utility import get_paths_files, split_paths_dataset_files

settings = Settings()
LOGGER = get_logger(__name__)


def handle_command_data_music(dataset_identifier, tokeniser_identifier, info_flags, skip_sanitisation=False,
                              skip_processing=False, skip_store=False, augment_transpose=True):
    # Load dataset settings
    source_type = settings.DATA_SETS[dataset_identifier]["source_class"]
    dataset_type = settings.DATA_SETS[dataset_identifier]["dataset_class"]
    path_dataset_base = Path(settings.DATA_SETS_BASE_PATH)
    path_source_dir = path_dataset_base.joinpath(
        Path(settings.DATA_SETS[dataset_identifier]["identifier"], "source"))
    path_sanitised_dir = path_dataset_base.joinpath(
        Path(settings.DATA_SETS[dataset_identifier]["identifier"], "sanitised"))
    path_processed_dir = path_dataset_base.joinpath(
        Path(settings.DATA_SETS[dataset_identifier]["identifier"], "processed"))
    path_database_dir = path_dataset_base.joinpath(
        Path(settings.DATA_SETS[dataset_identifier]["identifier"], "database", tokeniser_identifier))

    # Load midi files
    LOGGER.info("Loading MIDI files...")
    paths_midi_files = get_paths_files(path_source_dir, ["mid", "midi"])

    # Split into train, test, and val
    train, _, val = split_paths_dataset_files(paths_midi_files)

    # Sanitise files
    if not skip_sanitisation:
        LOGGER.info("Sanitising MIDI files...")
        LOGGER.info("Sanitising train dataset...")
        sanitise_midi_files(train, path_sanitised_dir.joinpath("train"), source_type)
        LOGGER.info("Sanitising validation dataset...")
        sanitise_midi_files(val, path_sanitised_dir.joinpath("val"), source_type)
    else:
        LOGGER.info("Skipping sanitisation...")

    # Process files
    if not skip_processing:
        LOGGER.info("Processing MIDI files...")
        paths_sanitised_train = get_paths_files(path_sanitised_dir.joinpath("train"), ["mid", "midi"])
        paths_sanitised_val = get_paths_files(path_sanitised_dir.joinpath("val"), ["mid", "midi"])
        merge_tracks = source_type == "dual" and dataset_type == "single"
        LOGGER.info("Processing train dataset...")
        process_midi_files(paths_sanitised_train,
                           path_processed_dir.joinpath("train"),
                           augment_transpose=augment_transpose,
                           assign_difficulties=InfoType.DIFFICULTY in info_flags,
                           merge_tracks=merge_tracks)
        LOGGER.info("Processing validation dataset...")
        process_midi_files(paths_sanitised_val,
                           path_processed_dir.joinpath("val"),
                           augment_transpose=augment_transpose,
                           assign_difficulties=InfoType.DIFFICULTY in info_flags,
                           merge_tracks=merge_tracks)
    else:
        LOGGER.info("Skipping processing...")

    # Store dataset
    if not skip_store:
        LOGGER.info("Storing MIDI files...")
        paths_processed_train = get_paths_files(path_processed_dir.joinpath("train"), ["pkl"])
        paths_processed_val = get_paths_files(path_processed_dir.joinpath("val"), ["pkl"])
        LOGGER.info("Storing train dataset...")
        music_store(paths_processed_train,
                    path_database_dir.joinpath("train.tar"),
                    info_flags,
                    tokeniser_identifier)
        LOGGER.info("Storing validation dataset...")
        music_store(paths_processed_val,
                    path_database_dir.joinpath("val.tar"),
                    info_flags,
                    tokeniser_identifier)
    else:
        LOGGER.info("Skipping storage...")

    LOGGER.info("Done")


def handle_command_data_text(dataset_identifier):
    # Load dataset settings
    source_type = settings.DATA_SETS[dataset_identifier]["source_class"]
    path_dataset_base = Path(settings.DATA_SETS_BASE_PATH)
    path_source_dir = path_dataset_base.joinpath(
        Path(settings.DATA_SETS[dataset_identifier]["identifier"], "source"))
    path_database_dir = path_dataset_base.joinpath(
        Path(settings.DATA_SETS[dataset_identifier]["identifier"], "database", "tokenwise"))

    # Load text files
    LOGGER.info("Loading text files...")
    paths_train_files = get_paths_files(path_source_dir.joinpath("train"), ["txt"])
    paths_val_files = get_paths_files(path_source_dir.joinpath("val"), ["txt"])

    if source_type == "single":
        text_process_store_single(paths_train_files, path_database_dir.joinpath("train.tar"))
    if source_type == "single":
        text_process_store_single(paths_val_files, path_database_dir.joinpath("val.tar"))

    LOGGER.info("Done")
