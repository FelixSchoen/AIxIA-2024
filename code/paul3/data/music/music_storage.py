import multiprocessing
from pathlib import Path

import torch
from scoda.elements.bar import Bar
from scoda.enumerations.tokeniser_type import TokeniserType
from scoda.tokenisation.base_tokenisation import BaseTokeniser
from tqdm import tqdm

from paul3.data.music.music_dataset import MusicDataset
from paul3.enumerations.info_type import InfoType
from paul3.exceptions.data_exception import DataException
from paul3.network.layouting import build_fc_information
from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings
from paul3.utils.utility import chunk_data, num_avail_cpus, num_avail_gpus, pickle_load, get_music_tokeniser

LOGGER = get_logger(__name__)
settings = Settings()

PARALLEL = True


def music_store(paths_pickled_files: list[Path],
                path_output: Path,
                info_flags: list[InfoType],
                tokeniser_type: str):
    path_output.parent.mkdir(parents=True, exist_ok=True)

    tokeniser = get_music_tokeniser(tokeniser_type)

    max_num_procs = num_avail_cpus()
    if not PARALLEL:
        max_num_procs = 1

    # Chunk files for parallel tokenisation
    computation_chunk = list(chunk_data(paths_pickled_files, settings.DATA_SAMPLES_PER_CPU))
    iteration_chunks = list(chunk_data(computation_chunk, max_num_procs))

    LOGGER.info("Tokenising and writing chunks...")
    with MusicDataset.Writer(path_output) as writer:
        with multiprocessing.Pool(max_num_procs) as pool:
            i = 0
            for iteration_chunk in tqdm(iteration_chunks, total=len(iteration_chunks)):
                output_iteration_chunk = list(pool.istarmap(_parallel_tokenisation,
                                                            [(chunk_paths_files, tokeniser, info_flags) for
                                                             i, chunk_paths_files in
                                                             enumerate(iteration_chunk)]))
                for output_computation_chunk in output_iteration_chunk:
                    for entry in output_computation_chunk:
                        writer.write(i, entry)
                        i += 1


def _parallel_tokenisation(paths_pickled_files: list[Path],
                           tokeniser: BaseTokeniser,
                           info_flags: list[InfoType]):
    output = []

    # Handle all files in chunk
    for path_file in paths_pickled_files:
        composition = pickle_load(path_file)

        # Check if all tracks have the same amount of bars
        assert (len(track.bars) == len(composition.tracks[0].bars) for track in composition.tracks)
        amount_bars = len(composition.tracks[0].bars)

        # Handle consecutive bars with stride given by settings
        for i in range(0, amount_bars - settings.DATA_MUSIC_CONSECUTIVE_BARS,
                       settings.DATA_MUSIC_CONSECUTIVE_BARS_STRIDE):
            output_dict = dict()

            try:
                # Handle tracks
                for i_track, track in enumerate(composition.tracks):
                    tokeniser.reset()
                    track_tokens = []

                    # Initialise output dict
                    track_output_dict = dict()
                    track_output_dict.setdefault(InfoType.SEQUENCE.value, [])

                    # Handle consecutive bars
                    bars = []
                    for j in range(settings.DATA_MUSIC_CONSECUTIVE_BARS):
                        bars.append(track.bars[i + j])

                    pass_empty_bars(bars)

                    for bar in bars:
                        bar_tokens = tokeniser.tokenise(bar.sequence)
                        bar_tokens.append(tokeniser.TOKEN_SEPARATOR)
                        track_tokens.extend(bar_tokens)

                        if InfoType.DIFFICULTY in info_flags:
                            track_output_dict[InfoType.DIFFICULTY.value].extend(
                                _convert_difficulty_info(bar.sequence._difficulty, bar_tokens))

                    track_output_dict[InfoType.SEQUENCE.value].extend(track_tokens)

                    output_dict[f"track_{i_track:02d}"] = track_output_dict

            except Exception as e:
                LOGGER.info(f"Encountered exception with {path_file.name}: {e}")
                break

            output.append(output_dict)

    return output


def _convert_difficulty_info(difficulty: int,
                             tokenised_sequence: list[int]) -> list[int]:
    if difficulty is None:
        difficulty = -1
    return [difficulty for _ in tokenised_sequence]


def pass_empty_bars(bars: list[Bar]):
    if len(bars) == 0:
        raise DataException("No bars found in track")
    else:
        empty_bars = sum(bar.is_empty() for bar in bars)
        if empty_bars > len(bars) * settings.DATA_MUSIC_MAXIMUM_PERCENTAGE_EMPTY_BARS:
            raise DataException("Too many empty bars found in track")
