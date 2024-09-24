import gzip
import pickle
import random
import re
from itertools import islice
from pathlib import Path

import matplotlib
import psutil
import torch
from matplotlib import pyplot as plt
from scoda.enumerations.tokeniser_type import TokeniserType
from scoda.exceptions.tokenisation_exception import TokenisationException
from scoda.tokenisation.gridlike_tokenisation import GridlikeTokeniser
from scoda.tokenisation.midilike_tokenisation import StandardMidilikeTokeniser, CoFMidilikeTokeniser, \
    RelativeMidilikeTokeniser
from scoda.tokenisation.notelike_tokenisation import StandardNotelikeTokeniser, CoFNotelikeTokeniser, \
    LargeVocabularyNotelikeTokeniser, LargeVocabularyCoFNotelikeTokeniser, RelativeNotelikeTokeniser
from scoda.tokenisation.transposed_notelike_tokenisation import TransposedNotelikeTokeniser
from torch import Tensor, Size

from paul3.exceptions.data_exception import DataException
from paul3.utils.settings import Settings

settings = Settings()


# TODO Sort and comment

def get_music_tokeniser(tokeniser_type, return_vocab_size=False):
    # Load tokeniser
    if tokeniser_type == TokeniserType.STANDARD_MIDILIKE_TOKENISER.value:
        tokeniser = StandardMidilikeTokeniser(running_time_sig=True)
    elif tokeniser_type == TokeniserType.RELATIVE_MIDILIKE_TOKENISER.value:
        tokeniser = RelativeMidilikeTokeniser(running_time_sig=True)
    elif tokeniser_type == TokeniserType.COF_MIDILIKE_TOKENISER.value:
        tokeniser = CoFMidilikeTokeniser(running_octave=True, running_time_sig=True)
    elif tokeniser_type == TokeniserType.STANDARD_NOTELIKE_TOKENISER.value:
        tokeniser = StandardNotelikeTokeniser(running_value=True, running_pitch=False, running_time_sig=True)
    elif tokeniser_type == "standard_notelike_tokeniser_remi":
        tokeniser = StandardNotelikeTokeniser(running_value=False, running_pitch=False, running_time_sig=True)
    elif tokeniser_type == TokeniserType.LARGE_VOCABULARY_NOTELIKE_TOKENISER.value:
        tokeniser = LargeVocabularyNotelikeTokeniser(running_time_sig=True)
    elif tokeniser_type == TokeniserType.RELATIVE_NOTELIKE_TOKENISER.value:
        tokeniser = RelativeNotelikeTokeniser(running_value=True, running_time_sig=True)
    elif tokeniser_type == TokeniserType.COF_NOTELIKE_TOKENISER.value:
        tokeniser = CoFNotelikeTokeniser(running_value=True, running_octave=True, running_time_sig=True)
    elif tokeniser_type == TokeniserType.LARGE_VOCABULARY_COF_NOTELIKE_TOKENISER.value:
        tokeniser = LargeVocabularyCoFNotelikeTokeniser(running_time_sig=True)
    elif tokeniser_type == TokeniserType.GRIDLIKE_TOKENISER.value:
        tokeniser = GridlikeTokeniser(running_time_sig=True)
    elif tokeniser_type == TokeniserType.TRANSPOSED_NOTELIKE_TOKENISER.value:
        tokeniser = TransposedNotelikeTokeniser(running_value=True, running_time_sig=True)
    else:
        raise TokenisationException("Unknown tokeniser type")

    if return_vocab_size:
        return tokeniser, tokeniser.VOCAB_SIZE

    return tokeniser


def find_phrase_or_word(query: str, target: str, word_only=True) -> re.Match:
    """Tries to find a word or a sequence of characters in a given sentence

    Args:
        query: The word or phrase to look for
        target: The sentence in which the word or phrase may occur
        word_only: If true only considers words, e.g., `word` must be surrounded by spaces

    Returns: A `match` object

    """
    if word_only:
        return re.compile(fr"\b({query})\b", flags=re.IGNORECASE).search(target)
    else:
        return re.compile(fr"{query}", flags=re.IGNORECASE).search(target)


def chunk_data(iterable, chunk_size):
    iterable_range = iter(iterable)
    return iter(lambda: tuple(islice(iterable_range, chunk_size)), ())


def bin_data(values, n):
    bins = [values[i::n] for i in range(n)]
    return bins


def compress_file(input_file, output_file):
    with open(input_file, "rb") as f_in:
        with gzip.open(output_file, "wb") as f_out:
            f_out.writelines(f_in)


def decompress_file(input_file):
    with gzip.open(input_file, "rb") as f_in:
        data = f_in.read()

    return data


def str_to_bool(s):
    if s.lower() in ('true', 't', 'yes', '1'):
        return True
    elif s.lower() in ('false', 'f', 'no', '0'):
        return False
    raise DataException("Invalid boolean value")


def num_avail_cpus():
    p = psutil.Process()
    return len(p.cpu_affinity())


def num_avail_gpus():
    return torch.cuda.device_count()


def get_paths_files(path_dataset_dir: Path, file_name_extensions: list[str], file_name: str = "*") -> list[Path]:
    paths_files = []

    for fne in file_name_extensions:
        paths_files.extend(path_dataset_dir.rglob(f"{file_name}.{fne}"))

    return paths_files


def split_paths_dataset_files(path_files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    random.Random(settings.RANDOM_SEED).shuffle(path_files)

    split_point_train = int((len(path_files) + 1) * float((settings.DATA_SPLIT_TRAIN / 100)))
    split_point_test = int((len(path_files) + 1) * float((settings.DATA_SPLIT_TEST / 100))) + split_point_train

    paths_train = sorted(path_files[:split_point_train])
    paths_test = sorted(path_files[split_point_train:split_point_test])
    paths_val = sorted(path_files[split_point_test:])

    return paths_train, paths_test, paths_val


def pickle_save(obj, path_file: Path) -> None:
    with gzip.open(path_file, "wb+") as f:
        pickle.dump(obj, f)


def pickle_load(path_file: Path):
    with gzip.open(path_file, "rb") as f:
        return pickle.load(f)


def visualise_attention(matrix: torch.Tensor, output_path: str, distinct_color_map=True, grid_size=16):
    while matrix.dim() > 2:
        matrix = matrix[0]
    matrix = matrix.cpu().detach().numpy()

    if distinct_color_map:
        levels = [0, 1, 2, 3]
        colors = ['darkviolet', 'aqua', 'gold']
        cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
    else:
        cmap = None
        norm = None

    plt.yticks([i - 0.5 for i in range(0, matrix.shape[0] + 1, grid_size)],
               [i if i % 2 == 0 else "" for i in range(0, matrix.shape[0] + 1, grid_size)])
    plt.xticks([i - 0.5 for i in range(0, matrix.shape[1] + 1, grid_size)],
               [i if i % 2 == 0 else "" for i in range(0, matrix.shape[1] + 1, grid_size)])

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black')

    plt.imshow(matrix, cmap=cmap, norm=norm, interpolation='nearest')
    plt.savefig(f"{output_path}.svg", format="svg")


def get_shard_paths(path):
    paths = list(path.parent.glob(f"{path.stem}_*.tar"))
    paths.sort()
    return paths


def compact(x: Tensor):
    t_x = x
    while t_x.dim() < 3:
        t_x = t_x.unsqueeze(0)
    t_x = t_x.reshape(-1, t_x.shape[-2], t_x.shape[-1])
    return t_x


def decompact(x: Tensor, target_shape: Size):
    t_x = x
    while t_x.dim() > len(target_shape):
        t_x = t_x.squeeze(0)
    t_x = t_x.reshape(target_shape)
    return t_x


def get_time_hours_minutes_seconds(time_in_seconds):
    minutes, seconds = divmod(time_in_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return int(hours), int(minutes), int(seconds)
