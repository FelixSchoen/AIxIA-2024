import copy
import multiprocessing
from pathlib import Path

import mido
from scoda.elements.bar import Bar
from scoda.elements.composition import Composition
from scoda.midi.midi_file import MidiFile
from scoda.sequences.sequence import Sequence
from tqdm import tqdm

from paul3.enumerations.info_type import InfoType
from paul3.exceptions.data_exception import DataException
from paul3.utils.settings import Settings
from paul3.utils.utility import pickle_save, num_avail_cpus

settings = Settings()
PARALLEL = True


def process_midi_files(paths_files: list[Path],
                       path_output_dir: Path,
                       augment_transpose: bool = True,
                       assign_difficulties: bool = False,
                       merge_tracks: bool = False) -> None:
    path_output_dir.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(num_avail_cpus()) as pool:
        if PARALLEL:
            for _ in tqdm(pool.istarmap(_process_midi_files,
                                        [(path_file, path_output_dir, augment_transpose, assign_difficulties,
                                          merge_tracks)
                                         for path_file in paths_files]),
                          total=len(paths_files)):
                pass
        else:
            for i_path_file, i_path_output_dir, i_augment_transpose, i_assign_difficulties, i_merge_tracks in \
                    tqdm([(path_file, path_output_dir, augment_transpose, assign_difficulties, merge_tracks)
                          for path_file in paths_files],
                         total=len(paths_files)):
                _process_midi_files(i_path_file, i_path_output_dir, i_augment_transpose, i_assign_difficulties,
                                    i_merge_tracks)


def _process_midi_files(path_file: Path,
                        path_output_dir: Path,
                        augment_transpose: bool = True,
                        assign_difficulties: bool = False,
                        merge_tracks: bool = False):
    # Load compositions and temporally scale using scaling factors
    compositions_scaled = _load_and_scale(path_file, merge_tracks)

    # Assign difficulties
    if assign_difficulties:
        for composition in compositions_scaled:
            for track in composition.tracks:
                for bar in track.bars:
                    _assign_difficulty(bar)

    compositions_augmented = []

    # Augment by transposing
    for composition in compositions_scaled:
        compositions_augmented.append(composition)
        if augment_transpose:
            compositions_augmented.extend(_augment_transpose(composition, assign_difficulties))

    # Store compositions to disk
    for i, composition in enumerate(compositions_augmented):
        split = path_file.name.split("_", 1)
        previous_index = split[0]
        previous_name = split[1].split(".", 1)[0]
        path_output = path_output_dir.joinpath(f"{previous_index}_{i + 1}_{previous_name}.pkl")
        pickle_save(composition, path_output)


def _load_and_scale(path_file: Path, merge_sequences: bool = False) -> list[Composition]:
    compositions = []
    sequences = load_standardised_dual_track_midi_file(path_file)

    if sequences is None or len(sequences) == 0:
        return compositions

    # Scale sequences by given factors
    for scale_factor in settings.DATA_MUSIC_SCALE_FACTORS:
        scaled_sequences = []

        for sequence in sequences:
            scaled_sequence = copy.copy(sequence)

            scaled_sequence.quantise_and_normalise()

            # Scale by scale factor, using first track as meta information track
            scaled_sequence.scale(scale_factor, sequences[0])

            scaled_sequences.append(scaled_sequence)

        if merge_sequences:
            sequence = scaled_sequences[0]
            sequence.merge(scaled_sequences[1:])
            scaled_sequences = [sequence]

        # Create composition from scaled sequences
        composition = Composition.from_sequences(scaled_sequences)
        compositions.append(composition)

    return compositions


def _assign_difficulty(bar: Bar) -> float:
    bar.difficulty()
    assert bar.sequence._difficulty is not None
    return bar.difficulty()


def _augment_transpose(composition: Composition, assign_difficulties) -> list[Composition]:
    compositions_transposed = []

    for transpose_by in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
        composition_cpy = copy.copy(composition)
        compositions_transposed.append(composition_cpy)

        for track in composition_cpy.tracks:
            for bar in track.bars:
                bar.transpose(transpose_by)
                if assign_difficulties:
                    bar.difficulty()

    return compositions_transposed


def _convert_to_sequences(bars: list[list[Bar]], assign_difficulties: bool) -> list[list[tuple[Sequence, dict]]]:
    result_track = []
    for track in bars:
        result_bar = []
        for bar in track:
            seq = bar.sequence
            seq.normalise()

            dif = -1
            if assign_difficulties:
                dif = bar.difficulty()

            result_bar.append((seq, {InfoType.DIFFICULTY: dif}))
        result_track.append(result_bar)

    return result_track


def load_standardised_dual_track_midi_file(path_file: Path = None, midi_file: MidiFile = None) -> list[Sequence]:
    if midi_file is None:
        midi_file = mido.MidiFile(path_file)

    sngl_tracks = []
    lead_tracks = []
    acmp_tracks = []
    sign_tracks = []

    for t, track in enumerate(midi_file.tracks):
        if track.name == settings.DATA_MUSIC_TRACK_NAME_SNGL:
            sngl_tracks.append(t)
        elif track.name == settings.DATA_MUSIC_TRACK_NAME_LEAD:
            lead_tracks.append(t)
        elif track.name == settings.DATA_MUSIC_TRACK_NAME_ACMP:
            acmp_tracks.append(t)
        elif track.name == settings.DATA_MUSIC_TRACK_NAME_SIGN:
            sign_tracks.append(t)
        elif track.name == settings.DATA_MUSIC_TRACK_NAME_META:
            sign_tracks.append(t)
        elif track.name == settings.DATA_MUSIC_TRACK_NAME_UNKN:
            if settings.DATA_MUSIC_ACCEPT_UNKNOWN_TRACKS:
                if len(lead_tracks) == 0:
                    lead_tracks.append(t)
                elif len(acmp_tracks) == 0:
                    acmp_tracks.append(t)
                else:
                    raise DataException("Illegal unknown track encountered")
            else:
                pass
        else:
            raise DataException("Invalid track name encountered")

    if (len(sngl_tracks) != 1) and (len(lead_tracks) == 0 or len(acmp_tracks) == 0):
        return []

    int_midi_file = MidiFile()
    int_midi_file.parse_mido(midi_file)

    if len(sngl_tracks) == 1:
        sequences = Sequence.sequences_load(None, midi_file=int_midi_file,
                                            track_indices=[sngl_tracks],
                                            meta_track_indices=sign_tracks)
    else:
        sequences = Sequence.sequences_load(None, midi_file=int_midi_file,
                                            track_indices=[lead_tracks, acmp_tracks],
                                            meta_track_indices=sign_tracks)

    for sequence in sequences:
        sequence.quantise_and_normalise()

    return sequences
