import multiprocessing
from pathlib import Path
from typing import Callable

import mido
import scoda.midi.midi_file
from scoda.elements.composition import Composition
from scoda.exceptions.bar_exception import BarException
from scoda.exceptions.track_exception import TrackException
from scoda.sequences.sequence import Sequence
from tqdm import tqdm

from paul3.data.music.music_processing import load_standardised_dual_track_midi_file
from paul3.exceptions.data_exception import DataException
from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings
from paul3.utils.utility import find_phrase_or_word, num_avail_cpus

settings = Settings()
LOGGER = get_logger(__name__)
PARALLEL = True


def sanitise_midi_files(paths_files: list[Path], path_output_dir: Path, source_type: str) -> None:
    if source_type == "single":
        sanitisation_method = sanitise_midi_file_single_track
    elif source_type == "dual":
        sanitisation_method = sanitise_midi_file_dual_tracks
    else:
        raise DataException("Unknown dataset type")

    path_output_dir.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(num_avail_cpus()) as pool:
        if PARALLEL:
            for _ in tqdm(pool.istarmap(_sanitise_midi_files,
                                        [(path_file_tuple, path_output_dir, sanitisation_method)
                                         for path_file_tuple in enumerate(paths_files)]),
                          total=len(paths_files),
                          mininterval=1):
                pass
        else:
            for i_path_file_tuple, i_path_output_dir, i_sanitisation_method in \
                    tqdm([(path_file_tuple, path_output_dir, sanitisation_method)
                          for path_file_tuple in enumerate(paths_files)],
                         total=len(paths_files)):
                LOGGER.info(f"Sanitising {i_path_file_tuple[1]}...")
                _sanitise_midi_files(i_path_file_tuple, i_path_output_dir, i_sanitisation_method)


def _sanitise_midi_files(input_tuple, path_output_dir: Path, sanitisation_method: Callable = None):
    i, path_file = input_tuple

    midi_file = None

    try:
        midi_file = mido.MidiFile(path_file)
        is_valid_file = sanitisation_method(midi_file)
    except mido.midifiles.meta.KeySignatureError:
        is_valid_file = False
    except EOFError:
        is_valid_file = False
    except IOError:
        is_valid_file = False
    except BarException:
        is_valid_file = False
    except TrackException:
        is_valid_file = False

    # Store the file if it is valid
    if is_valid_file:
        if not path_output_dir.exists():
            path_output_dir.mkdir(parents=True)

        file_name = path_file.name
        path_output = path_output_dir.joinpath(
            f"{i + 1}_{file_name}" if i is not None else file_name)

        sequences = load_standardised_dual_track_midi_file(midi_file=midi_file)
        midi_file = Sequence.sequences_save(sequences, path_output)

        if len(midi_file.tracks) == 1:
            midi_file.tracks[0].name = settings.DATA_MUSIC_TRACK_NAME_SNGL
        elif len(midi_file.tracks) == 2:
            midi_file.tracks[0].name = settings.DATA_MUSIC_TRACK_NAME_LEAD
            midi_file.tracks[1].name = settings.DATA_MUSIC_TRACK_NAME_ACMP
        else:
            raise DataException("Unknown amount of tracks")

        midi_file.save(path_output)


def sanitise_midi_file_dual_tracks(midi_file: scoda.midi.midi_file.MidiFile) -> bool:
    # Pass 1: Remove files that contain less than two tracks containing notes
    tracks_with_notes = []
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == "note_on":
                tracks_with_notes.append(track)
                break
    if len(tracks_with_notes) < 2:
        return False

    # Operation: Rename signature only track
    _op_rename_signature(midi_file, tracks_with_notes)

    # Operation: Rename tracks
    for track in midi_file.tracks:
        valid_track_name = True

        for name_pair in settings.DATA_MUSIC_VALID_TRACK_NAMES:
            name_lead = name_pair[0]
            name_acmp = name_pair[1]
            is_word = name_pair[2] == "word"

            # Standard track name
            if track.name in settings.DATA_MUSIC_TRACK_NAMES_KNOWN:
                valid_track_name = True
                break
            # Track contains no notes
            elif track not in tracks_with_notes:
                track.name = settings.DATA_MUSIC_TRACK_NAME_META
                valid_track_name = True
                break
            # Lead track name
            elif find_phrase_or_word(name_lead, track.name, word_only=is_word) is not None:
                track.name = settings.DATA_MUSIC_TRACK_NAME_LEAD
                valid_track_name = True
                break
            # Accompanying track name
            elif find_phrase_or_word(name_acmp, track.name, word_only=is_word) is not None:
                track.name = settings.DATA_MUSIC_TRACK_NAME_ACMP
                valid_track_name = True
                break
            # Track name not yet found
            else:
                valid_track_name = False

        if not valid_track_name:
            track.name = settings.DATA_MUSIC_TRACK_NAME_UNKN

    # Operation: If file has named tracks, remove all non-named tracks
    tracks_to_remove = []
    if any(track.name in [settings.DATA_MUSIC_TRACK_NAME_LEAD, settings.DATA_MUSIC_TRACK_NAME_ACMP] for track in
           midi_file.tracks):
        for track in midi_file.tracks:
            if track.name not in [settings.DATA_MUSIC_TRACK_NAME_LEAD, settings.DATA_MUSIC_TRACK_NAME_ACMP,
                                  settings.DATA_MUSIC_TRACK_NAME_SIGN, settings.DATA_MUSIC_TRACK_NAME_META]:
                tracks_to_remove.append(track)
    for track in tracks_to_remove:
        midi_file.tracks.remove(track)

    # Pass 2: Redo pass 1 after removal of all previous tracks
    tracks_with_notes = []
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == "note_on":
                tracks_with_notes.append(track)
                break
    if len(tracks_with_notes) < 2:
        return False

    # Pass 3: Remove files with only one type of named tracks
    if any(track.name in [settings.DATA_MUSIC_TRACK_NAME_LEAD, settings.DATA_MUSIC_TRACK_NAME_ACMP] for track in
           midi_file.tracks):
        if not any(track.name == settings.DATA_MUSIC_TRACK_NAME_LEAD for track in midi_file.tracks) \
                or not any(track.name == settings.DATA_MUSIC_TRACK_NAME_ACMP for track in midi_file.tracks):
            return False

    # Pass 4: Remove files with unknown tracks where amount of tracks with notes is equal to or larger than 2
    tracks_unknown = []
    for t, track in enumerate(midi_file.tracks):
        if track.name in [settings.DATA_MUSIC_TRACK_NAME_UNKN]:
            tracks_unknown.append(t)
    if settings.DATA_MUSIC_SANITISATION_ACCEPT_EXACT_UNKNOWN_TRACKS:
        if len(tracks_with_notes) > 2 and len(tracks_unknown) > 0:
            return False
    else:
        if len(tracks_with_notes) >= 2 and len(tracks_unknown) > 0:
            return False

    # Pass 5: Remove pieces with too many empty bars
    if not _pass_empty_bars(midi_file):
        return False

    return True


def sanitise_midi_file_single_track(midi_file: scoda.midi.midi_file.MidiFile) -> bool:
    # Pass 1: Remove files that contain no tracks containing notes
    tracks_with_notes = []
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == "note_on":
                tracks_with_notes.append(track)
                break
    if len(tracks_with_notes) < 1:
        return False

    # Operation: Rename all leftover tracks
    for track in tracks_with_notes:
        track.name = settings.DATA_MUSIC_TRACK_NAME_SNGL

    # Operation: Rename signature only track
    _op_rename_signature(midi_file, tracks_with_notes)

    # Pass 2: Remove pieces with too many empty bars
    if not _pass_empty_bars(midi_file):
        return False

    return True


def _op_rename_signature(midi_file, tracks_with_notes):
    # Operation: Rename signature only track
    tracks_with_signatures = []
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == "time_signature" or msg.type == "key_signature":
                tracks_with_signatures.append(track)
                break
    for track in tracks_with_signatures:
        if track not in tracks_with_notes:
            track.name = settings.DATA_MUSIC_TRACK_NAME_SIGN


def _pass_empty_bars(midi_file):
    # Pass: Remove pieces with too many empty bars
    sequences = load_standardised_dual_track_midi_file(midi_file=midi_file)

    if len(sequences) == 0:
        return False
    else:
        composition = Composition.from_sequences(sequences)
        for track in composition.tracks:
            empty_bars = sum(bar.is_empty() for bar in track.bars)
            if empty_bars > len(track.bars) * settings.DATA_MUSIC_MAXIMUM_PERCENTAGE_EMPTY_BARS:
                return False

    return True
