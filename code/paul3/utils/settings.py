import json
from pathlib import Path

# noinspection PyUnresolvedReferences
from paul3.utils.istarmap import istarmap


class Settings:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Settings, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        settings_file_path = Path(__file__).parent.parent.parent.joinpath("cfg/system_config.json")
        with open(settings_file_path) as settings_file:
            self._settings = json.load(settings_file)

        environment_file_path = Path(__file__).parent.parent.parent.joinpath("cfg/environment_config.json")
        with open(environment_file_path) as environment_file:
            self._environment = json.load(environment_file)

        dsinfo_file_path = Path(__file__).parent.parent.parent.joinpath("cfg/datasets_config.json")
        with open(dsinfo_file_path) as dsinfo_file:
            self._dsinfo = json.load(dsinfo_file)

        # General Settings

        """Seed used for many random operations"""
        self.RANDOM_SEED: float = 0

        """Base path to the datasets"""
        self.DATA_SETS_BASE_PATH: str = ""
        """Base path to the model weights, input, and output"""
        self.MODELS_BASE_PATH: str = ""

        # Data Settings

        """Dictionary containing information about all the datasets"""
        self.DATA_SETS: dict = {}

        # - General Settings

        """Percentage of data to use for the training set"""
        self.DATA_SPLIT_TRAIN: int = 0
        """Percentage of data to use for the test set"""
        self.DATA_SPLIT_TEST: int = 0
        """Percentage of data to use for the validation set"""
        self.DATA_SPLIT_VAL: int = 0
        """Determines how many samples can fit into a single shard"""
        self.DATA_SHARD_LENGTH: int = 0
        """Determines how many samples each core processes during the preprocessing of the data"""
        self.DATA_SAMPLES_PER_CPU: int = 0
        """The size of the buffer for data loading"""
        self.DATA_BATCH_BUFFER_SIZE: int = 0
        """The size of the shuffle buffer"""
        self.DATA_SHUFFLE_BUFFER_SIZE: int = 0
        """The prefetch factor used for the data loaders"""
        self.DATA_PREFETCH_FACTOR: int = 0

        # - Music Settings

        """Determines the amount of consecutive bars to make up a training sample"""
        self.DATA_MUSIC_CONSECUTIVE_BARS: int = 0
        """Determines the the stride of samples, i.e., the distance in bars between the starting points of two 
        samples"""
        self.DATA_MUSIC_CONSECUTIVE_BARS_STRIDE: int = 0
        """Determines whether MIDI files that contain tracks marked as unknown should be used"""
        self.DATA_MUSIC_ACCEPT_UNKNOWN_TRACKS: bool = False
        """Determines whether MIDI files that contain the exact amount of needed tracks should be used even if the tracks do not have valid names"""
        self.DATA_MUSIC_SANITISATION_ACCEPT_EXACT_UNKNOWN_TRACKS: bool = False
        """Gives the maximum percentage of the bars in a sequence that can be empty before it is disregarded"""
        self.DATA_MUSIC_MAXIMUM_PERCENTAGE_EMPTY_BARS: int = 0
        """Which token represents the summary token for FC attention"""
        self.DATA_MUSIC_SUMMARY_TOKEN: int = 0
        """Which segments are considered structure segments"""
        self.DATA_MUSIC_STRUCTURE_SEGMENTS: list[int] = []
        """Factors by which training files are temporally scaled"""
        self.DATA_MUSIC_SCALE_FACTORS: list[int] = [0]

        """Name of a single track"""
        self.DATA_MUSIC_TRACK_NAME_SNGL: str = ""
        """Name of the lead track"""
        self.DATA_MUSIC_TRACK_NAME_LEAD: str = ""
        """Name of the accompanying track"""
        self.DATA_MUSIC_TRACK_NAME_ACMP: str = ""
        """Name of the track containing meta information"""
        self.DATA_MUSIC_TRACK_NAME_META: str = ""
        """Name of the track containing time signatures"""
        self.DATA_MUSIC_TRACK_NAME_SIGN: str = ""
        """Name of an unidentifiable or unknown track"""
        self.DATA_MUSIC_TRACK_NAME_UNKN: str = ""
        """A list comprised of all the above"""
        self.DATA_MUSIC_TRACK_NAMES_KNOWN: list[str] = [""]
        """Defines all valid track names of tracks in a dataset to be cleansed"""
        self.DATA_MUSIC_VALID_TRACK_NAMES: list[str] = [""]

        # Train Settings

        """Determines how often the training process logs information"""
        self.TRAIN_LOG_FREQUENCY: int = 0

        """How many checkpoints to keep. Older checkpoints will be deleted. 0 means all checkpoints are kept."""
        self.TRAIN_CHECKPOINTS_TO_KEEP: int = 0

        """Dictionary containing information about all the datasets"""
        self.TRAIN_RESERVE_MEMORY: bool = False

        """Timeout in seconds for the training process."""
        self.TRAIN_TIMEOUT: int = 0

        self.load_values()

    def load_values(self):
        # General Settings

        self.RANDOM_SEED = self._settings["general"]["seed"]

        self.DATA_SETS_BASE_PATH = self._environment["paths"]["datasets"]
        self.MODELS_BASE_PATH = self._environment["paths"]["models"]

        # Data Settings

        self.DATA_SETS = self._dsinfo

        # - General Settings

        self.DATA_SPLIT_TRAIN = self._settings["data"]["general"]["split"]["train"]
        self.DATA_SPLIT_TEST = self._settings["data"]["general"]["split"]["test"]
        self.DATA_SPLIT_VAL = self._settings["data"]["general"]["split"]["val"]
        assert sum([self.DATA_SPLIT_TRAIN, self.DATA_SPLIT_TEST, self.DATA_SPLIT_VAL]) == 100
        self.DATA_SHARD_LENGTH = self._settings["data"]["general"]["shard_length"]
        self.DATA_SAMPLES_PER_CPU = self._settings["data"]["general"]["samples_per_cpu"]
        self.DATA_BATCH_BUFFER_SIZE = self._settings["data"]["general"]["batch_buffer_size"]
        self.DATA_SHUFFLE_BUFFER_SIZE = self._settings["data"]["general"]["shuffle_buffer_size"]
        self.DATA_PREFETCH_FACTOR = self._settings["data"]["general"]["prefetch_factor"]

        # - Music Settings

        self.DATA_MUSIC_CONSECUTIVE_BARS = self._settings["data"]["music"]["consecutive_bars"]
        self.DATA_MUSIC_CONSECUTIVE_BARS_STRIDE = self._settings["data"]["music"]["consecutive_bars_stride"]
        self.DATA_MUSIC_ACCEPT_UNKNOWN_TRACKS = bool(self._settings["data"]["music"]["accept_unknown_tracks"])
        self.DATA_MUSIC_SANITISATION_ACCEPT_EXACT_UNKNOWN_TRACKS = bool(
            self._settings["data"]["music"]["sanitisation_accept_exact_unknown_tracks"])
        self.DATA_MUSIC_MAXIMUM_PERCENTAGE_EMPTY_BARS = self._settings["data"]["music"]["maximum_percentage_empty_bars"] / 100
        self.DATA_MUSIC_SUMMARY_TOKEN = self._settings["data"]["music"]["summary_token"]
        self.DATA_MUSIC_STRUCTURE_SEGMENTS = self._settings["data"]["music"]["structure_segments"]
        self.DATA_MUSIC_SCALE_FACTORS = self._settings["data"]["music"]["scale_factors"]

        self.DATA_MUSIC_TRACK_NAME_SNGL = self._settings["data"]["music"]["variables"]["track_names"]["sngl"]
        self.DATA_MUSIC_TRACK_NAME_LEAD = self._settings["data"]["music"]["variables"]["track_names"]["lead"]
        self.DATA_MUSIC_TRACK_NAME_ACMP = self._settings["data"]["music"]["variables"]["track_names"]["acmp"]
        self.DATA_MUSIC_TRACK_NAME_META = self._settings["data"]["music"]["variables"]["track_names"]["meta"]
        self.DATA_MUSIC_TRACK_NAME_SIGN = self._settings["data"]["music"]["variables"]["track_names"]["sign"]
        self.DATA_MUSIC_TRACK_NAME_UNKN = self._settings["data"]["music"]["variables"]["track_names"]["unkn"]
        self.DATA_MUSIC_TRACK_NAMES_KNOWN = [self.DATA_MUSIC_TRACK_NAME_LEAD, self.DATA_MUSIC_TRACK_NAME_ACMP,
                                             self.DATA_MUSIC_TRACK_NAME_META,
                                             self.DATA_MUSIC_TRACK_NAME_SIGN, self.DATA_MUSIC_TRACK_NAME_UNKN]
        self.DATA_MUSIC_VALID_TRACK_NAMES = self._settings["data"]["music"]["variables"]["valid_track_names"]

        # Train Settings

        self.TRAIN_LOG_FREQUENCY = self._settings["train"]["log_frequency"]
        self.TRAIN_CHECKPOINTS_TO_KEEP = self._settings["train"]["checkpoints_to_keep"]
        self.TRAIN_RESERVE_MEMORY = bool(self._settings["train"]["reserve_memory"])
        self.TRAIN_TIMEOUT = self._settings["train"]["timeout"]
