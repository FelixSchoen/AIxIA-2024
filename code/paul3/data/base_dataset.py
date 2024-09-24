import collections
from abc import ABC
from pathlib import Path

import torch
import webdataset as wds
from torch.utils.data import IterableDataset
from webdataset.utils import pytorch_worker_info

from paul3.utils.paul_logging import get_logger
from paul3.utils.settings import Settings
from paul3.utils.utility import get_shard_paths

LOGGER = get_logger(__name__)
SETTINGS = Settings()


class BaseDataset(IterableDataset, ABC):

    def __init__(self, path: Path, resampled: bool, shuffle: bool, max_len: int, max_entries: int,
                 compatible_divisor: int):
        """

        Args:
            path: Path of the tar files.
            max_len: Maximum length of entries.
            max_entries: Maximum number of entries to include.
            compatible_divisor: A value the length of the individual entries have to be divisible by. Entries are padded to a length satisfying this criteria.
        """
        super().__init__()

        self.path = path
        self.resampled = resampled
        self.shuffle = shuffle
        self.max_len = max_len
        self.max_entries = max_entries
        self.compatible_divisor = compatible_divisor

    # I/O

    @property
    def reader(self):
        return self.Reader(self.path, resampled=self.resampled, shuffle=self.shuffle)

    class Reader:

        def __init__(self, path: Path, resampled: bool = True, shuffle: bool = False):
            super().__init__()
            self.path = path
            self.shuffle = shuffle

            shard_paths = get_shard_paths(self.path)
            self.dataset = (
                wds.WebDataset([str(shard_path) for shard_path in shard_paths],
                               resampled=resampled,
                               shardshuffle=self.shuffle,
                               nodesplitter=wds.split_by_node)
                .decode()
            )

            if self.shuffle:
                self.dataset.shuffle(SETTINGS.DATA_SHUFFLE_BUFFER_SIZE)

        def __iter__(self):
            yield from self.dataset

    class Writer:
        def __init__(self, path: Path):
            super().__init__()
            self.path = path

            self.stream = None

        def __enter__(self):
            self.clear_disk()
            self.stream = wds.ShardWriter(f"{self.path.parent.joinpath(self.path.stem)}_%04d.tar",
                                          maxcount=SETTINGS.DATA_SHARD_LENGTH)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stream.close()

        def write(self, index: int, data: dict):
            _tmp = dict()
            _tmp["__key__"] = f"{index:08}"
            for key in data.keys():
                _tmp[f"{key}.pyd"] = data[key]
            self.stream.write(_tmp)

        def clear_disk(self):
            """Clears the disk of all shards."""
            for path in get_shard_paths(self.path):
                path.unlink()

    # Iteration

    def __iter__(self):
        _, _, _, num_workers = pytorch_worker_info()
        yielded_entries = 0

        for entry in self.reader:
            if self.max_entries != -1 and yielded_entries >= self.max_entries:
                break

            entry = self.convert_data(entry)
            if entry is None:
                continue
            else:
                yielded_entries += num_workers
                yield entry

    # Data Conversion

    def convert_data(self, data):
        """Converts data from the webdataset representation to the internal one."""
        out_dict = dict()

        is_valid = True

        for key in data.keys():
            if key.startswith("__"):
                continue

            is_valid = self._iter_callback(data, key, out_dict)
            if not is_valid:
                break

        if is_valid:
            return out_dict
        return None

    # Collate Function

    def collate_fn(self, batch):
        if isinstance(batch[0], torch.Tensor):
            target_length = torch.max(torch.tensor([tensor.size(0) for tensor in batch]))
            target_length = torch.add(
                torch.mul(torch.ceil(torch.div(target_length, self.compatible_divisor)), self.compatible_divisor),
                1).int()
            tensors = [torch.nn.functional.pad(tensor, (0, target_length - tensor.shape[0]), value=0)
                       for tensor in batch]
            return torch.stack(tensors, 0)
        elif isinstance(batch[0], (int, float)):
            return torch.tensor(batch, dtype=torch.float32)
        elif isinstance(batch[0], (str, bytes)):
            return batch
        elif isinstance(batch[0], collections.abc.Mapping):
            return {key: self.collate_fn([d[key] for d in batch]) for key in batch[0]}
        elif isinstance(batch[0], collections.abc.Sequence):
            transposed = zip(*batch)
            return [self.collate_fn(samples) for samples in transposed]

        raise TypeError(f"Unsupported type for collate function: {type(batch[0])}")

    # Callbacks

    def _iter_callback(self, data, key, out_dict):
        """
        Callback for iterating over the dataset. This method is called for each entry in the dataset. Allows for custom handling of the data.
        """
        raise NotImplementedError
