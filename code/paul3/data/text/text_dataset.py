from pathlib import Path

import torch

from paul3.data.base_dataset import BaseDataset


class TextDataset(BaseDataset):

    def __init__(self, path: Path,
                 resampled: bool = True,
                 shuffle: bool = True,
                 max_len: int = -1,
                 max_entries: int = -1,
                 compatible_divisor: int = 1, ):
        super().__init__(path, resampled, shuffle, max_len, max_entries, compatible_divisor)

    def _iter_callback(self, data, key, out_dict):
        entry = data[key]

        entry.insert(0, 1)
        entry.append(2)
        tensor = torch.tensor(entry)
        out_dict[key[:key.rfind(".")]] = tensor

        return True

    def _sort_callback(self, x):
        return x["target"].shape[0]
        # return int(math.log(x["target"].shape[0], 2))
