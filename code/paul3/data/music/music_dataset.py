from pathlib import Path

import torch

from paul3.data.base_dataset import BaseDataset


class MusicDataset(BaseDataset):

    def __init__(self, path: Path, resampled: bool = True, shuffle: bool = True, max_len: int = -1,
                 max_entries: int = -1,
                 compatible_divisor: int = 1,
                 invalid_tokens=None):
        super().__init__(path, resampled, shuffle, max_len, max_entries, compatible_divisor)

        if invalid_tokens is None:
            invalid_tokens = []

        self.invalid_tokens = invalid_tokens

    def _iter_callback(self, data, key, out_dict):
        entry = data[key]

        for e_key in entry.keys():
            sub_out_dict = dict()
            sub_entry = entry[e_key]

            if self.max_len != -1 and len(sub_entry) + 2 > self.max_len:
                return False

            sub_entry.insert(0, 1)
            sub_entry.append(2)

            for invalid_token in self.invalid_tokens:
                sub_entry = list(filter(lambda a: a != invalid_token, sub_entry))

            tensor = torch.tensor(sub_entry)
            sub_out_dict[e_key] = tensor
            out_dict[key[:key.rfind(".")]] = sub_out_dict

        return True

    def _sort_callback(self, x):
        return int(x["track_00"]["sequence"].shape[0] * 0.1)
