from abc import abstractmethod, ABC

import torch
from torch import nn
from torch.nn import init


class BaseTransformer(nn.Module):

    def init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if ".weight" in name:
                    init.xavier_uniform_(p)
                elif ".bias" in name:
                    init.zeros_(p)
                else:
                    raise ValueError(f"Unknown parameter type: {name}")

    def reserve_memory(self, batch_size: int, max_len: int, vocab_sizes: dict):
        raise NotImplementedError


class BaseSingleStreamTransformer(BaseTransformer, ABC):

    def reserve_memory(self, batch_size: int, max_len: int, vocab_sizes: dict):
        raise NotImplementedError
