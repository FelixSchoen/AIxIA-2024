from abc import ABC, abstractmethod

import torch
from scoda.sequences.sequence import Sequence
from scoda.tokenisation.base_tokenisation import BaseTokeniser
from torch import Tensor


class InferenceConstrainer(ABC):
    MASK_VALUE = -1e9
    STOP_TOKEN = 2

    def __init__(self, vocab_size: int, **kwargs):
        super().__init__()

        self.vocab_size = vocab_size

    @abstractmethod
    def constrain(self, states: Tensor, logits: Tensor):
        pass


class MinimumLengthConstrainer(InferenceConstrainer):

    def __init__(self, vocab_size: int, min_len: int, **kwargs):
        super().__init__(vocab_size)

        self.min_len = min_len

    def constrain(self, states: Tensor, logits: Tensor):
        constrained_logits = torch.clone(logits)
        constrained_logits[:, :self.min_len, self.STOP_TOKEN] = self.MASK_VALUE
        return constrained_logits


class MinimumBarsConstrainer(InferenceConstrainer):

    def __init__(self, vocab_size: int, tokeniser: BaseTokeniser, min_bars: int, **kwargs):
        super().__init__(vocab_size)

        self.tokeniser = tokeniser
        self.min_bars = min_bars

    def constrain(self, states: Tensor, logits: Tensor):
        constrained_logits = torch.clone(logits)

        for i, state in enumerate(torch.split(states, 1)):
            sequence = self.tokeniser.detokenise(state.tolist()[0])
            bars = Sequence.sequences_split_bars([sequence], meta_track_index=0)
            if len(bars) < self.min_bars:
                constrained_logits[i, :, self.STOP_TOKEN] = self.MASK_VALUE

        return constrained_logits
