import math

import torch
from torch import nn, Tensor


class SinusoidPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, device: torch.device = None):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.device = device

        encoding = torch.zeros(max_len, d_model, device=device)
        pos = torch.arange(max_len, device=device).unsqueeze(-1)
        i = torch.arange(start=0, end=d_model, step=2, device=device).unsqueeze(0)
        div_term_matrix = 1 / torch.pow(10000, 2 * (i // 2) / d_model)
        encoding[:, 0::2] = torch.sin(pos * div_term_matrix)
        encoding[:, 1::2] = torch.cos(pos * div_term_matrix)
        pos_encoding = encoding.unsqueeze(0)

        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        interim = x + self.pos_encoding[:, :seq_len, :]

        return interim


class RotaryPositionalEncoding(nn.Module):
    # Source: https://blog.eleuther.ai/rotary-embeddings/

    def __init__(self, d_model: int, base: int = 10000, device: torch.device = None):
        super().__init__()

        self.d_model = d_model
        self.base = base
        self.device = device
        self.cached_seq_len = None
        self.cached_sin = None
        self.cached_cos = None

        self.freq_buf = (10000 ** (-2 * torch.arange(0, d_model, 2, device=device) / d_model))

    def _build_cache(self, x: Tensor):
        seq_len = x.size(-2)

        if (seq_len != self.cached_seq_len or
                self.cached_sin is None or
                self.cached_cos is None):
            self.cached_seq_len = seq_len
            positions = torch.arange(seq_len, device=self.device)
            freq = torch.einsum("i,j->ij", positions, self.freq_buf)
            emb = torch.cat((freq, freq), dim=-1)

            self.cached_sin = torch.sin(emb)
            self.cached_cos = torch.cos(emb)

            while self.cached_sin.dim() < x.dim():
                self.cached_sin = self.cached_sin.unsqueeze(dim=0)
                self.cached_cos = self.cached_cos.unsqueeze(dim=0)

        return self.cached_sin, self.cached_cos

    @staticmethod
    def rotate_half(x: Tensor):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rope(x: Tensor, sin: Tensor, cos: Tensor):
        seq_len = x.size(-2)

        sin = sin[..., :seq_len, :]
        cos = cos[..., :seq_len, :]

        return (x * cos) + (RotaryPositionalEncoding.rotate_half(x) * sin)

    def forward(self, x: Tensor):
        cached_sin, cached_cos = self._build_cache(x)
        return RotaryPositionalEncoding.apply_rope(x, cached_sin, cached_cos)


# Tested

class InputEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, device: torch.device = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0, device=device)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * math.pow(self.d_model, 0.5)


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float, device: torch.device = None):
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff, device=device),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model, device=device)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)


class LayerNormalisation(nn.Module):

    def __init__(self, d_model: int, device: torch.device):
        super().__init__()

        self.normalisation = nn.LayerNorm(d_model, eps=1e-6, device=device)

    def forward(self, x: Tensor) -> Tensor:
        return self.normalisation(x)

class Softmax(nn.Module):

    def __init__(self, dim: int):
        super().__init__()

        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.softmax(x)


class Dropout(nn.Module):

    def __init__(self, dropout_rate: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)


class ResidualConnection(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.add(x, y)
