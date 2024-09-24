import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from paul3.models.base_classes import BaseSingleStreamTransformer
from paul3.network.layouting import build_mask_padding


class PytorchSingleStreamTransformer(BaseSingleStreamTransformer):

    def __init__(self, d_model: int, d_att: int, d_ff: int, n_heads: int, n_layers: int, dropout_rate: float,
                 vocab_size: dict, max_len: int, device: torch.device = None):
        super().__init__()

        self.d_model = d_model
        self.d_att = d_att
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.device = device

        vocab_size_target = vocab_size["target"][0]

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        self.wrapper = TransformerModel(device, vocab_size_target, d_model, n_heads, d_ff, n_layers, dropout_rate)

    def forward(self, state: Tensor) -> Tensor:
        pad_mask = build_mask_padding(state, state)[:, 0]
        pad_mask = torch.where(pad_mask == 1, torch.tensor(float("-inf")), pad_mask)
        interim = self.wrapper(state.permute(1, 0), pad_mask=pad_mask)
        return interim.permute(1, 0, 2)


class TransformerModel(nn.Module):

    def __init__(self, device, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.device = device
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(device, d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, device=device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model, device=device)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken, device=device)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None, pad_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)
        output = self.transformer_encoder(src, src_mask, pad_mask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, device: torch.device, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
