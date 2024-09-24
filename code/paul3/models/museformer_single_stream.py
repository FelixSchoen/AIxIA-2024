import torch
from torch import Tensor
from torch import nn

from paul3.models.base_classes import BaseSingleStreamTransformer
from paul3.network.layers import LayerNormalisation, InputEmbedding, SinusoidPositionalEncoding, Dropout
from paul3.network.layouting import create_shuffle_layout, build_layout_summary, build_layout_sparsity, \
    build_mask_padded_lookahead, build_layout_aggregation
from paul3.network.modules import VanillaFeedForwardModule, FCAttentionModule
from paul3.utils.settings import Settings

settings = Settings()


class Museformer(BaseSingleStreamTransformer):

    def __init__(self, d_model: int, d_att: int, d_ff: int, n_heads: int, n_layers: int, block_size: int,
                 dropout_rate: float, vocab_size: dict, max_len: int, device: torch.device = None):
        super().__init__()

        self.d_model = d_model
        self.d_att = d_att
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.device = device

        vocab_size_target = vocab_size["target"][0]

        self.decoder = MuseformerEncoder(d_model=d_model, d_att=d_att, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers,
                                         block_size=block_size, dropout_rate=dropout_rate, vocab_size=vocab_size_target,
                                         max_len=max_len, device=device)
        self.normalisation = LayerNormalisation(d_model, device)
        self.linear = nn.Linear(d_model, vocab_size_target, device=device)

    def forward(self, state: Tensor) -> Tensor:
        summary_token = settings.DATA_MUSIC_SUMMARY_TOKEN
        structure_segments = settings.DATA_MUSIC_STRUCTURE_SEGMENTS
        structure_segments_lookahead = [segment for segment in structure_segments if segment <= 0]

        assert state.size(
            -1) % self.block_size == 0, f"Input size {state.size(-1)} not compatible with block size {self.block_size}"

        # Create masks
        (shuffle_layout_state_tokens, shuffle_layout_state_values, shuffle_layout_state_reverse_values,
         classification_state_shuffled, summary_sparsity_shuffled, aggregation_sparsity_shuffled,
         summary_layout_shuffled, aggregation_layout_shuffled_flat) = self.create_fc_masks(
            state, self.d_model, self.block_size, summary_token, structure_segments_lookahead,
            self.device)

        self.decoder.prime([summary_sparsity_shuffled, aggregation_sparsity_shuffled])

        interim = self.decoder(state, [{"shuffle_layout_state": shuffle_layout_state_values,
                                        "shuffle_layout_context": shuffle_layout_state_values,
                                        "shuffle_layout_reverse": shuffle_layout_state_reverse_values,
                                        "classification_context_shuffled": classification_state_shuffled,
                                        "mask_summary": 1 - summary_layout_shuffled,
                                        "mask_aggregation": 1 - aggregation_layout_shuffled_flat}])

        interim = self.normalisation(interim)
        interim = self.linear(interim)

        return interim

    @staticmethod
    def create_fc_masks(state: Tensor, d_model: int, block_size: int, summary_token: int,
                        structure_segments_lookahead: list[int], device: torch.device):
        # Create masks
        with torch.no_grad():
            # Create classification layout
            classification_state = (state == 4).clone().detach().to(device)

            # Create shuffle layouts
            shuffle_layout_state_tokens = create_shuffle_layout(classification_state, state.size(-1))
            shuffle_layout_state_values = create_shuffle_layout(classification_state, d_model)
            shuffle_layout_state_reverse_values = torch.argsort(shuffle_layout_state_values, dim=-2)

            # Apply shuffling
            classification_state_shuffled = torch.gather(classification_state, -1,
                                                         shuffle_layout_state_tokens[..., 0])

            # Summary layouting
            summary_layout = build_layout_summary(state, 4)
            summary_layout_shuffled = torch.gather(
                torch.gather(summary_layout, -1, shuffle_layout_state_tokens.transpose(-1, -2)), -2,
                shuffle_layout_state_tokens)
            summary_sparsity_shuffled = build_layout_sparsity(summary_layout_shuffled, block_size)

            # Aggregation layouting
            padded_lookahead_mask = build_mask_padded_lookahead(state, state, device).to(
                state.dtype)
            aggregation_layout = build_layout_aggregation(state, state, summary_token,
                                                          structure_segments_lookahead,
                                                          device)
            aggregation_layout_masked = aggregation_layout * (1 - padded_lookahead_mask)
            aggregation_layout_shuffled = torch.gather(
                torch.gather(aggregation_layout_masked, -1, shuffle_layout_state_tokens.transpose(-2, -1)), -2,
                shuffle_layout_state_tokens)
            aggregation_layout_shuffled_flat = aggregation_layout_shuffled.clone().detach()
            # TODO Prettier implementation
            aggregation_layout_shuffled_flat[aggregation_layout_shuffled_flat == 2] = 1
            aggregation_sparsity_shuffled = build_layout_sparsity(aggregation_layout_shuffled_flat, block_size)

        return (shuffle_layout_state_tokens, shuffle_layout_state_values, shuffle_layout_state_reverse_values,
                classification_state_shuffled, summary_sparsity_shuffled, aggregation_sparsity_shuffled,
                summary_layout_shuffled, aggregation_layout_shuffled_flat)


class MuseformerEncoder(nn.Module):

    def __init__(self, d_model: int, d_att: int, d_ff: int, n_heads: int, n_layers: int, block_size: int,
                 dropout_rate: float, vocab_size: int, max_len: int, device: torch.device):
        super().__init__()

        self.d_model = d_model
        self.d_att = d_att
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.device = device

        self.input_embedding = InputEmbedding(vocab_size, d_model, device=device)
        self.positional_encoding = SinusoidPositionalEncoding(self.d_model, max_len, device=device)
        self.dropout = Dropout(dropout_rate)
        self.encoder_layers = nn.ModuleList([nn.ModuleList([FCAttentionModule(d_model=self.d_model,
                                                                              d_att=self.d_att,
                                                                              n_heads=self.n_heads,
                                                                              block_size=self.block_size,
                                                                              dropout_rate=self.dropout_rate,
                                                                              info_max_dist=-1,
                                                                              device=device),
                                                            VanillaFeedForwardModule(d_model=self.d_model,
                                                                                     d_ff=self.d_ff,
                                                                                     dropout_rate=self.dropout_rate,
                                                                                     device=device)]) for _ in
                                             range(n_layers)])

    def prime(self, sparsity_layouts: list[list]):
        subject = None
        for i, encoder_layer in enumerate(self.encoder_layers):
            if i == 0:
                encoder_layer[0].prime(sparsity_layouts)
                subject = encoder_layer[0]
            else:
                encoder_layer[0].trans_prime(subject)

    def forward(self, state: Tensor, arguments: list[dict]) -> Tensor:
        interim = self.input_embedding(state)
        interim = self.positional_encoding(interim)
        interim = self.dropout(interim)

        for encoder_layer in self.encoder_layers:
            interim = encoder_layer[0](state=interim, context=interim, **arguments[0])
            interim = encoder_layer[1](interim)

        return interim
