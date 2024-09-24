import torch
from torch import Tensor, nn

from paul3.enumerations.attention_output_mode import AttentionOutputMode
from paul3.network.attention import MultiHeadAttention, ScaledDotProductAttention, RelativeGlobalAttention, \
    TritonBlockSparseAttention
from paul3.network.layers import LayerNormalisation, Dropout, ResidualConnection, PositionWiseFeedForward
from paul3.utils.paul_logging import get_logger

LOGGER = get_logger()


class BaseModule(nn.Module):

    def __init__(self, device: torch.device):
        super().__init__()

        self.device = device


class VanillaMultiHeadAttentionModule(BaseModule):

    def __init__(self, d_model: int, d_att: int, n_heads: int, dropout_rate: float, device: torch.device):
        super().__init__(device)

        self.normalisation = LayerNormalisation(d_model, device)
        self.sha = ScaledDotProductAttention(d_att, dropout_rate, device)
        self.mha = MultiHeadAttention(d_model, d_att, n_heads, dropout_rate, self.sha, device)
        self.dropout = Dropout(dropout_rate)
        self.residual_connection = ResidualConnection()

    def forward(self, state: Tensor, context: Tensor, mask: Tensor = None,
                output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY) -> Tensor:
        interim_state = self.normalisation(state)
        interim_context = self.normalisation(context)
        interim = self.mha(q=interim_state, k=interim_context, v=interim_context, mask=mask,
                           output_mode=output_mode)
        interim = self.dropout(interim)
        interim = self.residual_connection(interim, state)

        return interim


class RelativeGlobalMultiHeadAttentionModule(BaseModule):

    def __init__(self, d_model: int, d_att: int, n_heads: int, dropout_rate: float, max_len: int, max_dist: int,
                 device: torch.device):
        super().__init__(device)

        self.normalisation = LayerNormalisation(d_model, device)
        self.sha = RelativeGlobalAttention(d_att, dropout_rate, max_len, max_dist, device)
        self.mha = MultiHeadAttention(d_model, d_att, n_heads, dropout_rate, self.sha, device)
        self.dropout = Dropout(dropout_rate)
        self.residual_connection = ResidualConnection()

    def forward(self, state: Tensor, context: Tensor, mask: Tensor = None,
                output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY) -> Tensor:
        interim_state = self.normalisation(state)
        interim_context = self.normalisation(context)
        interim = self.mha(q=interim_state, k=interim_context, v=interim_context, mask=mask,
                           output_mode=output_mode)
        interim = self.dropout(interim)
        interim = self.residual_connection(interim, state)

        return interim


class VanillaFeedForwardModule(BaseModule):

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float, device: torch.device):
        super().__init__(device)

        self.normalisation = LayerNormalisation(d_model, device)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout_rate, device)
        self.dropout = Dropout(dropout_rate)
        self.residual_connection = ResidualConnection()

    def forward(self, state: Tensor) -> Tensor:
        interim = self.normalisation(state)
        interim = self.position_wise_feed_forward(interim)
        interim = self.dropout(interim)
        interim = self.residual_connection(interim, state)

        return interim


class FCAttentionModule(BaseModule):

    def __init__(self, d_model: int, d_att: int, n_heads: int, block_size: int, dropout_rate: float,
                 info_max_dist: int, device: torch.device):
        super().__init__(device)

        self.d_model = d_model
        self.d_att = d_att
        self.n_heads = n_heads
        self.block_size = block_size
        self.info_max_dist = info_max_dist

        self.normalisation = LayerNormalisation(d_model, device)
        self.q_net = nn.Linear(self.d_model, self.d_att, bias=False, device=self.device)
        self.k_net = nn.Linear(self.d_model, self.d_att, bias=False, device=self.device)
        self.v_net = nn.Linear(self.d_model, self.d_model, bias=False, device=self.device)
        self.s_net = nn.Linear(self.d_model, self.d_model, bias=False, device=self.device)
        self.q_bias_sum = nn.Parameter(torch.zeros(self.d_att, device=self.device))
        self.q_bias_agg = nn.Parameter(torch.zeros(self.d_att, device=self.device))
        self.k_bias_sum = nn.Parameter(torch.zeros(self.d_att, device=self.device))
        self.k_bias_agg = nn.Parameter(torch.zeros(self.d_att, device=self.device))
        self.v_bias_sum = nn.Parameter(torch.zeros(self.d_model, device=self.device))
        self.v_bias_agg = nn.Parameter(torch.zeros(self.d_model, device=self.device))
        self.summary_attention = TritonBlockSparseAttention(d_att, block_size, dropout_rate, device)
        self.mha_summary = MultiHeadAttention(d_model, d_att, n_heads, dropout_rate, self.summary_attention, device,
                                              transform_inputs=False)
        self.aggregation_attention = TritonBlockSparseAttention(d_att, block_size, dropout_rate, device)
        self.mha_aggregation = MultiHeadAttention(d_model, d_att, n_heads, dropout_rate, self.aggregation_attention,
                                                  device, transform_inputs=False)

        if self.info_max_dist != -1:
            self.info_embedding = nn.Embedding(2 * info_max_dist + 1, d_att, device=device)

        self.dropout = Dropout(dropout_rate)
        self.residual_connection = ResidualConnection()

    def prime(self, sparsity_layouts: list[list]):
        self.summary_attention.prime(sparsity_layouts[0])
        self.aggregation_attention.prime(sparsity_layouts[1])

    def reprime(self, other: "FCAttentionModule"):
        self.summary_attention.reprime(other.summary_attention)
        self.aggregation_attention.reprime(other.aggregation_attention)

    def forward(self, state: Tensor, context: Tensor,
                shuffle_layout_state: Tensor, shuffle_layout_context: Tensor, shuffle_layout_reverse: Tensor,
                classification_context_shuffled: Tensor,
                mask_summary: Tensor, mask_aggregation: Tensor,
                q_info: Tensor = None, k_info: Tensor = None,
                output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY) -> Tensor:
        interim_state = self.normalisation(state)
        interim_context = self.normalisation(context)

        interim_state_shuffled = torch.gather(interim_state, -2, shuffle_layout_state)
        interim_context_shuffled = torch.gather(interim_context, -2, shuffle_layout_context)

        # Relative information
        if q_info is not None and k_info is not None:
            distances = (torch.arange(-context.size(-1), context.size() + 1, device=self.device)
                         .clamp(min=-self.info_max_dist, max=self.info_max_dist) + self.info_max_dist).detach()
            distance_embeddings = self.info_embedding(distances).unsqueeze(0)


        q_s = self.q_net(interim_state_shuffled) + self.q_bias_sum
        k_s = self.k_net(interim_context_shuffled) + self.k_bias_sum
        v_s = self.v_net(interim_context_shuffled) + self.v_bias_sum

        # Summary Attention
        output_summary = self.mha_summary(q=q_s, k=k_s, v=v_s, mask=mask_summary,
                                          output_mode=output_mode)
        output_summary = self.s_net(output_summary) * classification_context_shuffled.unsqueeze(-1)
        output_summary += interim_context_shuffled * (classification_context_shuffled.logical_not().unsqueeze(-1))

        q_a = self.q_net(interim_state_shuffled) + self.q_bias_agg
        k_a = self.k_net(output_summary) + self.k_bias_agg
        v_a = self.v_net(output_summary) + self.v_bias_agg

        output_aggregation = self.mha_aggregation(q=q_a, k=k_a, v=v_a, mask=mask_aggregation, output_mode=output_mode)

        output = torch.gather(output_aggregation, -2, shuffle_layout_reverse)

        return output
