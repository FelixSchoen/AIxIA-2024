import math
from abc import ABC

import torch
from torch import nn, Tensor, Size

from paul3.enumerations.attention_output_mode import AttentionOutputMode
from paul3.network.layers import Dropout
from paul3.utils.paul_logging import get_logger
from paul3.utils.triton_operation_holder import TritonOperationHolder

LOGGER = get_logger(__name__)


# Single Head Attention

class BaseSingleHeadAttention(nn.Module, ABC):

    def __init__(self, device: torch.device) -> None:
        super().__init__()

        self.device = device

    def pass_pre(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 mask: torch.Tensor = None, output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY,
                 **kwargs) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor]:
        return {}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None,
                output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ScaledDotProductAttention(BaseSingleHeadAttention):
    """Scaled Dot-Product Attention as introduced by Vaswani et al. (2017)"""

    def __init__(self, d_att: int, dropout_rate: float, device: torch.device) -> None:
        super().__init__(device)

        self.d_att = d_att
        self.dropout_rate = dropout_rate

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = Dropout(dropout_rate)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None,
                output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor]:
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_att)
        return self.attention_post_processing(attention_scores, mask, output_mode, v)

    def attention_post_processing(self, attention_scores, mask, output_mode, v):
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 1, torch.finfo(attention_scores.dtype).min)

        if output_mode == AttentionOutputMode.SCORES_ONLY:
            return attention_scores

        attention_weights = self.softmax(attention_scores)
        attention_weights = self.dropout(attention_weights)

        if output_mode == AttentionOutputMode.WEIGHTS_ONLY:
            return attention_weights

        output = torch.matmul(attention_weights, v)

        if output_mode == AttentionOutputMode.OUTPUT_ONLY:
            return output
        elif output_mode == AttentionOutputMode.OUTPUT_AND_WEIGHTS:
            return output, attention_weights
        else:
            raise NotImplementedError


class RelativeGlobalAttention(ScaledDotProductAttention):
    """Relative Attention as introduced by Huang et al. (2018) with support for relative encodings for the entire matrix"""

    def __init__(self, d_att: int, dropout_rate: float, max_len: int, max_dist: int, device: torch.device) -> None:
        super().__init__(d_att, dropout_rate, device)

        self.max_len = max_len
        self.max_dist = max_dist

        self.relative_embedding = nn.Embedding(2 * max_dist + 1, d_att, device=device)

    def pass_pre(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 mask: torch.Tensor = None, output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY,
                 **kwargs) -> dict[str, torch.Tensor]:
        distances = (torch.arange(-k.shape[-2], k.shape[-2] + 1, device=self.device)
                     .clamp(min=-self.max_dist, max=self.max_dist) + self.max_dist).detach()
        relative_embeddings = self.relative_embedding(distances).unsqueeze(0)
        relative_embeddings = relative_embeddings.expand(*k.shape[:-2], *relative_embeddings.shape[1:])

        return {"relative_embeddings": relative_embeddings}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, relative_embeddings: torch.Tensor,
                mask: torch.Tensor = None,
                output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor]:
        qr = torch.matmul(q, relative_embeddings.transpose(-2, -1))
        qr = RelativeGlobalAttention._skew(qr)
        attention_scores = (torch.matmul(q, k.transpose(-2, -1)) + qr) / math.sqrt(self.d_att)

        return self.attention_post_processing(attention_scores, mask, output_mode, v)

    @staticmethod
    def _skew(x: Tensor):
        seq_len = x.shape[-2]
        emb_len = x.shape[-1]
        dst_len = (emb_len - 1) // 2

        interim = torch.nn.functional.pad(x, (0, 1))
        interim = interim.reshape((*x.shape[:-2], -1))
        interim = torch.nn.functional.pad(interim, (0, emb_len - seq_len))
        interim = interim.reshape((*x.shape[:-2], seq_len + 1, emb_len))
        interim = interim.narrow(-1, dst_len, dst_len)
        interim = interim.narrow(-2, 0, seq_len)

        return interim


class RelativeInformationAttention(ScaledDotProductAttention):
    """Relative Attention based on arbitrary information as introduced by Schoen et al. (2024)"""

    def __init__(self, d_att: int, dropout_rate: float, max_dist: int,
                 device: torch.device) -> None:
        super().__init__(d_att, dropout_rate, device)

        self.max_dist = max_dist

        self.relative_embedding = nn.Embedding(2 * max_dist + 1, d_att, device=device)

    def pass_pre(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 mask: torch.Tensor = None, output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY,
                 **kwargs) -> dict[str, torch.Tensor]:
        distances = (torch.arange(-self.max_dist, self.max_dist + 1, device=self.device) + self.max_dist).detach()
        distance_embeddings = self.relative_embedding(distances).unsqueeze(0)
        distance_embeddings = distance_embeddings.expand(*k.shape[:-2], *distance_embeddings.shape[1:])

        return {"distance_embeddings": distance_embeddings}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                q_info: torch.Tensor, k_info: torch.Tensor, distance_embeddings: torch.Tensor,
                mask: torch.Tensor = None,
                output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor]:
        qr = torch.matmul(q, distance_embeddings.transpose(-2, -1))
        qr = self.distribute_rel_info(qr, q_info, k_info)
        attention_scores = (torch.matmul(q, k.transpose(-2, -1)) + qr) / math.sqrt(self.d_att)

        return self.attention_post_processing(attention_scores, mask, output_mode, v)

    def distribute_rel_info(self, x: Tensor, q_info: Tensor, k_info):
        # Replace invalid distances with maximum distance
        q_info_adj = q_info.clone().detach()
        q_info_adj[q_info_adj == -1] = 2 * self.max_dist
        k_info_adj = k_info.clone().detach()
        k_info_adj[k_info_adj == -1] = 3 * self.max_dist

        # Calculate distances between each token in the source and target sequence
        distance_matrix = k_info_adj.unsqueeze(-2) - q_info_adj.unsqueeze(-1)
        distance_matrix = distance_matrix.clamp(min=-self.max_dist, max=self.max_dist)
        distance_matrix += self.max_dist

        # Reshape distance matrix
        while distance_matrix.dim() < x.dim():
            distance_matrix = distance_matrix.unsqueeze(-3)
            distance_matrix = distance_matrix.expand(*x.shape[:-2], *distance_matrix.shape[-2:])

        # Gather values based on distance matrix indices
        gathered_values = torch.gather(x, dim=-1, index=distance_matrix)

        return gathered_values


# Block-Sparse Attention

class BaseBlockSparseAttention(BaseSingleHeadAttention, ABC):
    def __init__(self, block_size: int, device: torch.device) -> None:
        super().__init__(device)

        self.block_size = block_size


class TritonBlockSparseAttention(BaseBlockSparseAttention):
    """Block-Sparse Attention based on Triton supporting block sizes of {16, 32, 64, 128} and arbitrary layouts"""

    def __init__(self, d_att: int, block_size: int, dropout_rate: float, device: torch.device) -> None:
        super().__init__(block_size, device)

        self.d_att = d_att
        self.dropout_rate = dropout_rate

        self.dropout = Dropout(dropout_rate)

        self.toh = TritonOperationHolder(block_size, device)

    def prime(self, sparsity_layout: torch.Tensor, modes: list[str] = None):
        self.toh.prime(sparsity_layout, modes)

    def reprime(self, other: "TritonBlockSparseAttention"):
        self.toh.trans_prime(other.toh)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None, info_tensor: torch.Tensor = None,
                output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor]:
        # Unsqueeze since Triton can only handle [head_size, batch_size, seq_len, d_model] tensors
        t_q = self.toh.shape_triton(q)
        t_k = self.toh.shape_triton(k)
        t_v = self.toh.shape_triton(v)
        t_mask = None if mask is None else self.toh.shape_triton(mask)

        attention_shape = [*q.shape[:-1], k.size(-2)]
        output_shape = v.shape

        # Matrix multiplication
        attention_scores = self.toh.matmul_sdd(t_q, t_k.transpose(-2, -1))
        if info_tensor is not None:
            t_info_tensor = TritonBlockSparseAttention.shape_triton(info_tensor)
            attention_scores += self.toh.sparsify_tensor(t_info_tensor)
        attention_scores /= math.sqrt(self.d_att)

        # Check if only scores are needed, if so, apply non-sparse mask
        if output_mode == AttentionOutputMode.SCORES_ONLY:
            attention_scores = self.toh.desparsify_tensor(attention_scores, [*t_q.shape[:-1], t_k.size(-2)])
            if t_mask is not None:
                attention_scores = attention_scores.masked_fill(t_mask == 1, torch.finfo(attention_scores.dtype).min)
            return TritonBlockSparseAttention.unshape_triton(attention_scores)

        # Apply sparse mask
        if t_mask is not None:
            attention_scores = attention_scores.masked_fill(self.toh.sparsify_tensor(t_mask) == 1,
                                                            torch.finfo(attention_scores.dtype).min)

        # Apply softmax
        if attention_scores.size(-3) != 0:
            attention_weights = self.toh.softmax(attention_scores)
        else:
            attention_weights = torch.zeros(*t_q.shape[:-1], t_k.size(-2), device=self.device)
            attention_weights = self.toh.sparsify_tensor(attention_weights)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        if output_mode == AttentionOutputMode.WEIGHTS_ONLY:
            return self.toh.unshape_triton(
                self.toh.desparsify_tensor(attention_weights, [*q.shape[:-1], k.size(-2)]), attention_shape)

        output = self.toh.matmul_dsd(attention_weights, t_v)

        if output_mode == AttentionOutputMode.OUTPUT_ONLY:
            return self.toh.unshape_triton(output, output_shape)
        elif output_mode == AttentionOutputMode.OUTPUT_AND_WEIGHTS:
            return (self.toh.unshape_triton(output, output_shape),
                    self.toh.unshape_triton(
                        self.toh.desparsify_tensor(attention_weights, [*q.shape[:-1], k.size(-2)]), attention_shape))
        else:
            raise NotImplementedError


# Multi Head Attention

class BaseMultiHeadAttention(nn.Module, ABC):

    def __init__(self, single_head_attention: BaseSingleHeadAttention, device: torch.device) -> None:
        super().__init__()

        self.single_head_attention = single_head_attention
        self.device = device

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None,
                output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY,
                transform_inputs: bool = True) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MultiHeadAttention(BaseMultiHeadAttention):
    """Multi-Head Attention as introduced by Vaswani et al. (2017)"""

    def __init__(self, d_model: int, d_att: int, n_heads: int, dropout_rate: float,
                 single_head_attention: BaseSingleHeadAttention,
                 device: torch.device, transform_inputs=True) -> None:
        super().__init__(single_head_attention, device)

        self.d_model = d_model
        self.d_att = d_att
        self.n_heads = n_heads
        self.d_head_v = d_model // n_heads
        self.d_head_k = d_att // n_heads
        self.dropout_rate = dropout_rate
        self.transform_values = transform_inputs

        if self.transform_values:
            self.q_net = nn.Linear(d_model, d_att, device=device)
            self.k_net = nn.Linear(d_model, d_att, device=device)
            self.v_net = nn.Linear(d_model, d_model, device=device)
            self.o_net = nn.Linear(d_model, d_model, device=device)
        self.dropout = Dropout(dropout_rate)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None,
                output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY,
                **kwargs) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor]:
        interim_q = q
        interim_k = k
        interim_v = v

        if self.transform_values:
            interim_q = self.q_net(q)
            interim_k = self.k_net(k)
            interim_v = self.v_net(v)

        interim_q = MultiHeadAttention.split_heads(interim_q, self.n_heads, self.d_head_k)
        interim_k = MultiHeadAttention.split_heads(interim_k, self.n_heads, self.d_head_k)
        interim_v = MultiHeadAttention.split_heads(interim_v, self.n_heads, self.d_head_v)

        prepass_args = self.single_head_attention.pass_pre(q, k, v, mask, output_mode, **kwargs)
        for key in prepass_args.keys():
            prepass_args[key] = MultiHeadAttention.split_heads(prepass_args[key], self.n_heads, self.d_head_k)

        extend_args = {key: kwargs[key] for key in kwargs.keys() if key not in prepass_args}
        for key in extend_args.keys():
            interim_subject = extend_args[key]
            interim_subject = interim_subject.unsqueeze(1)
            interim_subject = interim_subject.expand(
                (*interim_subject.shape[:1], self.n_heads, *interim_subject.shape[2:]))
            extend_args[key] = interim_subject

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.expand((*mask.shape[:1], self.n_heads, *mask.shape[2:]))

        single_head_output = self.single_head_attention(interim_q, interim_k, interim_v, **prepass_args, **extend_args,
                                                        mask=mask, output_mode=output_mode)

        if output_mode == AttentionOutputMode.WEIGHTS_ONLY:
            attention_weights = single_head_output
            return attention_weights
        elif output_mode == AttentionOutputMode.OUTPUT_ONLY:
            interim = single_head_output
        elif output_mode == AttentionOutputMode.OUTPUT_AND_WEIGHTS:
            interim, attention_weights = single_head_output
        else:
            raise NotImplementedError

        interim = MultiHeadAttention.merge_heads(interim, self.d_model)
        if self.transform_values:
            interim = self.o_net(interim)
        interim = self.dropout(interim)

        if output_mode == AttentionOutputMode.OUTPUT_ONLY:
            return interim
        elif output_mode == AttentionOutputMode.OUTPUT_AND_WEIGHTS:
            return interim, attention_weights
        else:
            raise NotImplementedError

    @staticmethod
    def split_heads(x: Tensor, num_heads: int, dim_head: int) -> Tensor:
        interim = x.reshape(*x.shape[:-1], num_heads, dim_head)
        return interim.permute((0, -2, *range(1, interim.dim() - 2), -1))

    @staticmethod
    def merge_heads(x: Tensor, d_model: int) -> Tensor:
        interim = x.permute((0, *range(2, x.dim() - 1), 1, -1))
        return interim.reshape((*interim.shape[:-2], d_model))


# Meta Attention

class BaseMetaAttention(nn.Module, ABC):

    def __init__(self, device: torch.device) -> None:
        super().__init__()

        self.device = device

    def forward(self, q: torch.Tensor, attention_weights: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None, output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY,
                **kwargs) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class StackedMetaAttention(BaseMetaAttention):

    def __init__(self, max_len: int, d_model: int, d_att: int, n_inputs: int, dropout_rate: float,
                 single_head_attention: BaseSingleHeadAttention, device: torch.device) -> None:
        super().__init__(device)

        self.max_len = max_len
        self.d_model = d_model
        self.d_att = d_att
        self.n_inputs = n_inputs
        self.single_head_attention = single_head_attention

        self.q_net = nn.Linear(d_model, d_att, device=device)
        self.s_net = nn.Linear(self.n_inputs * self.max_len, d_att, device=device)
        self.v_net = nn.Linear(d_model, d_model, device=device)
        self.mha = MultiHeadAttention(d_model, d_att, n_inputs, dropout_rate, single_head_attention, device,
                                      transform_inputs=False)

    def forward(self, q: torch.Tensor, attention_weights: list[torch.Tensor], v: torch.Tensor,
                mask: torch.Tensor = None, output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY,
                **kwargs) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor]:
        stacked_attention_weights = torch.cat(attention_weights, dim=-1)

        interim_q = self.q_net(q)
        interim_s = self.s_net(stacked_attention_weights)
        interim_v = self.v_net(v)

        interim = self.mha(interim_q, interim_s, interim_v, mask=mask, output_mode=output_mode, **kwargs)

        return interim


class DistributionMetaAttention(BaseMetaAttention):

    def __init__(self, max_len: int, d_model: int, d_att: int, n_inputs: int, n_heads: int, dropout_rate: float,
                 single_head_attention: BaseSingleHeadAttention, device: torch.device) -> None:
        super().__init__(device)

        self.max_len = max_len
        self.d_model = d_model
        self.d_att = d_att
        self.n_inputs = n_inputs
        self.n_heads = n_heads
        self.single_head_attention = single_head_attention

        if self.single_head_attention.dropout_rate != 0:
            LOGGER.warning("Single head attention dropout rate is not 0 which may lead to unexpected behaviour")

        self.q_net = nn.Linear(d_model, d_att, device=device)
        self.s_net = nn.Linear(self.max_len * self.max_len, d_att, device=device)
        self.v_net = nn.Linear(d_model, d_model, device=device)
        self.mha = MultiHeadAttention(d_model, d_att, n_heads, dropout_rate, single_head_attention, device)
        self.dropout = Dropout(dropout_rate)

    def forward(self, q: torch.Tensor, attention_weights: list[torch.Tensor], v: torch.Tensor,
                mask: torch.Tensor = None, output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY,
                **kwargs) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor]:
        # Linearly transform input tensors
        interim_q = self.q_net(q)
        interim_v = self.v_net(v)
        interim_w = [torch.flatten(w, start_dim=-2) for w in attention_weights]
        interim_w = [self.s_net(w) for w in interim_w]
        interim_w = torch.stack(interim_w, dim=-2)

        # Attention-over-attention
        interim = self.single_head_attention(interim_q, interim_w, interim_v, mask=mask,
                                             output_mode=AttentionOutputMode.WEIGHTS_ONLY, **kwargs)

        # Apply weights to attention scores
        interim_weights = torch.stack(attention_weights, dim=-3)
        interim = interim.transpose(-1, -2)
        interim = interim.unsqueeze(-1)
        interim = interim.expand(*interim.shape[:-1], interim_weights.shape[-1])
        interim = torch.mul(interim, interim_weights)

        attention_weights = torch.sum(interim, dim=-3)
        attention_weights = self.dropout(attention_weights)

        if output_mode == AttentionOutputMode.WEIGHTS_ONLY:
            return attention_weights

        output = torch.matmul(attention_weights, v)

        if output_mode == AttentionOutputMode.OUTPUT_ONLY:
            return output
        elif output_mode == AttentionOutputMode.OUTPUT_AND_WEIGHTS:
            return output, attention_weights
        else:
            raise NotImplementedError


class FullContextDistributionMetaAttention(BaseMetaAttention):

    def __init__(self, max_len: int, d_model: int, d_att: int, n_inputs: int, n_heads: int, dropout_rate: float,
                 single_head_attention: BaseSingleHeadAttention, device: torch.device) -> None:
        super().__init__(device)

        self.max_len = max_len
        self.d_model = d_model
        self.d_att = d_att
        self.n_inputs = n_inputs
        self.n_heads = n_heads
        self.single_head_attention = single_head_attention

        if self.single_head_attention.dropout_rate != 0:
            LOGGER.warning("Single head attention dropout rate is not 0 which may lead to unexpected behaviour")

        self.q_net = nn.Linear(d_model, d_att, device=device)
        self.s_net = nn.Linear(self.max_len * self.max_len, self.max_len * d_att, device=device)
        self.v_net = nn.Linear(d_model, d_model, device=device)
        self.mha = MultiHeadAttention(d_model, d_att, n_heads, dropout_rate, single_head_attention, device)
        self.softmax = nn.Softmax(dim=-3)
        self.dropout = Dropout(dropout_rate)

    def forward(self, q: torch.Tensor, attention_weights: list[torch.Tensor], v: torch.Tensor,
                mask: torch.Tensor = None, output_mode: AttentionOutputMode = AttentionOutputMode.OUTPUT_ONLY,
                **kwargs) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor]:
        # Linearly transform input tensors
        interim_q = self.q_net(q)
        interim_v = self.v_net(v)
        interim_w = [torch.flatten(w, start_dim=-2) for w in attention_weights]
        interim_w = [self.s_net(s) for s in interim_w]
        interim_w = [torch.reshape(s, (*s.shape[:-1], self.max_len, self.d_att)) for s in interim_w]
        interim_w = torch.cat(interim_w, dim=-2)

        # Attention-over-attention
        interim = self.single_head_attention(interim_q, interim_w, interim_v, mask=mask,
                                             output_mode=AttentionOutputMode.WEIGHTS_ONLY, **kwargs)
        interim = interim.chunk(self.n_inputs, dim=-1)
        interim = torch.stack(interim, dim=-3)
        interim = self.softmax(interim)

        # Apply weights to attention scores
        interim_weights = torch.stack(attention_weights, dim=-3)
        interim_weights = torch.mul(interim_weights, interim)

        attention_weights = torch.sum(interim_weights, dim=-3)
        attention_weights = self.dropout(attention_weights)

        if output_mode == AttentionOutputMode.WEIGHTS_ONLY:
            return attention_weights

        output = torch.matmul(attention_weights, v)

        if output_mode == AttentionOutputMode.OUTPUT_ONLY:
            return output
        elif output_mode == AttentionOutputMode.OUTPUT_AND_WEIGHTS:
            return output, attention_weights
        else:
            raise NotImplementedError


def _extract_values(interim, output_mode: AttentionOutputMode):
    output = None
    weights = None

    if output_mode == AttentionOutputMode.OUTPUT_ONLY:
        output = interim
    elif output_mode == AttentionOutputMode.WEIGHTS_ONLY:
        weights = interim
    elif output_mode == AttentionOutputMode.OUTPUT_AND_WEIGHTS:
        output, weights = interim
    else:
        raise NotImplementedError

    return output, weights
