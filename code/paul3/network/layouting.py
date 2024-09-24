import torch
from torch import Tensor


def build_mask_padding(state: Tensor, context: Tensor) -> Tensor:
    mask_x = torch.eq(state, 0).type(torch.float).unsqueeze(-1)
    mask_y = torch.eq(context, 0).type(torch.float).unsqueeze(-2)
    mask = torch.add(mask_x, mask_y).clamp(max=1)

    return mask


def build_mask_lookahead(size_x: int, size_y: int, device: torch.device) -> Tensor:
    mask = torch.triu(torch.ones(size_x, size_y, device=device), diagonal=1)

    return mask


def build_mask_padded_lookahead(state: Tensor, context: Tensor, device: torch.device) -> Tensor:
    mask_padding = build_mask_padding(state, context)
    mask_lookahead = build_mask_lookahead(state.size(-1), context.size(-1), device)
    mask_combined = torch.maximum(mask_padding, mask_lookahead)

    return mask_combined


def build_mask_constraints(x: Tensor, y: Tensor) -> Tensor:
    x_t = x.unsqueeze(-1)
    mask = build_mask_padding(x_t, y)

    return mask


# Blocksparse

def build_layout_sparsity(layout: torch.Tensor, block_size: int):
    """Creates a sparsity layout from a dense layout by applying a max pooling operation with a block size"""

    kernel_size = (block_size, block_size)
    stride = (block_size, block_size)
    pool_layer = torch.nn.MaxPool2d(kernel_size, stride)
    sparsity_layout = pool_layer(layout.to(torch.float))

    return sparsity_layout.to(torch.int)


# Museformer


def build_layout_summary(tokens: torch.Tensor, separator: int, self_attention: bool = False):
    """Builds a layout for the summary part of FC attention. Allows attention only for summary tokens over tokens of their own bar.

    Args:
        tokens: Tokens to create the layout for.
        separator: Which numerical token represents the summary token.
        self_attention: Whether summary tokens are allowed to attend to themselves.

    Returns: A layout for summary attention.

    """
    nonzero_indices = torch.eq(tokens, separator).nonzero()

    interim = torch.zeros((*tokens.size(), tokens.size(-1)), device=tokens.device)

    prev_tuple = torch.zeros_like(nonzero_indices)
    cur_c = 0

    for index, indices in enumerate(nonzero_indices):
        if not torch.equal(indices[:-1], prev_tuple):
            prev_tuple = indices[:-1]
            cur_c = 0

        interim[indices[0], indices[1], cur_c:indices[1]] = 1
        if self_attention:
            interim[indices[0], indices[1], indices[1]] = 2
        cur_c = indices[1] + 1

    return interim


def build_layout_aggregation(x: torch.Tensor, y: torch.Tensor, separator_token: int, structure_segments: list[int],
                             device: torch.device, separator_tokens_can_attend: bool = False):
    """Builds a layout for the aggregation part of FC attention. Allows attention only for music and summary tokens over music tokens of their structure bars, summary tokens otherwise.

    Args:
        x: State tensor.
        y: Context tensor.
        separator_token: Which numerical token represents the separator token.
        structure_segments: Which segments are considered structural, i.e., receive full attiontion.
        device: Which device to use.
        separator_tokens_can_attend: Whether separator tokens can attend to the same tokens their information tokens can attend to.

    Returns: A layout for aggregation attention.

    """

    # Get indices of separator token
    row_nonzero_indices = torch.eq(x, separator_token).nonzero()
    col_nonzero_indices = torch.eq(y, separator_token).nonzero()

    # Check if we can remove .item()
    row_index_list = [[] for _ in range(x.size(0))]
    for b, r in row_nonzero_indices:
        row_index_list[b].append(r.item())
    col_index_list = [[] for _ in range(y.size(0))]
    for b, c in col_nonzero_indices:
        col_index_list[b].append(c.item())

    # Add ghost separator token at the end of every row and column
    for b in range(x.size(0)):
        row_index_list[b].append(x.size(1))
    for b in range(y.size(0)):
        col_index_list[b].append(y.size(1))

    # Placeholder to insert mask into
    interim = torch.zeros(x.size(0), x.size(-1), y.size(-1), device=device)

    # Fill blocks between separator tokens with 1 if included in structure segments, otherwise fill structure separator with 2
    for b in range(x.size(0)):
        prev_r = 0

        for row_separator in row_index_list[b]:
            prev_c = 0
            index_row_separator = row_index_list[b].index(row_separator)
            is_row_ghost_separator = row_separator == x.size(-1)

            for col_separator in col_index_list[b]:
                index_col_separator = col_index_list[b].index(col_separator)
                distance_col_row = index_col_separator - index_row_separator
                is_col_ghost_separator = col_separator == y.size(-1)

                # row_separator+1 allows for summary tokens to attend to same tokens as their music tokens
                if distance_col_row in structure_segments:
                    interim[b, prev_r:row_separator, prev_c:col_separator] = 1
                    if separator_tokens_can_attend:
                        interim[b, row_separator, col_separator] = 1
                elif not is_col_ghost_separator:
                    interim[b, prev_r:row_separator, col_separator] = 2
                    if separator_tokens_can_attend:
                        interim[b, row_separator, col_separator] = 2

                prev_c = col_separator + 1

            prev_r = row_separator + 1

    return interim


# Utility

def create_shuffle_layout(bool_classification: torch.Tensor, expand_size: int):
    """Creates a layout for shuffling the input tensor such that all rows (or columns) given by `bool_classification` come before all other rows.

    Args:
        subject: Tensor to shuffle.
        bool_classification: Which rows (or columns) to move to the front.
        expand_size: The size to expand the layout to.

    Returns: A layout for shuffling the input tensor.

    """
    nonzero_indices = bool_classification.nonzero()

    shuffle_indices = torch.zeros_like(bool_classification, dtype=torch.int64)

    prev_tuple = torch.zeros_like(nonzero_indices[0])
    cur_r = 0
    cur_indices = []
    rem_indices = list(range(shuffle_indices.size(-1)))

    # Insert indices of non zero elements at the front, indices of other elements later on
    for index, indices in enumerate(nonzero_indices):
        if not torch.equal(indices[:-1], prev_tuple):
            to_insert = torch.tensor(rem_indices,
                                     dtype=shuffle_indices.dtype, device=shuffle_indices.device)
            shuffle_indices[(prev_tuple[0], *prev_tuple[1:-1]), cur_r:] = to_insert

            prev_tuple = indices[:-1]
            cur_r = 0
            cur_indices = []
            rem_indices = list(range(shuffle_indices.size(-1)))

        shuffle_indices[(indices[0], *indices[1:-1], cur_r)] = indices[-1]
        del rem_indices[indices[-1] - cur_r]
        cur_r += 1
        cur_indices.append(indices[-1])

    # Insert indices of last tuple
    if len(nonzero_indices) > 0:
        to_insert = torch.tensor(rem_indices,
                                 dtype=shuffle_indices.dtype, device=shuffle_indices.device)
        shuffle_indices[(prev_tuple[0], *prev_tuple[1:-1]), cur_r:] = to_insert

    shuffle_indices = shuffle_indices.unsqueeze(-1)
    shuffle_indices = shuffle_indices.expand(*shuffle_indices.shape[:-1], expand_size)

    return shuffle_indices


def build_fc_information(state: torch.Tensor, context: torch.Tensor, structure_segments: list[int],
                         separator_token: int, device: torch.device = None):
    """Builds all necessary layouts for the FC attention mechanism.

    Args:
        state: Tokens of current sequence.
        context: Tokens of sequence to attend to.
        structure_segments: Which segments to consider as structural.
        separator_token: Which tokens functions as the separator.
        device: Device to use.

    Returns: All necessary information tensors in the order (shuffle_layout_state, shuffle_layout_context,
    shuffle_layout_reverse, classification_context_shuffled, summary_layout, aggregation_layout).

    """
    if device is None:
        device = torch.device("cpu")

    # TODO Only works for state == context at the moment
    with (torch.no_grad()):
        state = state.to(device)
        context = context.to(device)

        # Create classification layout
        classification_state = (state == 4).clone().detach().to(device)

        # Create shuffle layouts
        shuffle_layout_state_tokens = create_shuffle_layout(classification_state, state.size(-1))
        shuffle_layout_state_values = create_shuffle_layout(classification_state, 1)
        shuffle_layout_state_reverse = torch.argsort(shuffle_layout_state_values, dim=-2)

        # Apply shuffling
        classification_state_shuffled = torch.gather(classification_state, -1,
                                                     shuffle_layout_state_tokens[..., 0])

        # Summary layouting
        summary_layout = build_layout_summary(state, 4)
        summary_layout_shuffled = torch.gather(
            torch.gather(summary_layout, -1, shuffle_layout_state_tokens.transpose(-1, -2)), -2,
            shuffle_layout_state_tokens)

        # Aggregation layouting
        padded_lookahead_mask = build_mask_padded_lookahead(state, state, device).to(
            state.dtype)
        aggregation_layout = build_layout_aggregation(state, state, separator_token,
                                                      structure_segments,
                                                      device)
        aggregation_layout_masked = aggregation_layout * (1 - padded_lookahead_mask)
        aggregation_layout_shuffled = torch.gather(
            torch.gather(aggregation_layout_masked, -1, shuffle_layout_state_tokens.transpose(-2, -1)), -2,
            shuffle_layout_state_tokens)
        aggregation_layout_shuffled_flat = torch.where(aggregation_layout_shuffled == 2, torch.tensor(1),
                                                       aggregation_layout_shuffled)

        return (shuffle_layout_state_values, shuffle_layout_state_values, shuffle_layout_state_reverse,
                classification_state_shuffled, summary_layout_shuffled, aggregation_layout_shuffled_flat)
