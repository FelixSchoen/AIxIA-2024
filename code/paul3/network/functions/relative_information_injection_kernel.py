import pydevd
import torch
import triton
import triton.language as tl
from torch import nn, Tensor

from paul3.utils.triton_operation_holder import get_cuda_autotune_config, TritonOperationHolder
from paul3.utils.utility import compact, decompact


@triton.jit
def _relative_information_injection_kernel_forward(t_q, t_emb, t_info, t_output,
                                                   m_q, n_q: tl.constexpr,
                                                   m_emb, n_emb: tl.constexpr,
                                                   m_info, n_info: tl.constexpr,
                                                   b_out, m_out, n_out: tl.constexpr,
                                                   idxs_batch_sparsity, idxs_row_sparsity,
                                                   stride_q_b, stride_q_m, stride_q_n,
                                                   stride_emb_b, stride_emb_m, stride_emb_n,
                                                   stride_info_b, stride_info_m, stride_info_n,
                                                   stride_output_b, stride_output_m, stride_output_n,
                                                   block_size_sparsity: tl.constexpr,
                                                   BLOCK_SIZE_TRITON: tl.constexpr):
    # Get program IDs
    pid_batch = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get non-zero indices in order to check which sparsity block we are in
    spar_batch = tl.load(idxs_batch_sparsity + pid_batch)
    spar_row = tl.load(idxs_row_sparsity + pid_batch)

    # Iterate over entries of triton block
    for i_row in range(0, BLOCK_SIZE_TRITON):
        for i_col in range(0, BLOCK_SIZE_TRITON):
            # Value that is to be embedded (index of embedding matrix)
            info_index = ((pid_batch * stride_info_b) +
                          (pid_row * BLOCK_SIZE_TRITON * stride_info_m) +
                          (pid_col * BLOCK_SIZE_TRITON * stride_info_n) +
                          (i_row * stride_info_m) +
                          (i_col * stride_info_n))
            info_mask = (pid_row * BLOCK_SIZE_TRITON + i_row < m_info) & (pid_col * BLOCK_SIZE_TRITON + i_col < n_info)
            info_value = tl.load(t_info + info_index, mask=info_mask).to(tl.int32)

            # Query vector
            q_index = ((spar_batch * stride_q_b) +
                       (spar_row * block_size_sparsity * stride_q_m) +
                       (pid_row * BLOCK_SIZE_TRITON * stride_q_m) +
                       (i_row * stride_q_m))
            q_offsets = (tl.arange(0, n_q) * stride_q_n)
            q_mask = (spar_row * block_size_sparsity + pid_row * BLOCK_SIZE_TRITON + i_row < m_q)
            q_values = tl.load(t_q + q_index + q_offsets, mask=q_mask)

            # Embedding vector
            emb_index = ((spar_batch * stride_emb_b) +
                         (info_value * stride_emb_m))
            emb_offsets = (tl.arange(0, n_emb) * stride_emb_n)
            emb_mask = (info_value < m_emb)
            emb_values = tl.load(t_emb + emb_index + emb_offsets, mask=emb_mask)

            # Multiply q and emb, store in output matrix
            output_index = ((pid_batch * stride_output_b) +
                            (pid_row * BLOCK_SIZE_TRITON * stride_output_m) +
                            (pid_col * BLOCK_SIZE_TRITON * stride_output_n) +
                            (i_row * stride_output_m) +
                            (i_col * stride_output_n))
            output_mask = (pid_row * BLOCK_SIZE_TRITON + i_row < m_out) & (pid_col * BLOCK_SIZE_TRITON + i_col < n_out)
            final_mask = (output_index <
                          (b_out * stride_output_b + m_out * stride_output_m + n_out * stride_output_n))
            tl.store(t_output + output_index, tl.sum(q_values * emb_values), mask=output_mask & final_mask)


@triton.jit
def _relative_information_injection_kernel_backward_q(t_grad, t_emb, t_info, t_output,
                                                      m_grad, n_grad: tl.constexpr,
                                                      m_emb, n_emb: tl.constexpr,
                                                      m_info, n_info: tl.constexpr,
                                                      b_out, m_out, n_out: tl.constexpr,
                                                      idxs_batch_sparsity, idxs_row_sparsity,
                                                      stride_grad_b, stride_grad_m, stride_grad_n: tl.constexpr,
                                                      stride_emb_b, stride_emb_m, stride_emb_n: tl.constexpr,
                                                      stride_info_b, stride_info_m, stride_info_n: tl.constexpr,
                                                      stride_output_b, stride_output_m, stride_output_n: tl.constexpr,
                                                      block_size_sparsity: tl.constexpr,
                                                      BLOCK_SIZE_TRITON: tl.constexpr):
    # Get program IDs
    pid_batch = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get non-zero indices in order to check which sparsity block we are in
    spar_batch = tl.load(idxs_batch_sparsity + pid_batch)
    spar_row = tl.load(idxs_row_sparsity + pid_batch)

    # Iterate over rows of triton block
    for i_row in range(0, BLOCK_SIZE_TRITON):
        # Indices of embeddings to multiply with gradients
        info_index = ((pid_batch * stride_info_b) +
                      (pid_row * BLOCK_SIZE_TRITON * stride_info_m) +
                      (pid_col * BLOCK_SIZE_TRITON * stride_info_n) +
                      (i_row * stride_info_m))
        info_offsets = (tl.arange(0, BLOCK_SIZE_TRITON) * stride_info_n)
        info_mask = (pid_row * BLOCK_SIZE_TRITON + i_row < m_info) & (info_offsets < n_info * stride_info_n)
        info_values = tl.load(t_info + info_index + info_offsets, mask=info_mask).to(tl.int32)

        # Gradient values
        grad_index = ((pid_batch * stride_grad_b) +
                      (pid_row * BLOCK_SIZE_TRITON * stride_grad_m) +
                      (pid_col * BLOCK_SIZE_TRITON * stride_grad_n) +
                      (i_row * stride_grad_m))
        grad_offsets = (tl.arange(0, BLOCK_SIZE_TRITON) * stride_grad_n)
        grad_mask = (pid_row * BLOCK_SIZE_TRITON + i_row < m_grad) & (
                grad_offsets + pid_col * BLOCK_SIZE_TRITON < n_grad * stride_grad_n)
        grad_values = tl.load(t_grad + grad_index + grad_offsets, mask=grad_mask)

        # Iterate over columns of output gradients
        for i_dim in range(0, n_emb):
            emb_index = ((spar_batch * stride_emb_b) +
                         (i_dim * stride_emb_n))
            emb_offsets = (info_values * stride_emb_m)
            emb_mask = (i_dim < n_emb) & (emb_offsets < m_emb * stride_emb_m)
            emb_values = tl.load(t_emb + emb_index + emb_offsets, mask=emb_mask)

            # Compute gradients of q for current column in block
            output_index = ((spar_batch * stride_output_b) +
                            (spar_row * block_size_sparsity * stride_output_m) +
                            (pid_row * BLOCK_SIZE_TRITON * stride_output_m) +
                            (i_row * stride_output_m) +
                            (i_dim * stride_output_n))
            output_mask = (spar_row * block_size_sparsity + pid_row * BLOCK_SIZE_TRITON + i_row < m_out) & (
                    i_dim < n_out)
            final_mask = (output_index <
                          (b_out * stride_output_b + m_out * stride_output_m + n_out * stride_output_n))
            tl.atomic_add(t_output + output_index, tl.sum(grad_values * emb_values), mask=output_mask & final_mask)


@triton.jit
def _relative_information_injection_kernel_backward_emb(t_grad, t_q, t_info, t_output,
                                                        m_grad, n_grad: tl.constexpr,
                                                        m_q, n_q: tl.constexpr,
                                                        m_info, n_info: tl.constexpr,
                                                        b_out, m_out, n_out: tl.constexpr,
                                                        idxs_batch_sparsity, idxs_row_sparsity,
                                                        stride_grad_b, stride_grad_m, stride_grad_n,
                                                        stride_q_b, stride_q_m, stride_q_n,
                                                        stride_info_b, stride_info_m, stride_info_n,
                                                        stride_output_b, stride_output_m, stride_output_n,
                                                        block_size_sparsity: tl.constexpr,
                                                        BLOCK_SIZE_TRITON: tl.constexpr):
    # Get program IDs
    pid_batch = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get non-zero indices in order to check which sparsity block we are in
    spar_batch = tl.load(idxs_batch_sparsity + pid_batch)
    spar_row = tl.load(idxs_row_sparsity + pid_batch)

    # Iterate over columns of triton block
    for i_col in range(0, BLOCK_SIZE_TRITON):
        # Indices of where to store gradients
        info_index = ((pid_batch * stride_info_b) +
                      (pid_row * BLOCK_SIZE_TRITON * stride_info_m) +
                      (pid_col * BLOCK_SIZE_TRITON * stride_info_n) +
                      (i_col * stride_info_n))
        info_offsets = (tl.arange(0, BLOCK_SIZE_TRITON) * stride_info_m)
        info_mask = ((pid_row * BLOCK_SIZE_TRITON * stride_info_m + info_offsets < m_info * stride_info_m) &
                     (i_col < n_info))
        info_values = tl.load(t_info + info_index + info_offsets, mask=info_mask).to(tl.int32)

        grad_index = ((pid_batch * stride_grad_b) +
                      (pid_row * BLOCK_SIZE_TRITON * stride_grad_m) +
                      (pid_col * BLOCK_SIZE_TRITON * stride_grad_n) +
                      (i_col * stride_grad_n))
        grad_offsets = (tl.arange(0, BLOCK_SIZE_TRITON) * stride_grad_m)
        grad_mask = ((pid_row * BLOCK_SIZE_TRITON * stride_grad_m + grad_offsets < m_grad * stride_grad_m) &
                     (i_col < n_grad))
        grad_values = tl.load(t_grad + grad_index + grad_offsets, mask=grad_mask)

        # Iterate over dimensions of q
        for i_dim in range(0, n_q):
            # Load q_n values of current column
            q_index = ((spar_batch * stride_q_b) +
                       (spar_row * block_size_sparsity * stride_q_m) +
                       (pid_row * BLOCK_SIZE_TRITON * stride_q_m) +
                       (i_dim * stride_q_n))
            q_offsets = (tl.arange(0, BLOCK_SIZE_TRITON) * stride_q_m)
            q_mask = (spar_row * block_size_sparsity * stride_q_m +
                      pid_row * BLOCK_SIZE_TRITON * stride_q_m +
                      q_offsets < m_q * stride_q_m) & (i_dim < n_q)
            q_values = tl.load(t_q + q_index + q_offsets, mask=q_mask)

            # Compute and store gradients at positions specified by info
            output_index = ((spar_batch * stride_output_b) +
                            (i_dim * stride_output_n))
            output_offsets = (info_values * stride_output_m)
            output_mask = (info_values < m_out) & (i_dim < n_out)
            final_mask = (output_index + output_offsets <
                          b_out * stride_output_b +
                          m_out * stride_output_m +
                          n_out * stride_output_n)
            tl.atomic_add(t_output + output_index + output_offsets, grad_values * q_values,
                          mask=output_mask & final_mask)


class _RelativeInformationInjection(torch.autograd.Function):
    bst = 32

    @staticmethod
    def forward(ctx, q, emb, info, sparsity_layout, block_size_sparsity):
        # Reshape input tensors
        t_q = compact(q)
        t_emb = compact(emb)
        t_info = compact(info)
        t_sparsity_layout = compact(sparsity_layout)
        output = torch.zeros_like(t_info, dtype=torch.float)

        # Obtain indices of positive entries
        idxs_batch_sparsity, idxs_row_sparsity, idxs_col_sparsity = t_sparsity_layout.nonzero(as_tuple=True)

        # Extract dimensions
        b_info, m_info, n_info = t_info.shape
        b_q, m_q, n_q = t_q.shape
        b_emb, m_emb, n_emb = t_emb.shape
        b_out, m_out, n_out = output.shape

        # Create operation grid
        triton_grid = lambda meta: [b_info,
                                    triton.cdiv(m_info, meta["BLOCK_SIZE_TRITON"]),
                                    triton.cdiv(n_info, meta["BLOCK_SIZE_TRITON"])]

        ctx.save_for_backward(t_q, t_emb, t_info, t_sparsity_layout)
        ctx.size_q = q.size()
        ctx.size_emb = emb.size()
        ctx.block_size_sparsity = block_size_sparsity
        ctx.triton_grid = triton_grid

        # Apply kernel
        _relative_information_injection_kernel_forward[triton_grid](t_q, t_emb, t_info, output,
                                                                    m_q, n_q,
                                                                    m_emb, n_emb,
                                                                    m_info, n_info,
                                                                    b_out, m_out, n_out,
                                                                    idxs_batch_sparsity, idxs_row_sparsity,
                                                                    t_q.stride(0), t_q.stride(1), t_q.stride(2),
                                                                    t_emb.stride(0), t_emb.stride(1), t_emb.stride(2),
                                                                    t_info.stride(0), t_info.stride(1),
                                                                    t_info.stride(2),
                                                                    output.stride(0), output.stride(1),
                                                                    output.stride(2),
                                                                    block_size_sparsity,
                                                                    BLOCK_SIZE_TRITON=_RelativeInformationInjection.bst)

        # Undo sparsification and compacting
        output = decompact(output, info.size())

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO Debug
        if pydevd.GetGlobalDebugger() is not None:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        prt_gradient = grad_output.contiguous()
        t_q, t_emb, t_info, t_sparsity_layout = ctx.saved_tensors
        size_q = ctx.size_q
        size_emb = ctx.size_emb
        block_size_sparsity = ctx.block_size_sparsity
        triton_grid = ctx.triton_grid

        # Extract dimensions
        b_grad, m_grad, n_grad = prt_gradient.shape
        b_q, m_q, n_q = t_q.shape
        b_emb, m_emb, n_emb = t_emb.shape
        b_info, m_info, n_info = t_info.shape

        # Obtain indices of positive entries
        idxs_batch_sparsity, idxs_row_sparsity, idxs_col_sparsity = t_sparsity_layout.nonzero(as_tuple=True)

        # Gradients for q
        grad_q = torch.zeros_like(t_q, dtype=torch.float)
        b_out, m_out, n_out = grad_q.shape
        _relative_information_injection_kernel_backward_q[triton_grid](prt_gradient, t_emb, t_info, grad_q,
                                                                       m_grad, n_grad,
                                                                       m_emb, n_emb,
                                                                       m_info, n_info,
                                                                       b_out, m_out, n_out,
                                                                       idxs_batch_sparsity, idxs_row_sparsity,
                                                                       prt_gradient.stride(0), prt_gradient.stride(1),
                                                                       prt_gradient.stride(2),
                                                                       t_emb.stride(0), t_emb.stride(1),
                                                                       t_emb.stride(2),
                                                                       t_info.stride(0), t_info.stride(1),
                                                                       t_info.stride(2),
                                                                       grad_q.stride(0), grad_q.stride(1),
                                                                       grad_q.stride(2),
                                                                       block_size_sparsity,
                                                                       BLOCK_SIZE_TRITON=_RelativeInformationInjection.bst)
        grad_q = decompact(grad_q, size_q)

        # Gradients for emb
        grad_emb = torch.zeros_like(t_emb, dtype=torch.float)
        b_out, m_out, n_out = grad_emb.shape
        _relative_information_injection_kernel_backward_emb[triton_grid](prt_gradient, t_q, t_info, grad_emb,
                                                                         m_grad, n_grad,
                                                                         m_q, n_q,
                                                                         m_info, n_info,
                                                                         b_out, m_out, n_out,
                                                                         idxs_batch_sparsity, idxs_row_sparsity,
                                                                         prt_gradient.stride(0), prt_gradient.stride(1),
                                                                         prt_gradient.stride(2),
                                                                         t_q.stride(0), t_q.stride(1),
                                                                         t_q.stride(2),
                                                                         t_info.stride(0), t_info.stride(1),
                                                                         t_info.stride(2),
                                                                         grad_emb.stride(0), grad_emb.stride(1),
                                                                         grad_emb.stride(2),
                                                                         block_size_sparsity,
                                                                         BLOCK_SIZE_TRITON=_RelativeInformationInjection.bst)
        grad_emb = decompact(grad_emb, size_emb)

        return grad_q, grad_emb, None, None, None, None


class RelativeInformationInjection(nn.Module):

    # TODO
    def __init__(self, device: torch.device, block_size: int = 16) -> None:
        super().__init__()

        self.block_size = block_size
        self.device = device

        self.toh = TritonOperationHolder(block_size, device)

        self.default_sparsity_layout = None

    def prime(self, sparsity_layout: Tensor, modes: list[str] = None):
        self.toh.prime(sparsity_layout, modes)

    def trans_prime(self, other: "RelativeInformationInjection"):
        self.toh.trans_prime(other.toh)

    def forward(self, q, emb, info, max_dist, sparsity_layout=None):
        # Create default sparsity layout if none is provided
        if sparsity_layout is None:
            if self.default_sparsity_layout is None:
                assert info.shape[-2] % self.block_size == 0 and info.shape[-1] % self.block_size == 0, \
                    "Block size not compatible with info shape"

                self.default_sparsity_layout = torch.ones(
                    (torch.prod(torch.tensor(info.shape[:-2], device=self.device)),
                     info.shape[-2] // self.block_size, info.shape[-1] // self.block_size),
                    dtype=torch.int, device=self.device)
                self.prime(self.default_sparsity_layout)
            else:
                assert self.default_sparsity_layout.shape == (
                    *info.shape[:-1], info.shape[-2] // self.block_size, info.shape[-1] // self.block_size), \
                    "Default sparsity layout does no longer cover provided info shape"

            sparsity_layout = self.default_sparsity_layout

        sparsity_info = self.toh.sparsify_tensor(info)

        interim = _RelativeInformationInjection.apply(q, emb, sparsity_info + max_dist, sparsity_layout,
                                                      self.block_size)

        interim = self.toh.desparsify_tensor(interim, info.size())
        interim = interim.reshape(info.shape)

        return interim
