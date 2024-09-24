import torch
import triton.ops
from torch import Tensor, Size

from paul3.utils.utility import compact, decompact


class TritonOperationHolder:

    def __init__(self, block_size: int, device: torch.device):
        self.block_size = block_size
        self.device = device

        self._triton_matmul_sdd = None
        self._triton_matmul_dsd = None
        self._triton_matmul_dds = None
        self._triton_softmax = None

        self.primed = False

    def prime(self, sparsity_layout: torch.Tensor, modes: list[str] = None):
        if modes is None:
            modes = ["sdd", "dsd", "softmax"]

        assert sparsity_layout.dim() == 3, "Sparsity layout must be 3-dimensional"

        if "sdd" in modes:
            self._triton_matmul_sdd = triton.ops.blocksparse.matmul(sparsity_layout, self.block_size, "sdd",
                                                                    self.device)
        if "dsd" in modes:
            self._triton_matmul_dsd = triton.ops.blocksparse.matmul(sparsity_layout, self.block_size, "dsd",
                                                                    self.device)
        if "dds" in modes:
            self._triton_matmul_dds = triton.ops.blocksparse.matmul(sparsity_layout, self.block_size, "dds",
                                                                    self.device)
        if "softmax" in modes:
            self._triton_softmax = triton.ops.blocksparse.softmax(sparsity_layout, self.block_size, self.device)

        self.primed = True

    def trans_prime(self, other: "TritonOperationHolder"):
        self._triton_matmul_sdd = other._triton_matmul_sdd
        self._triton_matmul_dsd = other._triton_matmul_dsd
        self._triton_matmul_dds = other._triton_matmul_dds
        self._triton_softmax = other._triton_softmax
        self.primed = True

    # Blocksparse Functions

    def matmul_sdd(self, x: torch.Tensor, y: torch.Tensor):
        assert self.primed, "TritonOperationHolder must be primed before use"
        return self._triton_matmul_sdd(x, y)

    def matmul_dsd(self, x: torch.Tensor, y: torch.Tensor):
        assert self.primed, "TritonOperationHolder must be primed before use"
        return self._triton_matmul_dsd(x, y)

    def matmul_dds(self, x: torch.Tensor, y: torch.Tensor):
        assert self.primed, "TritonOperationHolder must be primed before use"
        return self._triton_matmul_dds(x, y)

    def softmax(self, x: torch.Tensor):
        assert self.primed, "TritonOperationHolder must be primed before use"
        return self._triton_softmax(x)

    # Utility Functions

    def sparsify_tensor(self, x: torch.Tensor):
        t_x = self.shape_triton(x)
        identity_matrix = torch.eye(t_x.size(-1), t_x.size(-2), device=self.device)

        while identity_matrix.dim() < t_x.dim():
            identity_matrix = identity_matrix.unsqueeze(0)
        identity_matrix = identity_matrix.expand(*t_x.shape[:-2], *identity_matrix.shape[-2:])

        t_x = self._triton_matmul_sdd(t_x, identity_matrix.to(t_x.dtype))
        t_x = t_x.squeeze(0)

        return t_x

    def desparsify_tensor(self, x: torch.Tensor, target_size):
        t_x = self.shape_triton(x)
        identity_matrix = torch.eye(target_size[-2], target_size[-1], device=self.device)

        while identity_matrix.dim() < t_x.dim():
            identity_matrix = identity_matrix.unsqueeze(0)
        identity_matrix = identity_matrix.expand(*target_size)
        identity_matrix = self.shape_triton(identity_matrix)

        return self._triton_matmul_dsd(t_x, identity_matrix.to(t_x.dtype))

    @staticmethod
    def shape_triton(x: Tensor) -> Tensor:
        t_x = compact(x)
        t_x = t_x.unsqueeze(0)
        return t_x

    @staticmethod
    def unshape_triton(x: Tensor, target_shape: Size) -> Tensor:
        t_x = x.squeeze(0)
        t_x = decompact(t_x, target_shape)
        return t_x

    @staticmethod
    def _force_dimensionality(x: Tensor):
        if x.dim() == 3:
            return x.unsqueeze(0)
        elif x.dim() == 4:
            return x
        else:
            raise NotImplementedError


# TODO If block size too large will overdraw
def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_TRITON': 16}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_TRITON': 32}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_TRITON': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_TRITON': 32}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_TRITON': 64}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_TRITON': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_TRITON': 64}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_TRITON': 128}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_TRITON': 128}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_TRITON': 128}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_TRITON': 256}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_TRITON': 256}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_TRITON': 256}, num_stages=5, num_warps=2),
    ]
