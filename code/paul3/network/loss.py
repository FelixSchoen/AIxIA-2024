import torch
from torch import nn

from paul3.network.layouting import build_mask_lookahead


class UnlikelihoodLoss(nn.Module):

    def __init__(self, ignore_index: int, label_smoothing_epsilon: float, unlikelihood_alpha: float,
                 unlikelihood_window_size: int, device: torch.device) -> None:
        super().__init__()

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing_epsilon
        self.unlikelihood_alpha = unlikelihood_alpha
        self.unlikelihood_window_size = unlikelihood_window_size
        self.device = device

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Compute cross entropy loss
        ce_lf = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, label_smoothing=self.label_smoothing)
        permuted_logits = logits.permute(0, -1, *range(1, logits.dim() - 1))
        ce_l = ce_lf(permuted_logits, labels)

        # Build candidates
        candidates = labels.unsqueeze(-1)
        unsqueezed_labels = candidates
        candidates = candidates.expand(*candidates.size()[:-2], candidates.size(-2), candidates.size(-2))
        candidates = candidates.transpose(-2, -1)
        mask = 1 - build_mask_lookahead(candidates.size(-1), candidates.size(-1), device=self.device)
        candidates = candidates * mask
        # Limit lookback
        if self.unlikelihood_window_size > 0:
            candidates = candidates * (
                    1 - torch.tril(torch.ones_like(candidates), diagonal=-self.unlikelihood_window_size))
        candidates = candidates.to(torch.int64)
        # Remove current target from candidates
        candidates = candidates.masked_fill(candidates == unsqueezed_labels, 0)
        candidate_targets = torch.zeros_like(logits).scatter_(-1, candidates, 1)
        # Remove padding values
        candidate_targets[..., 0] = 0

        inner_term = torch.clamp((1.0 - torch.nn.functional.softmax(logits, dim=-1)), min=1e-5)
        u_l = -torch.log(inner_term) * candidate_targets
        u_l = torch.log(1 + u_l.sum() / candidates.size(0))

        return ce_l + self.unlikelihood_alpha * u_l


class CrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index: int, label_smoothing_epsilon: float, device: torch.device) -> None:
        super().__init__()

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing_epsilon
        self.device = device

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Compute cross entropy loss
        ce_lf = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, label_smoothing=self.label_smoothing)
        permuted_logits = logits.permute(0, -1, *range(1, logits.dim() - 1))
        ce_l = ce_lf(permuted_logits, labels)

        return ce_l
