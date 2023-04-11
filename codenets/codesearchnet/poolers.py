from __future__ import annotations
from abc import abstractmethod
from torch import nn
import torch
from codenets.recordable import RecordableTorchModule


class EmbeddingPooler(RecordableTorchModule):
    """
    Compute pooler

    Args:
        seq_outputs (torch.tensor): [B x T x D] (B is batch, T is sequence size, D is embedding size)
        tokens_mask (torch.tensor): [B x T]
    
    Returns:
        tensor: [B x D]
    """

    @abstractmethod
    def forward(self, seq_outputs: torch.Tensor, tokens_mask: torch.Tensor) -> torch.Tensor:
        pass


class MeanPooler(EmbeddingPooler):
    def __init__(self, input_size: int = 128, eps: float = 1e-8):
        super().__init__()
        self.dense = nn.Linear(input_size, 1, bias=False)
        self.activation = nn.Sigmoid()
        self.eps = eps

    def forward(self, seq_outputs: torch.Tensor, tokens_mask: torch.Tensor) -> torch.Tensor:
        # TO TEST
        lg = torch.sum(tokens_mask, dim=-1)
        mask = tokens_mask.unsqueeze(dim=-1)
        seq_outputs_masked = seq_outputs * mask
        seq_outputs_sum = torch.sum(seq_outputs_masked, dim=-1)
        output = seq_outputs_sum / lg.unsqueeze(dim=-1).clamp(self.eps)
        return output


class MeanWeightedPooler(EmbeddingPooler):
    def __init__(self, input_size: int = 512, eps: float = 1e-8):  # default params required for module construction
        super().__init__()
        self.dense = nn.Linear(input_size, 1, bias=False)
        self.activation = nn.Sigmoid()
        self.eps = eps

    def forward(self, seq_outputs: torch.Tensor, tokens_mask: torch.Tensor) -> torch.Tensor:
        token_weights = self.activation(self.dense(seq_outputs))  # B x T x 1
        token_weights = token_weights * tokens_mask.unsqueeze(dim=-1)  # B x T x 1
        # sum on the T dimension
        seq_weighted_sum = torch.sum(seq_outputs * token_weights, dim=1)  # B x D
        output = seq_weighted_sum / torch.sum(token_weights, dim=1).clamp(min=self.eps)
        return output
