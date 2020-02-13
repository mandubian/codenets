# from __future__ import annotations
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

    def forward(self, seq_outputs: torch.Tensor, tokens_mask: torch.Tensor) -> torch.Tensor:  # type: ignore
        pass


class MeanPooler(EmbeddingPooler):
    def __init__(self, input_size: int = 128, eps: float = 1e-8):
        super().__init__()
        self.dense = nn.Linear(input_size, 1, bias=False)
        self.activation = nn.Sigmoid()
        self.eps = eps

    def forward(self, seq_outputs: torch.Tensor, tokens_mask: torch.Tensor) -> torch.Tensor:  # type: ignore
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

    def forward(self, seq_outputs: torch.Tensor, tokens_mask: torch.Tensor) -> torch.Tensor:  # type: ignore
        token_weights = self.activation(self.dense(seq_outputs))  # B x T x 1
        token_weights = token_weights * tokens_mask.unsqueeze(dim=-1)  # B x T x 1
        # sum on the T dimension
        seq_weighted_sum = torch.sum(seq_outputs * token_weights, dim=1)  # B x D
        output = seq_weighted_sum / torch.sum(token_weights, dim=1).clamp(min=self.eps)
        return output

    # def save(self, output_dir: Union[Path, str]) -> bool:
    #     full_dir = Path(output_dir) / instance_full_classname(self)
    #     logger.debug(f"Saving MeanWeightedPooler to {full_dir}")
    #     os.makedirs(full_dir, exist_ok=True)
    #     torch.save(self.state_dict(), full_dir / "state_dict.pth")
    #     return True

    # @classmethod
    # def load(cls, restore_dir: Union[Path, str], *model_args, **kwargs) -> MeanWeightedPooler:
    #     full_dir = Path(restore_dir) / full_classname(cls)
    #     logger.debug(f"Loading MeanWeightedPooler from {full_dir}")
    #     state_dict = torch.load(full_dir / "state_dict.pth")
    #     pooler = MeanWeightedPooler()
    #     pooler.load_state_dict(state_dict)
    #     return pooler
