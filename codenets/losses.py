from __future__ import annotations
from torch import nn

import torch
from pyhocon import ConfigTree

from typing import Tuple


class LossAndSimilarityScore(nn.Module):
    def forward(  # type: ignore
        self, x1: torch.Tensor, x2: torch.Tensor, ground_similarity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Loss and similarity score between x1 and x2
        
        Args:
            x1 (torch.Tensor): The first tensor [B x T x D].
            x2 (torch.Tensor): The second tensor [B x T x D].
            ground_similarityx2 (torch.Tensor): The second tensor [B x T x 1].

        Returns:
            Tuple[torch.tensor, torch.tensor]: loss [B x 1], similarity [B x 1]
        """
        pass


def cosine_similarities(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1 / w1.clamp(eps), (x2 / w2.clamp(eps)).t())


class CosineSimilarityScoreAndMarginLoss(LossAndSimilarityScore):
    def __init__(self, device: torch.device, margin: float = 1.0, eps: float = 1e-8):
        super(CosineSimilarityScoreAndMarginLoss, self).__init__()
        self.device = device
        self.margin = margin
        self.eps = eps

    def forward(  # type: ignore
        self, query_embeddings: torch.Tensor, code_embeddings: torch.Tensor, ground_similarity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_sims = cosine_similarities(query_embeddings, code_embeddings)  # B x B
        similarity_scores = cos_sims
        z = torch.zeros_like(cos_sims).fill_diagonal_(float("-Inf"))
        best_wrongs, _ = torch.max(torch.relu(cos_sims + z), dim=-1)  # B x 1
        m = torch.tensor(self.margin).to(self.device) - cos_sims.diagonal() + best_wrongs  # B x 1
        per_sample_losses = torch.relu(m)  # B x1

        return per_sample_losses, similarity_scores  # B x 1, B x 1


class SoftmaxCrossEntropyLossAndSimilarityScore(LossAndSimilarityScore):
    def __init__(self, device: torch.device, margin: float = 1.0):
        super(SoftmaxCrossEntropyLossAndSimilarityScore, self).__init__()
        self.device = device
        self.margin = margin

    def forward(  # type: ignore
        self, query_embeddings: torch.Tensor, code_embeddings: torch.Tensor, ground_similarity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.mm(query_embeddings, code_embeddings.t())  # B x B
        similarity_scores = logits
        per_sample_loss = torch.nn.functional.cross_entropy(
            input=logits.to(self.device),
            target=torch.arange(0, code_embeddings.size()[0]).to(self.device),
            reduction="none",
        )

        return per_sample_loss, similarity_scores


class LogSoftmaxLossAndSimilarityScore(LossAndSimilarityScore):
    def __init__(self, device: torch.device, margin: float = 1.0):
        super(LogSoftmaxLossAndSimilarityScore, self).__init__()
        self.device = device
        self.margin = margin

    def forward(  # type: ignore
        self, query_embeddings: torch.Tensor, code_embeddings: torch.Tensor, ground_similarity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # basic vector product
        logits = torch.mm(query_embeddings, code_embeddings.t())  # B x B
        # keep them as similarity score (the highest the most similar)
        similarity_scores = logits

        # make it probabilistic
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        # puts to -inf the scores of diagonal elements and gets the highest positive score among others
        z = torch.zeros_like(logprobs).fill_diagonal_(float("-inf"))
        best_wrongs, _ = torch.max(torch.relu(logprobs + z), dim=-1)  # B

        m = torch.tensor(self.margin).to(self.device) - logprobs.diagonal() + best_wrongs  # B
        per_sample_loss = torch.relu(m)

        return per_sample_loss, similarity_scores


def load_loss_and_similarity_function(loss_config: ConfigTree, device: torch.device) -> LossAndSimilarityScore:
    if loss_config["type"] == "softmax_cross_entropy":
        return LogSoftmaxLossAndSimilarityScore(device, loss_config["margin"])
    elif loss_config["type"] == "cosine_similarity":
        return CosineSimilarityScoreAndMarginLoss(device, loss_config["margin"])
    else:
        raise ValueError("loss.type can be softmax_cross_entropy or ...")
