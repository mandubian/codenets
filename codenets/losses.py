from __future__ import annotations
from torch import nn

import torch
from pyhocon import ConfigTree
from loguru import logger
from typing import Tuple


class LossAndSimilarityScore(nn.Module):
    def forward(  # type: ignore
        self, x1: torch.Tensor, x2: torch.Tensor, ground_similarity: torch.Tensor, code_lang_weights: torch.Tensor
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
        self,
        query_embeddings: torch.Tensor,
        code_embeddings: torch.Tensor,
        ground_similarity: torch.Tensor,
        code_lang_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_sims = cosine_similarities(query_embeddings, code_embeddings)  # B x B
        similarity_scores = cos_sims
        z = torch.zeros_like(cos_sims).fill_diagonal_(float("-Inf"))
        best_wrongs, _ = torch.max(torch.relu(cos_sims + z), dim=-1)  # B x 1
        m = torch.tensor(self.margin).to(self.device) - cos_sims.diagonal() + best_wrongs  # B x 1
        per_sample_losses = torch.relu(m)  # B x1

        per_sample_losses = per_sample_losses * code_lang_weights

        return per_sample_losses, similarity_scores  # B x 1, B x 1


class SoftmaxCrossEntropyLossAndSimilarityScore(LossAndSimilarityScore):
    def __init__(self, device: torch.device, margin: float = 1.0):
        super(SoftmaxCrossEntropyLossAndSimilarityScore, self).__init__()
        self.device = device
        self.margin = margin

    def forward(  # type: ignore
        self,
        query_embeddings: torch.Tensor,
        code_embeddings: torch.Tensor,
        ground_similarity: torch.Tensor,
        code_lang_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.mm(query_embeddings, code_embeddings.t())  # B x B
        similarity_scores = logits
        per_sample_losses = torch.nn.functional.cross_entropy(
            input=logits.to(self.device),
            target=torch.arange(0, code_embeddings.size()[0]).to(self.device),
            reduction="none",
        )

        per_sample_losses = per_sample_losses * code_lang_weights

        return per_sample_losses, similarity_scores


class LogSoftmaxLossAndSimilarityScore(LossAndSimilarityScore):
    def __init__(self, device: torch.device, margin: float = 1.0):
        super(LogSoftmaxLossAndSimilarityScore, self).__init__()
        self.device = device
        self.margin = margin

    def forward(  # type: ignore
        self,
        query_embeddings: torch.Tensor,
        code_embeddings: torch.Tensor,
        ground_similarity: torch.Tensor,
        code_lang_weights: torch.Tensor,
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
        per_sample_losses = torch.relu(m)

        per_sample_losses = per_sample_losses * code_lang_weights

        return per_sample_losses, similarity_scores


def load_loss_and_similarity_function(loss_config: ConfigTree, device: torch.device) -> LossAndSimilarityScore:
    if loss_config["type"] == "softmax_cross_entropy":
        logger.info("Initializing Sofmax Cross Entroopy Loss")
        return LogSoftmaxLossAndSimilarityScore(device, loss_config["margin"])
    elif loss_config["type"] == "cosine_similarity":
        logger.info("Initializing Cosine Similarity Loss")
        return CosineSimilarityScoreAndMarginLoss(device, loss_config["margin"])
    elif loss_config["type"] == "lambda_loss":
        logger.info("Initializing Lambda Loss")
        return LambdaLossAndSimilarityScore(device)
    else:
        raise ValueError("loss.type can be softmax_cross_entropy or ...")


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.0) - torch.pow(torch.abs(D[0, delta_idxs]), -1.0))
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lamdbaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.0) - torch.pow(D[:, None, :], -1.0)) * torch.abs(
        G[:, :, None] - G[:, None, :]
    )


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lamdbaRank_scheme(G, D)


def rankNet_scheme(G, D, *args):
    return 1.0


def rankNetWeightedByGTDiff_scheme(G, D, *args):
    return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed_scheme(G, D, *args):
    return torch.abs(torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))


# Directly copied from original repository AllRank under Apache 2 license
# https://github.com/allegro/allRank/tree/master/allrank
#
class LambdaLossAndSimilarityScore(LossAndSimilarityScore):
    def __init__(
        self,
        device: torch.device,
        weighing_scheme="ndcgLoss2_scheme",
        scheme_func=ndcgLoss2_scheme,
        k=None,
        eps=1e-10,
        sigma=1.0,
        mu=10.0,
        reduction="sum",
        reduction_log="binary",
    ):
        super(LambdaLossAndSimilarityScore, self).__init__()
        self.device = device
        self.weighing_scheme = weighing_scheme
        self.scheme_func = scheme_func
        self.k = k
        self.eps = eps
        self.sigma = sigma
        self.mu = mu
        self.reduction = reduction
        self.reduction_log = reduction_log

    def forward(  # type: ignore
        self,
        query_embeddings: torch.Tensor,
        code_embeddings: torch.Tensor,
        ground_similarity: torch.Tensor,
        code_lang_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # basic vector product
        y_pred = torch.mm(query_embeddings, code_embeddings.t())  # B x B
        # we are in the binary case
        y_true = torch.diag(ground_similarity)

        # keep them as similarity score (the highest the most similar)
        similarity_scores = y_pred

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        # logger.debug(f"y_pred {y_pred}")
        # logger.debug(f"y_pred_sorted {y_pred_sorted}")
        # logger.debug(f"y_true_sorted {y_true_sorted}")

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs).to(self.device)

        if self.weighing_scheme != "ndcgLoss1_scheme":
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=self.device)
        ndcg_at_k_mask[: self.k, : self.k] = 1

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.0)
        y_true_sorted.clamp_(min=0.0)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(self.device)
        D = torch.log2(1.0 + pos_idxs.float())[None, :]
        # in our binary case, this is pretty simple
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, : self.k], dim=-1).clamp(min=self.eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # logger.debug(f"D {D}")
        # logger.debug(f"G {G}")

        # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
        if self.weighing_scheme is None:
            weights = 1.0
        else:
            weights = self.scheme_func(G, D, self.mu, true_sorted_by_preds)  # type: ignore

        # logger.debug(f"weights {weights}")

        # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.0
        weighted_probas = (torch.sigmoid(self.sigma * scores_diffs).clamp(min=self.eps) ** weights).clamp(min=self.eps)

        # logger.debug(f"weighted_probas {weighted_probas}")

        if self.reduction_log == "natural":
            losses = torch.log(weighted_probas)
        elif self.reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            raise ValueError("Reduction logarithm base can be either natural or binary")

        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        if self.reduction == "sum":
            loss = -torch.sum(masked_losses)
        elif self.reduction == "mean":
            loss = -torch.mean(masked_losses)
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss, similarity_scores


# def lambda_loss(
#     y_pred,
#     y_true,
#     device,
#     eps=1e-10,
#     padded_value_indicator=-1,
#     weighing_scheme=None,
#     k=None,
#     sigma=1.0,
#     mu=10.0,
#     reduction="sum",
#     reduction_log="binary",
# ):
#     """
#     Compute LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
#     Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
#     :param y_pred: predictions from the model, shape [batch_size, slate_length]
#     :param y_true: ground truth labels, shape [batch_size, slate_length]
#     :param eps: epsilon value, used for numerical stability
#     :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
#     :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
#     :param k: rank at which the loss is truncated
#     :param sigma: score difference weight used in the sigmoid function
#     :param mu: optional weight used in NDCGLoss2++ weighing scheme
#     :param reduction: losses reduction method, could be either a sum or a mean
#     :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
#     :return: loss value, a torch.Tensor
#     """
#     # device = y_pred.device
#     y_pred = y_pred.clone()
#     y_true = y_true.clone()

#     padded_mask = y_true == padded_value_indicator
#     y_pred[padded_mask] = float("-inf")
#     y_true[padded_mask] = float("-inf")

#     # Here we sort the true and predicted relevancy scores.
#     y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
#     y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

#     # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
#     true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
#     true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
#     padded_pairs_mask = torch.isfinite(true_diffs)

#     if weighing_scheme != "ndcgLoss1_scheme":
#         padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

#     ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
#     ndcg_at_k_mask[:k, :k] = 1

#     # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
#     true_sorted_by_preds.clamp_(min=0.0)
#     y_true_sorted.clamp_(min=0.0)

#     # Here we find the gains, discounts and ideal DCGs per slate.
#     pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
#     D = torch.log2(1.0 + pos_idxs.float())[None, :]
#     maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
#     G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

#     # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
#     if weighing_scheme is None:
#         weights = 1.0
#     else:
#         weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

#     # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
#     scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
#     scores_diffs[torch.isnan(scores_diffs)] = 0.0
#     weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
#     if reduction_log == "natural":
#         losses = torch.log(weighted_probas)
#     elif reduction_log == "binary":
#         losses = torch.log2(weighted_probas)
#     else:
#         raise ValueError("Reduction logarithm base can be either natural or binary")

#     masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
#     if reduction == "sum":
#         loss = -torch.sum(masked_losses)
#     elif reduction == "mean":
#         loss = -torch.mean(masked_losses)
#     else:
#         raise ValueError("Reduction method can be either sum or mean")

#     return loss
