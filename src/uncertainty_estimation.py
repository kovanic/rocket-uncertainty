import torch
import torch.nn.functional as F
from typing import (
    List, Tuple, Dict, Any,
    Optional, Sequence, Callable
)

def estimate_uncertainty_on_ensemble(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Calculate uncertainties metrics on ensemble:
    :param logits: outputs of classifier before normalization,
                   torch.Tensor[N x C x E] N=number of observations, C=number of classes, E=ensemble size
    :param labels: true classes, torch.Tensor[N]

    Note: the same logic is applied to all uncertainty metrics:
          the more is value of the metric, the more is uncertainty.

    :return: dict with the following metrics:
        1. Predictive entropy
        2. Mutual information
        3. Margin
        4. Standard deviation of predicted probabilities averaged over ensemble
        5. Standard deviation of averaged predicted probabilities
        6. 1 - maximum of predicted probabilities
    """
    probas = F.softmax(logits, dim=1)
    probas_ensemble = probas.mean(dim=2)

    predictive_entropy = torch.sum(-probas_ensemble * torch.log(probas_ensemble + 1e-30), dim=1)
    expected_entropy = torch.sum(-probas * F.log_softmax(logits, dim=1), dim=1).mean(dim=1)
    mutual_information = predictive_entropy - expected_entropy

    top2values = torch.topk(probas_ensemble, 2, dim=1).values

    margin = top2values[:, 0] - top2values[:, 1]
    std_averaged_over_ensemble = probas.std(dim=2).mean(dim=1)
    std = probas_ensemble.std(dim=1)

    maxprob = top2values[:, 0]

    return {
        "predictive_entropy": predictive_entropy,
        "mutual_information": mutual_information,
        "margin": -margin,
        "std_averaged_over_ensemble": std_averaged_over_ensemble,
        "std": -std,
        "maxprob": 1 - maxprob
    }


def ideal_rejection_curve(preds: torch.Tensor, labels: torch.Tensor, rejection_rates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Provide the best possible rejection curve (IRC) for a given model preictions.

    :param preds: predictions, size(N)
    :param labels: true labels, size(N)
    :param rejection_rates: rejection_rates, size(N)
    :return: IRC values
    """
    preds_total = len(labels)
    preds_correct = (preds == labels).sum()
    preds_incorrect = (preds != labels).sum()
    irc_values = preds_correct / (preds_total - torch.minimum(rejection_rates * preds_total, preds_incorrect))
    return irc_values


def AUC(rejection_rates: torch.Tensor, irc_values: torch.Tensor, rc_values: torch.Tensor) -> float:
    """Calcualte area under rejection curve, normalized by the area under IRC.
    """
    auc = torch.trapz(rc_values - irc_values[0], rejection_rates)
    norm_auc = torch.trapz(irc_values - irc_values[0], rejection_rates)
    return (auc / norm_auc).item()


def sort_data_by_metric(
    metric: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort preds and labels by descending uncertainty metric.
    :param metric: uncertainty metric according to which preds and labels will be sorted
                   (Expected logic: The more is uncertainty metric, the more is uncertainty),
                   torch.tensor[N]
    :param preds: model predictions, toаch.tensor[N]
    :param labels: ground truth labels, torch.tensor[N]
    :return: a tuple of
        - torch.Tensor[N] of predictions, sorted according to metric
        - torch.Tensor[N] of labels, sorted according to metric
    """
    sorted_metric_idx = torch.argsort(metric, descending=True)
    return preds[sorted_metric_idx], labels[sorted_metric_idx]


def accuracy(labels: torch.Tensor, preds: torch.Tensor) -> float:
    """Accuracy calculation.
    :param labels: true classes, torch.tensor[N]
    :param predictions: predictions, torch.tensor[N]
    """
    return (labels == preds).float().mean().item()


def rejection_curve(
    uncertainty_proxy: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    rejection_rates: torch.Tensor,
    scoring_func: Callable,
) -> List:
    """Reject points from preds and labels based on uncertainty estimate of choice.
    :param uncertainty_proxy: tesnor of unceratinty estimations, torch.tensor[N]
    :param preds: model label predictions or predicted class probabilities, torch.tensor[N]
    :param labels: ground truth labels, torch.tensor[N]
    :param rejection_rates: rejection rates to use, torch.tensor[N]
    :param scoring_func: scoring function that takes labels and predictions or probabilities (in that order)
    :return: tensor of scores calculated for each rejection rate
    """
    preds_sorted, labels_sorted = sort_data_by_metric(uncertainty_proxy, preds, labels)

    scores = []
    for s in torch.ceil(rejection_rates * len(preds)).long():
        scores.append(scoring_func(labels_sorted[s:], preds_sorted[s:]))
    return torch.Tensor(scores)
