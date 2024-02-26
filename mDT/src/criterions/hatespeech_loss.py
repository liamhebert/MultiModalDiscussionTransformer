"""
This file contains the implementation of the loss function for the node classification task.
The loss function is a cross-entropy loss, with the option to weight the positive and negative classes.
The loss function also logs the accuracy, recall, precision, and F1 score.

We use this loss function in the hate speech detection task, where we want to classify each node 
in the graph as hate speech or not.
"""

from fairseq.dataclass.configs import FairseqDataclass

import torch
from torch.nn import functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn as nn
from dataclasses import dataclass, field
from src.models import GraphormerModel
from typing import Any, Dict, List, Tuple


@dataclass
class GraphPredictionNodeCrossEntropyConfig(FairseqDataclass):
    """
    Tunable hyperparameters for the node cross-entropy loss function.
    Each flag here is accessible from the command line.
    """

    positive_weight: float = field(
        default=1.0,
        metadata={"help": "Weight to associate to the positive class"},
    )
    negative_weight: float = field(
        default=1.0,
        metadata={"help": "Weight to associate to the negative class"},
    )


@register_criterion(
    "node_cross_entropy", dataclass=GraphPredictionNodeCrossEntropyConfig
)
class GraphPredictionNodeCrossEntropy(FairseqCriterion):
    """Node cross-entropy loss for graph node classification."""

    def __init__(
        self, task, positive_weight: float, negative_weight: float
    ) -> None:
        """
        Initialize the loss function with the given task and weights. Each hyper
        parameter is taken from GraphPredictionNodeCrossEntropyConfig.
        Args:
            task: the task object
            positive_weight: the weight to associate to the positive class
            negative_weight: the weight to associate to the negative class
        """
        super().__init__(task)
        self.weight = (
            torch.Tensor([negative_weight, positive_weight])
            .type(torch.HalfTensor)
            .cuda()
        )

    def forward(
        self, model: GraphormerModel, sample: Dict[str, Any], reduce=True
    ):
        """Compute the loss for the given sample.

        Args:
            model: pointer to the current model state
            sample: the training sample containing "net_input" as model input
            reduce: whether to reduce the loss across sentences

        Returns:
            a tuple with three elements:
            1) the loss
            2) the sample size, which is used as the denominator for the gradient
            3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        num_comments = sample["net_input"]["batched_data"]["x"].shape[1]

        # forward pass batched data, keeping comment logits
        comment_embeddings, _ = model(**sample["net_input"])

        # extract targets and comment embeddings which correspond to those targets
        # True, False values corresponding to Hate/Not Hate respectively
        targets = sample["net_input"]["batched_data"]["y"]
        # True if we have a label, False otherwise
        target_mask = sample["net_input"]["batched_data"]["y_mask"]

        logits = comment_embeddings[target_mask].type(torch.HalfTensor).cuda()

        sample_size = len(logits)
        targets = torch.flatten(targets).type(torch.LongTensor).cuda()

        # compute sample metrics
        with torch.no_grad():
            pred_labels = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            ncorrect = (pred_labels == targets).sum()
            num_positive_correct = torch.logical_and(
                pred_labels == targets, pred_labels == 1
            ).sum()
            total_positive = (targets == 1).sum()

            num_pred_positive = (pred_labels == 1).sum()

        # compute loss
        loss = F.cross_entropy(
            logits,
            targets,
            reduction="sum" if reduce else "none",
            weight=self.weight,
        )

        # logging output to aggregate in reduce_metrics
        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": num_comments,
            "ncorrect": ncorrect,
            "num_positive_correct": num_positive_correct,
            "total_positive": total_positive,
            "num_pred_positive": num_pred_positive,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs: Dict[str, List[Any]]) -> None:
        """Aggregate logging outputs from data parallel training.

        Args:
            logging_outputs: list of logging outputs from each data parallel worker
        """
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            num_positive_correct = sum(
                log.get("num_positive_correct", 0) for log in logging_outputs
            )
            total_positive = sum(
                log.get("total_positive", 0) for log in logging_outputs
            )
            num_pred_positive = sum(
                log.get("num_pred_positive", 0) for log in logging_outputs
            )
            # To prevent division by zero
            if total_positive == 0:
                recall = 0
            else:
                recall = num_positive_correct / total_positive
            if num_pred_positive == 0:
                precision = 0
            else:
                precision = num_positive_correct / num_pred_positive
            if precision == 0 and recall == 0:
                f1 = 0
            else:
                f1 = 2 * ((precision * recall) / (precision + recall))
            accuracy = ncorrect / nsentences

            metrics.log_scalar("accuracy", accuracy, nsentences, round=3)
            metrics.log_scalar("recall", recall, total_positive, round=3)
            metrics.log_scalar(
                "precision", precision, num_pred_positive, round=3
            )
            metrics.log_scalar("f1", f1, total_positive, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
