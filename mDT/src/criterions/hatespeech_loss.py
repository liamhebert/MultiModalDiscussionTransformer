# hate speech losses
from fairseq.dataclass.configs import FairseqDataclass

import torch
from torch.nn import functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class GraphPredictionNodeCrossEntropyConfig(FairseqDataclass):
    positive_weight: float = field(
        default=1.0, metadata={"help": "weight to associate to the positive class"}
    )
    negative_weight: float = field(
        default=1.0, metadata={"help": "weight to associate to the negative class"}
    )


@register_criterion(
    "node_cross_entropy", dataclass=GraphPredictionNodeCrossEntropyConfig
)
class GraphPredictionNodeCrossEntropy(FairseqCriterion):
    def __init__(self, task, positive_weight, negative_weight) -> None:
        super().__init__(task)
        self.weight = torch.Tensor([negative_weight, positive_weight]).cuda()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        out_all_subset, _ = model(**sample["net_input"])

        targets = sample["net_input"]["batched_data"]["y"]
        target_mask = sample["net_input"]["batched_data"]["y_mask"]
        logits = out_all_subset[target_mask]

        sample_size = len(logits)
        targets = torch.flatten(targets)
        with torch.no_grad():
            pred_labels = torch.argmax(functional.softmax(logits, dim=-1), dim=-1)
            ncorrect = (pred_labels == targets).sum()
            num_positive_correct = torch.logical_and(
                pred_labels == targets, pred_labels == 1
            ).sum()
            total_positive = (targets == 1).sum()

            num_pred_positive = (pred_labels == 1).sum()

        # print(targets)
        loss = F.cross_entropy(
            logits, targets, reduction="sum" if reduce else "none", weight=self.weight
        )

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
            "ncorrect": ncorrect,
            "num_positive_correct": num_positive_correct,
            "total_positive": total_positive,
            "num_pred_positive": num_pred_positive,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
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
            if total_positive == 0:
                recall = 0
            else:
                recall = num_positive_correct / total_positive
            if num_pred_positive == 0:
                precision = 0
            else:
                # print(num_pred_positive)
                precision = num_positive_correct / num_pred_positive
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
            metrics.log_scalar("recall", recall, total_positive, round=3)
            metrics.log_scalar("precision", precision, num_pred_positive, round=3)
            if precision == 0 and recall == 0:
                metrics.log_scalar("f1", 0, total_positive, round=3)
            else:
                metrics.log_scalar(
                    "f1",
                    2 * ((precision * recall) / (precision + recall)),
                    total_positive,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
