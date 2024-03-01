"""
Contains the contrastive loss function used for discussion embedding pre-training.

In practice, the contrastive loss is applied on the global discussion embeddings, 
where discussions from the same community are considered as positive pairs, 
and discussions from different communities are considered as negative pairs.

This loss is currently not used in HatefulDiscussions, but rather work in 
progress experiments
"""

from fairseq.dataclass.configs import FairseqDataclass

import torch
from torch.nn import functional as F
from fairseq import metrics
from src.models import GraphormerModel
from fairseq.criterions import FairseqCriterion, register_criterion
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class GraphContrastiveLossConfig(FairseqDataclass):
    """
    Tunable hyperparameters for the contrastive loss function.
    Each flag here is accessible from the command line.
    """

    soft_negative_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight to associate to soft negative pairs in the contrastive loss. Flag is exclusive against adaptive_soft_negative_weight"
        },
    )
    adaptive_soft_negative_weight: bool = field(
        default=True,
        metadata={
            "help": "Whether to adapt the soft negative weight based on the number of positive pairs and negative pairs. Flag is exclusive against soft_negative_weight"
        },
    )
    multiplication_scale: float = field(
        default=20.0,
        metadata={
            "help": "Multiplcation factor to scale the similarity matrix"
            "higher values will result in less strict contrastive loss."
            "(1 = strict match, 20 = less strict)"
        },
    )


@register_criterion("contrastive_loss", dataclass=GraphContrastiveLossConfig)
class GraphContrastiveLoss(FairseqCriterion):
    """Contrastive loss for discussion embeddings."""

    def __init__(
        self,
        task,
        soft_negative_weight: float,
        multiplication_scale: float,
        adaptive_soft_negative_weight: bool,
    ) -> None:
        super().__init__(task)
        self.soft_negative_weight = soft_negative_weight
        self.multiplication_scale = multiplication_scale
        self.adaptive_soft_negative_weight = adaptive_soft_negative_weight

        if (
            self.adaptive_soft_negative_weight
            and self.soft_negative_weight != 0
        ):
            raise ValueError(
                "adaptive_soft_negative_weight and soft_negative_weight are mutually exclusive"
            )

    def forward(
        self,
        model: GraphormerModel,
        sample: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        reduce=True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
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

        num_comments = sample["net_input"]["batched_data"]["x"].shape[1]

        # invalid samples
        if "batched_data" not in sample["net_input"]:
            raise ValueError(
                f"Invalid sample, missing batched_data: {sample['net_input']}"
            )

        # forward pass batched data, keeping discussion embeddings
        _, embeddings = model(**sample["net_input"])

        # compute similarity matrix for contrastive loss
        normalized_A = F.normalize(embeddings, p=2, dim=1)
        normalized_B = F.normalize(embeddings, p=2, dim=1)
        sim = (
            torch.mm(normalized_A, normalized_B.transpose(0, 1))
            * self.multiplication_scale
        ).float()  # scaling factor

        # Targets is an array of int labels, discussions sharing the same label
        # are from the same community/topic
        targets = sample["net_input"]["batched_data"]["y"].float()
        # Format y into a n x n matrix where n is the number of graphs and each
        # row has 1 for the correct label and 0 for the rest
        target_matrix = targets.unsqueeze(1).eq(targets).half()

        # Same as targets, but for hard negatives
        hard_targets = sample["net_input"]["batched_data"]["hard_y"].float()
        hard_target_metrix = hard_targets.unsqueeze(1).eq(targets).half()

        soft_labels = torch.logical_and(
            target_matrix.eq(0), hard_target_metrix.eq(0)
        )
        if self.adaptive_soft_negative_weight:
            # soft_negs are proportionally weighted to the number of hard_negs and hard_pos
            num_hard_labels = (
                torch.logical_or(target_matrix.eq(1), hard_target_metrix.eq(1))
            ).sum(dim=1)
            extra_weight = (num_hard_labels / soft_labels.sum(dim=1)) * 2
        else:
            extra_weight = self.soft_negative_weight

        # compute loss weights. Hard labels are given 1 weight, soft labels
        # are given extra_weight
        soft_matrix = torch.where(soft_labels, extra_weight, 1).cuda()
        # Since we do intra-modality contrastive loss, remove diagonal from
        # loss matrix. We don't want to include itself in the loss

        # having to set this to cuda() is gross, there should be a better way
        soft_matrix = torch.where(
            torch.eye(soft_matrix.size(0)).eq(1).cuda(), 0, soft_matrix
        )

        # compute sample metrics
        with torch.no_grad():
            pred_labels = F.sigmoid(sim).round()
            ncorrect = (pred_labels == targets).sum()
            num_positive_correct = torch.logical_and(
                pred_labels == targets, pred_labels == 1
            ).sum()
            total_positive = (targets == 1).sum()
            pred_positive = (pred_labels == 1).sum()

        # compute loss
        loss = F.binary_cross_entropy_with_logits(
            sim,
            target_matrix,
            weight=soft_matrix,
            reduction="sum" if reduce else "none",
        ).float()

        sim_count = sim.shape[0] * sim.shape[1]

        # logging output to aggregate in reduce_metrics
        logging_output = {
            "loss": loss.detach(),
            "sample_size": sim_count,
            "nsentences": sim_count,
            "ntokens": num_comments,
            "ncorrect": ncorrect,
            "positive_correct": num_positive_correct,
            "total_positive": total_positive,
            "pred_positive": pred_positive,
        }
        return loss, sim_count, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs: Dict[str, List[Any]]) -> None:
        """Aggregate logging outputs from data parallel training. Logs resulting metrics.

        Args:
            logging_outputs: list of logging outputs from each data parallel worker
        """
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            poscorrect = sum(
                log.get("positive_correct", 0) for log in logging_outputs
            )  # tp
            total_positive = sum(
                log.get("total_positive", 0) for log in logging_outputs
            )  # tp + fn
            pred_positive = sum(
                log.get("pred_positive", 0) for log in logging_outputs
            )  # tp + fp
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=2
            )
            metrics.log_scalar(
                "precision",
                100.0 * poscorrect / pred_positive,
                sample_size,
                round=2,  # tp / (tp + fp)
            )
            metrics.log_scalar(
                "recall",
                100.0 * poscorrect / total_positive,
                sample_size,
                round=2,  # tp / (tp + fn)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
