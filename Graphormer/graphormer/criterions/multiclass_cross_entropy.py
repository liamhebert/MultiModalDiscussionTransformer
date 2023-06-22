# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fairseq.dataclass.configs import FairseqDataclass

import torch
from torch.nn import functional
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn as nn
from dataclasses import dataclass, field

@dataclass
class GraphPredictionNodeCrossEntropyConfig(FairseqDataclass):
    positive_weight: float = field(
        default = 1.0,
        metadata={'help': "weight to associate to the positive class"}
    )
    negative_weight: float = field(
        default = 1.0,
        metadata={'help': "weight to associate to the negative class"}
    )



@register_criterion("node_cross_entropy", dataclass=GraphPredictionNodeCrossEntropyConfig)
class GraphPredictionNodeCrossEntropy(FairseqCriterion):
    """
    Implementation for the multi-class log loss used in graphormer model training.
    """
    
    def __init__(self, task, positive_weight, negative_weight) -> None:
        super().__init__(task)
        #self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.loss = nn.CrossEntropyLoss(reduction='sum', weight=torch.Tensor([negative_weight, positive_weight]).cuda())
        

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
        
        logits = model(**sample["net_input"])
        #print(sample["net_input"]["batched_data"]["x"].shape, logits.shape, sample["net_input"]["batched_data"]["y"].shape, sample["net_input"]['batched_data']['y_mask'].shape, sample["net_input"]['batched_data']['x_token_mask'].shape)
        #targets = model.get_targets(sample, [logits])[: logits.size(0)]
        targets = sample["net_input"]["batched_data"]["y"]
        target_mask = sample["net_input"]['batched_data']['y_mask']
        #mask = sample["net_input"]['batched_data']['x_token_mask']
        #logits = logits[target_mask[mask].flatten(), :]
        
        #targets = targets[target_mask]
        
        #logits = torch.flatten(logits, end_dim=-2)
        sample_size = len(logits)
        targets = torch.flatten(targets)
        with torch.no_grad():
            pred_labels = torch.argmax(functional.softmax(logits, dim=-1), dim=-1)
            ncorrect = (pred_labels == targets).sum()
            num_positive_correct = torch.logical_and(pred_labels == targets, pred_labels == 1).sum()
            total_positive = (targets == 1).sum()
            
            num_pred_positive = (pred_labels == 1).sum()
            
        #print(targets)
        loss = self.loss(
            logits, targets
        )

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
            "ncorrect": ncorrect,
            "num_positive_correct": num_positive_correct,
            "total_positive": total_positive,
            "num_pred_positive": num_pred_positive
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
            num_positive_correct = sum(log.get("num_positive_correct", 0) for log in logging_outputs)
            total_positive = sum(log.get("total_positive", 0) for log in logging_outputs)
            num_pred_positive = sum(log.get("num_pred_positive", 0) for log in logging_outputs)
            if total_positive == 0:
                recall = 0
            else:
                recall = num_positive_correct / total_positive
            if num_pred_positive == 0:
                precision = 0
            else:
                #print(num_pred_positive)
                precision = num_positive_correct / num_pred_positive
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
            metrics.log_scalar(
                "recall", recall, total_positive, round=3
            )
            metrics.log_scalar(
                "precision", precision, num_pred_positive, round=3
            )
            if precision == 0 and recall == 0:
                metrics.log_scalar(
                    "f1", 0, total_positive, round=3
                )
            else:
                metrics.log_scalar(
                    "f1", 2 * ((precision * recall) / (precision + recall)), total_positive, round=3
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

@register_criterion("node_binary_cross_entropy", dataclass=FairseqDataclass)
class GraphPredictionBinaryNodeCrossEntropy(FairseqCriterion):
    """
    Implementation for the multi-class log loss used in graphormer model training.
    """
    def __init__(self, task) -> None:
        super().__init__(task)
        self.loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.Tensor([1.5]))
        
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

        logits = model(**sample["net_input"])
        #targets = model.get_targets(sample, [logits])[: logits.size(0)]
        targets = sample["net_input"]["batched_data"]["y"]
        target_mask = sample["net_input"]['batched_data']['y_mask']
        mask = sample["net_input"]['batched_data']['x_token_mask']
        #logits = logits[target_mask[mask].flatten(), :]
        
        targets = targets
        
        logits = torch.flatten(logits, start_dim=1).float()
        
        
        sample_size = len(logits)
       
        targets = targets.unsqueeze(-1).float()
        
        ncorrect = (torch.where(torch.sigmoid(logits) < 0.5, 0, 1) == targets).sum()
        #print(targets)
        loss = self.loss(
            logits, targets
        )
             
        # print(loss / sample_size)

        logging_output = {
            "loss": loss.data,
            "sample_size": 1,
            "nsentences": sample_size,
            "ntokens": natoms,
            "ncorrect": ncorrect,
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
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


# @register_criterion("multiclass_cross_entropy_with_flag", dataclass=FairseqDataclass)
# class GraphPredictionMulticlassCrossEntropyWithFlag(GraphPredictionMulticlassCrossEntropy):
#     """
#     Implementation for the multi-class log loss used in graphormer model training.
#     """

#     def forward(self, model, sample, reduce=True):
#         """Compute the loss for the given sample.

#         Returns a tuple with three elements:
#         1) the loss
#         2) the sample size, which is used as the denominator for the gradient
#         3) logging outputs to display while training
#         """
#         sample_size = sample["nsamples"]
#         perturb = sample.get("perturb", None)

#         with torch.no_grad():
#             natoms = sample["net_input"]["batched_data"]["x"].shape[1]

#         logits = model(**sample["net_input"], perturb=perturb)
#         logits = logits[:, 0, :]
#         targets = model.get_targets(sample, [logits])[: logits.size(0)]
#         ncorrect = (torch.argmax(logits, dim=-1).reshape(-1) == targets.reshape(-1)).sum()

#         loss = functional.cross_entropy(
#             logits, targets.reshape(-1), reduction="sum"
#         )

#         logging_output = {
#             "loss": loss.data,
#             "sample_size": sample_size,
#             "nsentences": sample_size,
#             "ntokens": natoms,
#             "ncorrect": ncorrect,
#         }
#         return loss, sample_size, logging_output
