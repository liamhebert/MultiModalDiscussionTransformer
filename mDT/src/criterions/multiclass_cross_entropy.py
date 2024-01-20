
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
        default = 1.0,
        metadata={'help': "weight to associate to the positive class"}
    )
    negative_weight: float = field(
        default = 1.0,
        metadata={'help': "weight to associate to the negative class"}
    )



@register_criterion("node_cross_entropy", dataclass=GraphPredictionNodeCrossEntropyConfig)
class GraphPredictionNodeCrossEntropy(FairseqCriterion):

    
    def __init__(self, task, positive_weight, negative_weight) -> None:
        super().__init__(task)
        #self.loss = nn.CrossEntropyLoss(reduction='sum')
        #self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        

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
        if 'batched_data' not in sample["net_input"]:
            print(sample["net_input"])
            raise Exception
        embeddings = model(**sample["net_input"])
        #print(embeddings.shape)
        normalized_A = F.normalize(embeddings, p=2, dim=1)
        normalized_B = F.normalize(embeddings, p=2, dim=1)
        sim = (torch.mm(normalized_A, normalized_B.transpose(0, 1)) * 20).float() # scaling factor
        #print(sim.shape)

        targets = sample["net_input"]["batched_data"]["y"].float()
        y_weights = sample["net_input"]["batched_data"]["y_weight"].float()
        # print(targets)
        
        with torch.no_grad():
            pred_labels = F.sigmoid(sim).round()
            ncorrect = (pred_labels == targets).sum()
            num_positive_correct = torch.logical_and(pred_labels == targets, pred_labels == 1).sum()
            total_positive = (targets == 1).sum()
            pred_positive = (pred_labels == 1).sum()
            
            #num_pred_positive = (pred_labels == 1).sum()
            
        #print(targets)
        
        loss = F.binary_cross_entropy_with_logits(
            sim, targets, weight=y_weights, reduction='sum' if reduce else 'none'
        ).float()
        
       
   
        #print(loss / sample_size)
        # print(loss)
        # print('S', sim.shape)
        # print('S', sim)
        # print('T', targets.shape)
        # print('T', targets)
        # print('W', y_weights.shape)
        # print('W', y_weights)
        
        logging_output = {
            "loss": loss.detach(),
            "sample_size": sim.shape[0] * sim.shape[1],
            "nsentences": sim.shape[0] * sim.shape[1],
            "ntokens": natoms,
            "ncorrect": ncorrect,
            "positive_correct": num_positive_correct,
            "total_positive": total_positive,
            "pred_positive": pred_positive,
        }
        return loss, sim.shape[0] * sim.shape[1], logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        #print(loss_sum, loss_sum / sample_size, sample_size)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            poscorrect = sum(log.get("positive_correct", 0) for log in logging_outputs) #tp
            total_positive = sum(log.get("total_positive", 0) for log in logging_outputs) # tp + fn
            pred_positive = sum(log.get("pred_positive", 0) for log in logging_outputs) # tp + fp
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=2
            )
            metrics.log_scalar(
                "precision", 100.0 * poscorrect / pred_positive, sample_size, round=2 # tp / (tp + fp)
            )
            metrics.log_scalar(
                "recall", 100.0 * poscorrect / total_positive, sample_size, round=2 # tp / (tp + fn)
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
