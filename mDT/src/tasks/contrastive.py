import logging

import torch.nn as nn
import contextlib
from dataclasses import dataclass
from omegaconf import open_dict, OmegaConf

from fairseq.tasks import register_task

from ..data.dataset import ContrastiveBatchedDataDataset, GraphormerPYGDataset

from .task import TaskConfig, Task

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveLearningConfig(TaskConfig):
    # TODO(liamhebert): Add contrastive learning specific hyperparameters
    ...


@register_task("contrastive_learning", dataclass=ContrastiveLearningConfig)
class ContrastiveLearningTask(Task):
    """
    Graph prediction (classification or regression) task.
    """

    def get_batched_dataset(self, dataset: GraphormerPYGDataset):
        return ContrastiveBatchedDataDataset(
            dataset,
            spatial_pos_max=self.cfg.spatial_pos_max,
        )

    def build_model(self, cfg: ContrastiveLearningConfig):
        from fairseq import models

        with (
            open_dict(cfg)
            if OmegaConf.is_config(cfg)
            else contextlib.ExitStack()
        ):
            cfg.max_nodes = self.cfg.max_nodes

        model = models.build_model(cfg, self)
        # model.node_encoder_stack = nn.ModuleList([nn.Identity()])

        return model
