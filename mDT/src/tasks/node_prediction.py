import logging

import contextlib
from dataclasses import dataclass
from omegaconf import open_dict, OmegaConf
from fairseq.tasks import register_task
import torch.nn as nn

from ..data.dataset import NodeBatchedDataDataset, GraphormerPYGDataset

from .task import Task, TaskConfig

logger = logging.getLogger(__name__)


@dataclass
class NodePredictionConfig(TaskConfig):
    # TODO(liamhebert): Add contrastive learning specific hyperparameters
    ...


@register_task("node_prediction", dataclass=NodePredictionConfig)
class NodePredictionTask(Task):
    """
    Node prediction (classification or regression) task.
    """

    def get_batched_dataset(self, dataset: GraphormerPYGDataset):
        return NodeBatchedDataDataset(
            dataset,
            spatial_pos_max=self.cfg.spatial_pos_max,
        )

    def build_model(self, cfg: NodePredictionConfig):
        from fairseq import models

        with (
            open_dict(cfg)
            if OmegaConf.is_config(cfg)
            else contextlib.ExitStack()
        ):
            cfg.max_nodes = self.cfg.max_nodes

        model = models.build_model(cfg, self)
        model.node_encoder_stack = nn.ModuleList(
            [
                model.encoder.graph_encoder.text_pooler,
                model.encoder.graph_encoder.text_dropout,
                nn.Linear(768, 2),
            ]
        )

        return model
