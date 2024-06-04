"""
General wrappers for datasets and dataloaders

The organization of dataloading functions as follows:
- `pyg_datasets` contain dataloading functions for loading and pre-processing 
    PyG objects into general data objects we can use. Includes logic for splitting
    datasets into train, validation, and test sets
- `GraphormerDataset` is a wrapper for PyG datasets and provides a unified interface 
    for datasets. This returns individual samples from the dataset without batching
- `BatchedDataDataset` interface with `GraphormerDataset` and provide task-specific 
    collator functions to batch data for contrastive and node prediction tasks
- `EpochShuffleDataset` is a wrapper for `BatchedDataDataset` datasets to include 
    shuffling logic for each epoch

As such, the hierarchy of the dataloading process is:
`pyg_datasets` 
-> `GraphormerDataset` 
-> `BatchedDataDataset` 
-> `EpochShuffleDataset`
"""

import numpy as np
from fairseq.data import data_utils, FairseqDataset, BaseWrapperDataset
from abc import abstractmethod, ABC
from .collator import collator
import torch
from typing import Optional, Union
from torch_geometric.data import Data as PYGDataset
from .pyg_datasets import GraphormerPYGDataset


# TODO(liamhebert): We always use PyG datasets, so we may be able to merge
# `pyg_datasets` with `GraphormerDataset`
class GraphormerDataset:
    """
    Wrapper for PyG datasets to provide a unified interface for datasets.
    """

    def __init__(
        self,
        dataset: PYGDataset,
        dataset_source: Optional[str] = None,
        seed: int = 0,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
    ):
        super().__init__()
        if dataset is not None:
            if dataset_source == "pyg":
                self.dataset = GraphormerPYGDataset(
                    dataset,
                    seed=seed,
                    train_idx=train_idx,
                    valid_idx=valid_idx,
                    test_idx=test_idx,
                )
            else:
                raise ValueError("Customized dataset can only have source pyg")
        self.setup()

    def setup(self):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        # these three pointers are fed into BatchedDataDatasets
        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data


class BatchedDataDataset(ABC, FairseqDataset):
    """
    Dataloader to batch examples from `GraphormerDataset` into task specific
    batches with provided collator functions
    """

    def __init__(self, dataset: GraphormerPYGDataset, spatial_pos_max=1024):
        """
        Args:
            dataset: (GraphormerDataset) dataset to batch
            spatial_pos_max: (int) maximum spatial position to consider. Any
                node farther away then this distance is attention masked
        """
        super().__init__()
        self.dataset = dataset
        self.spatial_pos_max = spatial_pos_max

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def collater(self, samples):
        """
        Collate function to merge data samples of various sizes into a batch.

        Individual data samples are comprised of the following attributes:
        - idxs: (int) list of unique indices from 0 to batch_size for each item
        - attn_biases: (List[float]) list of attention biases values for each
            node in the graph
        - spatial_poses: (List[int]) list of spatial indexes for each node in the graph.
            Used to fetch spatial position embeddings
        - in_degrees: (List[int]) list of the in-degree for each node in the graph.
            Used to fetch degree embeddings
        - x_text: (List[Dict[str, torch.Tensor]]) list of text input data for
            each node in the graph. Each input is a dictionary with pre-tokenized
            text tokens
        - x_image_indexes: (List[torch.Tensor]) list of boolean tensors indicating
            which nodes have images
        - x_images: (List[torch.Tensor]) list of image features for each node
            in the graph
        - distance: (List[torch.Tensor]) list of exact spatial distance between
            nodes, used to clip attention bias
        - ys: (List[torch.Tensor]) list of target labels for each node in the
            graph or a single label per graph

        Args:
            items: list of data samples
            spatial_pos_max: maximum spatial pos

        Returns:
            A collated patch of data samples where each item is padded to the
            largest size in the batch.

            Each output dictionary must contains the following keys:
            - idx: (torch.Tensor) batched indices
            - attn_bias: (torch.Tensor) batched attention biases
            - spatial_pos: (torch.Tensor) batched spatial positions
            - in_degree: (torch.Tensor) batched in-degrees
            - out_degree: (torch.Tensor) batched out-degrees
            - x_token_mask: (torch.Tensor) batched token mask
            - x: (torch.Tensor) batched tokenized text input
            - x_token_type_ids: (torch.Tensor) batched token type ids
            - x_attention_mask: (torch.Tensor) batched attention mask
            - x_images: (torch.Tensor) batched image features
            - x_image_indexes: (torch.Tensor) batched image indexes
            - y: (torch.Tensor) batched target labels

            Additional features for task specific loss functions may also be
            added but they are not needed for general processing
        """
        ...


class ContrastiveBatchedDataDataset(BatchedDataDataset):
    def collater(self, samples):
        """
        Collate function specific to contrastive learning tasks.

        Each item follows the data structure as the general collate function but
        with each item having the following version specific attributes:
        - hard_y: (torch.Tensor) tensor of labels of the polar opposite communities
        - y: (torch.Tensor) tensor of labels of which topic the community belongs to
        """
        samples = [item for item in samples if item is not None]
        items = [
            (
                item.idx,
                item.attn_bias,
                item.spatial_pos,
                item.in_degree,
                item.x,
                item.x_image_index,
                item.x_images,
                item.distance,
                item.y,
            )
            for item in samples
        ]
        hard_ys = [item.hard_y for item in samples]
        hard_y = torch.cat(hard_ys)
        collated_output = collator(items, self.spatial_pos_max)
        collated_output["hard_y"] = hard_y
        return collated_output


class NodeBatchedDataDataset(BatchedDataDataset):
    def collater(self, samples):
        """
        Collate function specific to node prediction tasks.

        Each item follows the data structure as the general collate function but
        with each item having the following version specific attributes:
        - y_mask: (torch.Tensor) boolean tensor for each node in the graph indicating
            if it has a label
        - y: (torch.Tensor) tensor of labels for each node in the graph.
            If a node does not have a label, it is padded with 0
        """

        samples = [sample for sample in samples if sample is not None]
        items = [
            (
                item.idx,
                item.attn_bias,
                item.spatial_pos,
                item.in_degree,
                item.x,
                item.x_image_index,
                item.x_images,
                item.distance,
                item.y,
            )
            for item in samples
        ]
        y_masks = [item.y_mask for item in samples]
        y_mask = torch.cat(y_masks).bool()
        collated_output = collator(items, self.spatial_pos_max)
        collated_output["y_mask"] = y_mask
        return collated_output


class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset: FairseqDataset, num_samples, seed):
        super().__init__(dataset)
        self.num_samples = num_samples
        self.seed = seed
        self.set_epoch(1)

    def set_epoch(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.random.permutation(self.num_samples)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False
