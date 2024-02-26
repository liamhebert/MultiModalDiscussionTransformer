"""Collator functions to merge data samples into a batch."""

import torch
from typing import List, Dict, Any


def pad_1d_unsqueeze(
    x: torch.Tensor, padlen: int, pad_value: int = 0, shift=True
):
    if (
        pad_value == 0 and shift
    ):  # to avoid existing 0 values being treated as padding
        x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_full([padlen], fill_value=pad_value, dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(
    x: torch.Tensor, padlen: int, pad_value: int = 0, shift=True
):
    if (
        pad_value == 0 and shift
    ):  # to avoid existing 0 values being treated as padding
        x = x + 1
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_full([padlen, xdim], fill_value=pad_value, dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x: torch.Tensor, padlen: int):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(
            float("-inf")
        )
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x: torch.Tensor, padlen: int):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x: torch.Tensor, padlen: int):
    x = x + 1  # to avoid existing 0 values being treated as padding
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def collator(items: List[List[torch.Tensor]], spatial_pos_max=10):
    """Collate function to merge data samples of various sizes into a batch.

    Individual data samples are comprised of the following attributes:
    - idxs: (int) list of unique indices from 0 to batch_size for each item
    - attn_biases: (List[float]) list of attention biases values for each node in the graph
    - spatial_poses: (List[int]) list of spatial indexes for each node in the graph.
        Used to fetch spatial position embeddings
    - in_degrees: (List[int]) list of the in-degree for each node in the graph. Used to fetch degree embeddings
    - x_text: (List[Dict[str, torch.Tensor]]) list of text input data for each node in the graph.
        Each input is a dictionary with pre-tokenized text tokens
    - x_image_indexes: (List[torch.Tensor]) list of boolean tensors indicating which nodes have images
    - x_images: (List[torch.Tensor]) list of image features for each node in the graph
    - distance: (List[torch.Tensor]) list of exact spatial distance between nodes, used to clip attention bias
    - ys: (List[torch.Tensor]) list of target labels for each node in the graph or a single label per graph

    Args:
        items: list of data samples
        spatial_pos_max: maximum spatial position to attend to when computing attention

    Returns:
        A collated patch of data samples where each item is padded to the largest
        size in the batch.

        Each output dictionary contains the following keys:
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
    """
    (
        idxs,
        attn_biases,
        spatial_poses,
        in_degrees,
        x_text,
        x_image_indexes,
        x_images,
        distance,
        ys,
    ) = zip(*items)

    # Clip attention bias to -inf for nodes that are farther then spatial_pos_max
    # setting to -inf sets the attention value to 0, removing them from inference
    for idx, _ in enumerate(attn_biases):
        # [1: , 1:] to avoid setting the global token to -inf
        attn_biases[idx][1:, 1:][distance[idx] >= spatial_pos_max] = float(
            "-inf"
        )
    max_node_num = max(i["input_ids"].size(0) for i in x_text)

    y = torch.cat(ys)

    x = {}
    # currently in the format of [tokens, size]
    for key in ["input_ids", "token_type_ids", "attention_mask"]:
        x[key] = torch.cat(
            [
                pad_2d_unsqueeze(a[key], max_node_num, pad_value=0, shift=False)
                for a in x_text
            ]
        )

    token_mask = ~x["input_ids"].eq(0).all(dim=2)

    # Remove placeholder images
    images = [x for x in x_images if not torch.all(x.eq(0))]
    if len(images) != 0:
        x_images = torch.cat(images)
    else:
        x_images = None
    x_image_indexes = torch.cat(
        [
            pad_1d_unsqueeze(z, -1, pad_value=False).squeeze(0)
            for z in x_image_indexes
        ]
    ).bool()

    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    ).int()
    in_degree = torch.cat(
        [pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]
    )

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # Since we are using undirected graph, in_degree == out_degree
        x_token_mask=token_mask,
        x=x["input_ids"],
        x_token_type_ids=x["token_type_ids"],
        x_attention_mask=x["attention_mask"],
        x_images=x_images,
        x_image_indexes=x_image_indexes,
        y=y,
    )
