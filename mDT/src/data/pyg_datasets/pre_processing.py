import torch
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(
        0, feature_num * offset, offset, dtype=torch.long
    )
    x = x + feature_offset
    return x


def cantor(x):
    x = sorted(x)
    return ((x[0] + x[1]) * (x[0] + x[1] + 1)) / 2 + x[0]


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x

    N = x["input_ids"].size(0)

    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    res = []
    for i in range(6):
        for k in range(6):
            res += [cantor([i, k])]

    res = list(set(res))
    mapping = {val: i for i, val in enumerate(res)}

    spatial = list(
        map(
            lambda x: list(
                map(
                    lambda k: (
                        mapping[cantor(k)]
                        if cantor(k) in mapping
                        else mapping[cantor([5, 5])]
                    ),
                    x,
                )
            ),
            item.distance_matrix,
        )
    )
    distance = list(
        map(lambda x: list(map(lambda k: sum(k), x)), item.distance_matrix)
    )
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float
    )  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.spatial_pos = torch.Tensor(spatial)
    item.distance = torch.Tensor(distance)
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph

    return item
