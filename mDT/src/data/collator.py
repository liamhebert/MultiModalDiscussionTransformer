

import torch

def pad_1d_custom_unsqueeze(x, padlen, pad_value):
    #x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_full([padlen], fill_value=pad_value, dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_custom_unsqueeze(x, padlen, pad_value):
    #x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_full([padlen, xdim], fill_value=pad_value, dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=10):
    items = [item for item in items if item is not None]
    items = [
        (
            item.idx,
            item.attn_bias,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.y,
            item.hard_y,
            item.x_image_index,
            item.x_images,
            item.distance
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        ys,
        hard_ys,
        x_image_indexes,
        x_images,
        distance
    ) = zip(*items)
 
    for idx, _ in enumerate(attn_biases):

        attn_biases[idx][1:, 1:][distance[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i['input_ids'].size(0) for i in xs)

    y = torch.cat(ys)
    
    # format y into a n x n matrix where n is the number of graphs and each row has 1 for the correct label and 0 for the rest
    y_matrix = y.unsqueeze(1).eq(y).half()
    # if ys have the same label, they also have a 1 in the column
    
    # print('normal_y\n', y_matrix)
    hard_y = torch.cat(hard_ys)
    # print('pre_y\n', hard_y)
    hard_y_matrix = hard_y.unsqueeze(1).eq(y).half()
    # print('hard_y\n', hard_y_matrix)
    # print(y_matrix.eq(0))

    # normalized weight
    extra_weight = (torch.logical_or(y_matrix.eq(1), hard_y_matrix.eq(1))).sum(dim=1) / (torch.logical_and(y_matrix.eq(0), hard_y_matrix.eq(0))).sum(dim=1) # soft_negs are proportionally weighted to the number of hard_negs and hard_pos
    # unnormalzied
    extra_weight = extra_weight * 2
 
    soft_matrix = torch.where(torch.logical_and(y_matrix.eq(0), hard_y_matrix.eq(0)), extra_weight, 1)
    soft_matrix = torch.where(torch.eye(soft_matrix.size(0)).eq(1), 0, soft_matrix) # we dont include itself in the loss
    #print('SOFT', soft_matrix)
    #y_matrix = torch.where(hard_y_matrix.eq(1), 0, y_matrix)
    
    # print(y, y_matrix, hard_y_matrix)
    # y_matrix = y_matrix - hard_y_matrix
    # print('FINAL\n', y_matrix)
    
    x = {}
    
    # currently in the format of [tokens, size]
    for key in ['input_ids', 'token_type_ids', 'attention_mask']:
        x[key] = torch.cat([pad_2d_custom_unsqueeze(a[key], max_node_num, 0) for a in xs])

    
    token_mask = ~x['input_ids'].eq(0).all(dim=2)
    
    images = [x for x in x_images if not torch.all(x.eq(0))]
    if len(images) != 0:
        x_images = torch.cat(images)
    else:
        x_images = None
    x_image_indexes = torch.cat([pad_1d_custom_unsqueeze(z, -1, False).squeeze(0) for z in x_image_indexes]).bool()

    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )

    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    ).int()
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    
    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x_token_mask=token_mask,
        x=x['input_ids'],
        x_token_type_ids=x['token_type_ids'],
        x_attention_mask=x['attention_mask'],
        x_images=x_images,
        x_image_indexes=x_image_indexes,
        y=y_matrix,
        y_weight=soft_matrix,
        true_y=y
    )
