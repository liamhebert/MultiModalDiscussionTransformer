# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from fairseq.modules import FairseqDropout, LayerDropModuleList, LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .multihead_attention import MultiheadAttention
from .graphormer_layers import GraphNodeFeature, GraphAttnBias
from .graphormer_graph_encoder_layer import GraphormerGraphEncoderLayer
from .multi_graphormer_fusion_layer import GraphFusionLayer
from transformers import AutoModel

def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model.
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class NoGraphormerGraphEncoder(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        num_bottle_neck: int,
        num_fusion_layers: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ) -> None:

        super().__init__()
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable = traceable

        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if pre_layernorm:
            self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.vit_model, vit_other_layers, self.bert_model, bert_other_layers = self.build_vit_bert_encoders(num_fusion_layers)
        self.fusion_layers = nn.ModuleList([])
        
        self.fusion_layers.extend([
            GraphFusionLayer(bert_layer, vit_layer, num_bottle_neck, use_projection=True) for bert_layer, vit_layer in zip(bert_other_layers, vit_other_layers)
        ])
        self.graph_to_bert_projection_layers = nn.ModuleList([])
        self.graph_to_bert_projection_layers.extend([
            nn.Linear(768, 768) for _ in range(num_fusion_layers)
        ])

        self.graph_to_vit_projection_layers = nn.ModuleList([])
        self.graph_to_vit_projection_layers.extend([
            nn.Linear(768, 768) for _ in range(num_fusion_layers)
        ])
        print('NUMBER OF FUSION:', num_fusion_layers)
        # self.fusion_layers = self.fusion_layers[:1]
        self.layers.extend(
            [
                self.build_graphormer_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_fusion_layers)
            ]
        )
        self.num_bottle_neck = num_bottle_neck
        self.bottle_neck = nn.Embedding(num_bottle_neck, 768)
        
        self.graph_norm_layers = nn.ModuleList([nn.LayerNorm(self.embedding_dim, device='cuda:0') for _ in range(num_fusion_layers)])

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        # for layer in range(n_trans_layers_to_freeze):
        #     freeze_module_params(self.layers[layer])

    def build_vit_bert_encoders(self, num_fusion_layers):
        vit_model = AutoModel.from_pretrained('google/vit-base-patch16-224')
        vit_other_layers = vit_model.encoder.layer[-num_fusion_layers:]
        vit_model.encoder.layer = vit_model.encoder.layer[:-num_fusion_layers]
        # note: this still includes the layernorm and pooler layers at the end, may want to remove
        vit_model = nn.DataParallel(vit_model)
        # bert_model = AutoModel.from_pretrained('microsoft/MiniLM-L12-H384-uncased')
        bert_model = AutoModel.from_pretrained('bert-base-cased')
      
        bert_other_layers = bert_model.encoder.layer[-num_fusion_layers:]
        bert_model.encoder.layer = bert_model.encoder.layer[:-num_fusion_layers]
        bert_model = nn.DataParallel(bert_model)
        # this has a pooler at the end
        return vit_model, vit_other_layers, bert_model, bert_other_layers

    def build_graphormer_graph_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        pre_layernorm,
    ):
        return GraphormerGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            pre_layernorm=pre_layernorm,
        )

    def forward(
        self,
        batched_data,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
   
        mask = batched_data['y_mask']
        print('X_TOKEN SHAPE', batched_data['x_token_type_ids'].shape)
        print('X TOKEN MASK', batched_data['x_token_mask'].shape)
        print('Y MASK', batched_data['y_mask'].shape)
        x_token_type_ids = batched_data['x_token_type_ids'][mask, :]
        x_attention_mask = batched_data['x_attention_mask'][mask, :]
        x_input_ids = batched_data['x'][mask, :]
        bert_output = self.bert_model(token_type_ids=x_token_type_ids, attention_mask=x_attention_mask, input_ids=x_input_ids).last_hidden_state
        

        n_graph, n_node = bert_output.size()[:2]
        #print('GRAPH STATS', n_graph, n_node)
    
        bottle_neck = self.bottle_neck.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        bert_output = torch.cat([bottle_neck, bert_output], dim=1)
        added_mask = torch.Tensor([1] * self.num_bottle_neck).unsqueeze(0).repeat(n_graph, 1).cuda()
        x_attention_mask = torch.cat([added_mask, x_attention_mask], dim=1)
        extended_attention_mask = x_attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.half)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.half).min
        
        if batched_data['x_images'] != None:
            print('X IMAGE INDEX', batched_data['x_image_indexes'].shape)
            res = batched_data['x_image_indexes'][batched_data['y_mask']]
            print('FILTERED X_IMAGE_INDEX', res.shape)
            vit_output = self.vit_model(batched_data['x_images']).last_hidden_state
            n_graph, n_node = vit_output.size()[:2]
            bottle_neck = self.bottle_neck.weight.unsqueeze(0).repeat(n_graph, 1, 1)
            vit_output = torch.cat([bottle_neck, vit_output], dim=1)
        else:
            vit_output = None
        print('IMAGE INDEXES', batched_data['x_image_indexes'].shape)

        bert_output, vit_output, bottle_neck = self.fusion_layers[0](bert_output, vit_output, extended_attention_mask, None, None, None, None, batched_data['x_image_indexes'])
        bottle_neck_nodes = bottle_neck[:, 0, :]
        shape = batched_data['x'].shape
        # graph_data = torch.zeros((shape[0], shape[1], 768)).half().cuda()
        # graph_data[mask, :] = bottle_neck_nodes
       
 
        # # compute padding mask. This is needed for multi-head attention
        # x = graph_data
        
        # n_graph, n_node = x.size()[:2]
        # padding_mask = (x[:, :, 0]).eq(0)  # B x T x 1
        # padding_mask_cls = torch.zeros(
        #     n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        # )
        # #print(padding_mask_cls.shape, padding_mask.shape)
        # padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        # mask = torch.cat((padding_mask_cls, mask), dim=1)
        # # B x (T+1) x 1

   
        # x = self.graph_node_feature(x, batched_data['in_degree'], batched_data['out_degree'])
       
        # if perturb is not None:
        #     #ic(torch.mean(torch.abs(x[:, 1, :])))
        #     #ic(torch.mean(torch.abs(perturb)))
        #     x[:, 1:, :] += perturb

        # # x: B x T x C

        # attn_bias = self.graph_attn_bias(batched_data)

        # if self.embed_scale is not None:
        #     x = x * self.embed_scale

        # if self.quant_noise is not None:
        #     x = self.quant_noise(x)

        # if self.emb_layer_norm is not None:
        #     x = self.emb_layer_norm(x)

        # x = self.dropout_module(x)

        # # account for padding while computing the representation
        # # B x T x C -> T x B x C
        # x = x.transpose(0, 1)
        

        # inner_states = []
        # if not last_state_only:
        #     inner_states.append(x)
        
        for g_layer, f_layer, g_norm, g_to_b_projection, g_to_v_projection in zip(self.layers, self.fusion_layers, self.graph_norm_layers, self.graph_to_vit_projection_layers, self.graph_to_bert_projection_layers):
            
            
            # x, _ = g_layer(
            #     x,
            #     self_attn_padding_mask=padding_mask,
            #     self_attn_mask=attn_mask,
            #     self_attn_bias=attn_bias,
            # )
            
            # # extract bottle_neck tokens
            # # T x B x C -> B x T x C
            # x = x.transpose(0, 1) 
           
            # bottle_neck = x[mask, :]
            # bert_output[:, 0, :] = g_to_b_projection(bottle_neck) + bert_output[:, 0, :]
            
            # if vit_output != None:
            #     vit_output[:, 0, :] = g_to_v_projection(bottle_neck[batched_data['x_image_indexes']]) + vit_output[:, 0, :]
            
            bert_output, vit_output, bottle_neck = f_layer(bert_output, vit_output, extended_attention_mask, None, None, None, None, batched_data['x_image_indexes'])
            
            # x[mask, :] = bottle_neck[:, 0, :]
            # # B x T x C -> T x B x C
            # x = g_norm(x)
            # x = x.transpose(0, 1)
            # if not last_state_only:
            #     inner_states.append(x)

        # graph_rep = x[0, :, :]

        # if last_state_only:
        #     inner_states = [bert_output]

        if self.traceable:
            if vit_output != None:
                return bert_output[:, self.num_bottle_neck, :], vit_output[:, self.num_bottle_neck, :]
            else:
                return bert_output[:, self.num_bottle_neck, :], None
        else:
            if vit_output != None:
                return bert_output[:, self.num_bottle_neck, :], vit_output[:, self.num_bottle_neck, :]
            else:
                return bert_output[:, self.num_bottle_neck, :], None
