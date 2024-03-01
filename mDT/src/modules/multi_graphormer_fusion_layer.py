import torch
import torch.nn as nn
from transformers.models.vit.modeling_vit import ViTLayer
from transformers.models.bert.modeling_bert import BertLayer
from typing import Dict, List, Optional, Set, Tuple, Union


class GraphFusionLayer(nn.Module):
    def __init__(
        self,
        bert_layer: BertLayer,
        vit_layer: ViTLayer,
        num_bottle_neck_tokens: int,
        use_projection: bool = False,
    ) -> None:
        super().__init__()

        self.bert_encoder = bert_layer
        self.vit_encoder = vit_layer
        self.gradient_checkpointing = False
        self.num_bottle_neck_tokens = num_bottle_neck_tokens
        if use_projection:
            self.bert_projection = nn.Linear(768, 768)
            self.vit_projection = nn.Linear(768, 768)
        else:
            self.bert_projection = nn.Identity()
            self.vit_projection = nn.Identity()

    def forward(
        self,
        bert_hidden_states: torch.Tensor,
        vit_hidden_states: torch.Tensor,
        bottle_neck: torch.Tensor,
        bert_attention_mask: Optional[torch.FloatTensor] = None,
        x_image_indexes: Optional[torch.Tensor] = None,
    ):
        bert_hidden_states_in = torch.cat(
            [bottle_neck, bert_hidden_states], dim=1
        )

        bert_hidden_output_out = self.bert_forward(
            bert_hidden_states_in, bert_attention_mask, None, None, None
        )

        bert_hidden_output = bert_hidden_output_out[
            :, self.num_bottle_neck_tokens :
        ]
        bottle_neck_output = bert_hidden_output_out[
            :, : self.num_bottle_neck_tokens
        ]
        # print("BOTTLENECK", bottle_neck.shape)
        # print("IMAGE_INDEX", x_image_indexes.shape)
        # print("APPLIED", bottle_neck[x_image_indexes].shape)
        # print("VIT_HIDDEN", vit_hidden_states.shape)
        if vit_hidden_states != None:
            vit_hidden_states_in = torch.cat(
                [bottle_neck[x_image_indexes], vit_hidden_states], dim=1
            )
            vit_hidden_output_out = self.vit_forward(vit_hidden_states_in)
            vit_hidden_output = vit_hidden_output_out[
                :, self.num_bottle_neck_tokens :
            ]
            bottle_neck_output[x_image_indexes] = (
                vit_hidden_output_out[:, : self.num_bottle_neck_tokens]
                + bottle_neck_output[x_image_indexes]
            ) / 2
            # graph_bottleneck[x_image_indexes] = (graph_bottleneck[x_image_indexes] + vit_bottleneck[:, 0, :]) / 2
        else:
            vit_hidden_output = None

        return bert_hidden_output, vit_hidden_output, bottle_neck_output

    def vit_forward(self, hidden_states: torch.Tensor):
        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, False)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.vit_encoder),
                hidden_states,
                layer_head_mask,
            )
        else:
            # layer_outputs = self.vit_encoder(hidden_states, layer_head_mask, None)
            layer_outputs = self.vit_encoder(
                hidden_states, layer_head_mask, False
            )

        hidden_states = layer_outputs[0]

        return hidden_states

        # if output_attentions:
        #     all_self_attentions = all_self_attentions + (layer_outputs[1],)

    def bert_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        layer_head_mask = head_mask if head_mask is not None else None
        past_key_value = None

        if self.gradient_checkpointing and self.training:
            # if use_cache:
            #     logger.warning(
            #         "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            #     )
            #     use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, False)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.bert_encoder),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs = self.bert_encoder(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                False,
            )

        hidden_states = layer_outputs[0]
        return hidden_states


class GraphFusionStack(nn.Module):
    def __init__(
        self,
        bert_layers,
        vit_layers,
        num_bottle_neck_tokens,
        use_projection=False,
    ) -> None:
        super().__init__()

        self.fusion_layers = nn.ModuleList(
            [
                GraphFusionLayer(
                    bert_layer,
                    vit_layer,
                    num_bottle_neck_tokens,
                    use_projection,
                )
                for bert_layer, vit_layer in zip(bert_layers, vit_layers)
            ]
        )
        print("LEN FUSION STACK", len(self.fusion_layers))

    def forward(
        self,
        bert_hidden_states: torch.Tensor,
        vit_hidden_states: torch.Tensor,
        bottle_neck: torch.Tensor,
        bert_attention_mask: Optional[torch.FloatTensor] = None,
        x_image_indexes: Optional[torch.Tensor] = None,
    ):
        for f_layer in self.fusion_layers:
            bert_hidden_states, vit_hidden_states, bottle_neck = f_layer(
                bert_hidden_states,
                vit_hidden_states,
                bottle_neck,
                bert_attention_mask,
                x_image_indexes,
            )

        return bert_hidden_states, vit_hidden_states, bottle_neck
