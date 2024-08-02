#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.memattn_components import MemoryAttentionLayer

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV2MemoryAttention(nn.Module):

    # .................................................................................................................

    def __init__(
        self,
        d_model: int = 256,
        d_memory: int = 64,
        num_layers: int = 4,
        # pos_enc_at_input: bool = True,
        # batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        # self.pos_enc_at_input = pos_enc_at_input
        # self.batch_first = batch_first

        layers_list = []
        for _ in range(num_layers):
            layer = MemoryAttentionLayer(d_model, d_memory, mlp_ratio=8)
            layers_list.append(layer)
        self.layers = nn.ModuleList(layers_list)  # get_clones(layer, num_layers)

    # .................................................................................................................

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Tensor | None = None,  # pos_enc for self-attention inputs
        memory_pos: Tensor | None = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert curr.shape[1] == memory.shape[1], "Batch size must be the same for curr and memory"

        output = curr
        if curr_pos is not None:
            output = output + 0.1 * curr_pos

        # Convert to batch first
        output = output.transpose(0, 1)
        curr_pos = curr_pos.transpose(0, 1)
        memory = memory.transpose(0, 1)
        memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                num_k_exclude_rope=num_obj_ptr_tokens,
            )
        normed_output = self.norm(output)

        # Convert back to seq first
        normed_output = normed_output.transpose(0, 1)
        curr_pos = curr_pos.transpose(0, 1)

        return normed_output

    # .................................................................................................................
