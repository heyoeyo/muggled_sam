#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from torch import nn

from .mask_decoder_attention import CrossAttentionNormed, SelfAttentionNoPosenc, SelfAttentionNormed

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class CrossAttentionTransformer(nn.Module):
    """
    Simplified implementation of the 'TwoWayTransformer' model/component described in:
        "Segment Anything"
        By: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
            Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
        @ https://arxiv.org/abs/2304.02643

    The code here is adapted from the original segment-anything repo:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L16

    This model is used to perform cross-attention between prompt tokens and image tokens
    for the SAM model.
    """

    # .................................................................................................................

    def __init__(self, depth: int, num_heads: int, features_per_token: int, downsample_features: int) -> None:

        # Inherit from parent
        super().__init__()

        # Form transformer body
        self.layers = nn.ModuleList()
        for idx in range(depth):
            skip_self_attn_posenc = idx == 0
            attn_block = CrossAttentionBlock(num_heads, features_per_token, downsample_features, skip_self_attn_posenc)
            self.layers.append(attn_block)

        # Create final output attention layer
        self.final_prompt_crossattn = CrossAttentionNormed(num_heads, features_per_token, downsample_features)

    # .................................................................................................................

    def forward(
        self, prompt_tokens: Tensor, image_tokens_bchw: Tensor, image_posenc_bchw: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Encodes prompt & image tokens using multiple cross-attention layers
        Input prompt tokens are expected to have shape: BxNxF
        Image tokens should have shape: BxCxHxW
        Image position encoding should shape matching image tokens: BxCxHxW
        -> B is batch size, N is number of (prompt) tokens, F is features per token
        -> C is image token channels (i.e. features), H is image token 'height' and W is width
        -> F and C should match! That is, features per prompt & features per image token should be the same

        Returns:
            encoded_prompt_tokens, encoded_image_tokens
            -> Both have shape: BxNxF
            -> N is number of tokens (same as input for prompts, N=H*W for image tokens)
            -> F is same features per token as inputs
        """

        # Convert image-related inputs from image-like shape to 'row-of-tokens' format
        # -> Shape changes from: BxCxHxW -> BxNxC (where N is the number of tokens = H*W)
        image_tokens = image_tokens_bchw.flatten(2).permute(0, 2, 1)
        image_posenc = image_posenc_bchw.flatten(2).permute(0, 2, 1)

        # Run transformer blocks to encode tokens
        # (original implementation uses prompts as their own positional encoding!)
        prompt_posenc = prompt_tokens
        for layer in self.layers:
            prompt_tokens, image_tokens = layer(prompt_tokens, prompt_posenc, image_tokens, image_posenc)

        # Apply final cross-encoding of prompt-tokens
        prompt_tokens = self.final_prompt_crossattn(prompt_tokens, prompt_posenc, image_tokens, image_posenc)

        return prompt_tokens, image_tokens

    # .................................................................................................................


class CrossAttentionBlock(nn.Module):
    """
    Simplified implementation of the 'TwoWayAttentionBlock' model/component described in:
        "Segment Anything"
        By: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
            Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
        @ https://arxiv.org/abs/2304.02643

    The code here is adapted from the original segment-anything repo:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L109

    This is a single cross-attention block used within the cross-attention transformer.
    It is specifically built to encode prompt & image inputs for the SAM mask decoder.
    The model has a structure that is very different from the vision transformer blocks.

    It takes in prompt tokens, images tokens along with position encodings for each.
    The model processing consists of 4 steps:
        1. Self-attention on prompt tokens (with or without positional encoding, based on config)
        2. Cross-attention to further encode prompt tokens
        3. A 2 layer MLP on prompt tokens
        4. Cross-attention to encode image tokens
    """

    # .................................................................................................................

    def __init__(
        self,
        num_heads: int,
        features_per_token: int,
        downsample_features: int,
        skip_selfattn_posenc=False,
        mlp_ratio=8,
    ):

        # Inherit from parent
        super().__init__()

        # Switch self-attention mechanism, depending on whether we're meant to skip the position encoding
        SelfAttnModule = SelfAttentionNoPosenc if skip_selfattn_posenc else SelfAttentionNormed
        self.prompt_selfattn = SelfAttnModule(num_heads, features_per_token)

        # Build out remaining layers for further cross-attention processing
        self.prompt_crossattn = CrossAttentionNormed(num_heads, features_per_token, downsample_features)
        self.prompt_mlpnorm = MLP2LayersNormed(features_per_token, mlp_ratio)
        self.image_crossattn = CrossAttentionNormed(num_heads, features_per_token, downsample_features)

    # .................................................................................................................

    def forward(self, prompt_tokens, prompt_posenc, image_tokens, image_posenc):
        """
        Encodes prompt & image tokens using cross-attention
        Returns:
            encoded_prompt_tokens, encoded_image_tokens
        """

        prompt_tokens = self.prompt_selfattn(prompt_tokens, prompt_posenc)
        prompt_tokens = self.prompt_crossattn(prompt_tokens, prompt_posenc, image_tokens, image_posenc)
        prompt_tokens = self.prompt_mlpnorm(prompt_tokens)
        image_tokens = self.image_crossattn(image_tokens, image_posenc, prompt_tokens, prompt_posenc)

        return prompt_tokens, image_tokens

    # .................................................................................................................


class MLP2LayersNormed(nn.Module):
    """
    Simplified implementation of the 2-layer MLP model used by the cross-attention-block model of SAM:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L141

    There are several (minor) changes in this implementation:
        - The activation function (ReLU) is hard-coded
        - The number of hidden features is specified using a multiplier (instead of directly providing the value)
        - A layernorm has been added on the output, as this matches the other attention blocks where it is used
    """

    # .................................................................................................................

    def __init__(self, num_features, hidden_features_ratio=8):

        # Inherit from parent
        super().__init__()

        num_hidden_features = int(round(hidden_features_ratio * num_features))
        self.mlp = nn.Sequential(
            nn.Linear(num_features, num_hidden_features),
            nn.ReLU(),
            nn.Linear(num_hidden_features, num_features),
        )
        self.norm = nn.LayerNorm(num_features)

    def forward(self, tokens):
        mlp_out = self.mlp(tokens)
        return self.norm(tokens + mlp_out)

    # .................................................................................................................
