#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from .key_regex import get_nth_integer

from collections import Counter


# ---------------------------------------------------------------------------------------------------------------------
# %% Main function


def get_model_config_from_state_dict(state_dict):

    # Get feature count separate, since we need it to determine number of heads
    features_per_image_patch_token = get_image_encoder_features_per_token(state_dict)

    # Not clear if this value can be interpretted from model weights...
    hardcoded_num_decoder_heads = 8

    # Get model config from state dict
    config_dict = {
        "patch_size_px": get_patch_size_px(state_dict),
        "base_patch_grid_hw": get_image_encoder_base_patch_grid_size(state_dict),
        "features_per_image_token": features_per_image_patch_token,
        "features_per_prompt_token": get_features_per_prompt_token(state_dict),
        "features_per_decoder_token": get_mask_decoder_features_per_token(state_dict),
        "num_encoder_stages": get_image_encoder_stage_count(state_dict),
        "num_encoder_blocks": get_image_encoder_block_count(state_dict),
        "num_encoder_heads": get_image_encoder_heads(state_dict, features_per_image_patch_token),
        "base_window_size": get_image_encoder_window_size(state_dict),
        "num_output_mask_tokens": get_num_output_mask_tokens(state_dict),
        "num_decoder_blocks": get_mask_decoder_block_count(state_dict),
        "num_decoder_heads": hardcoded_num_decoder_heads,
    }

    return config_dict


# ---------------------------------------------------------------------------------------------------------------------
# %% Component functions


def get_image_encoder_block_count(state_dict):
    """
    State dict contains keys like:
        'image_encoder.blocks.0.norm1.weight'
        'image_encoder.blocks.4.attn.rel_pos_h',
        'image_encoder.blocks.10.mlp.lin1.weight',
        ... etc
    This function tries to find the largest number from the '...blocks.#...' part of these keys,
    since this determines how many layers (aka depth) are in the transformer.
    """

    # Get indexing of every image encoder 'block' entry
    is_block_key = lambda key: key.startswith("image_encoder.blocks")
    block_idxs = [get_nth_integer(key, 0) for key in state_dict.keys() if is_block_key(key)]

    # Take the max block index (+1 due to zero-indexing) to determine number of transformer blocks
    assert len(block_idxs) > 0, "Error determining number of image encoder blocks! Could not find any blocks"
    num_imgenc_blocks = 1 + max(block_idxs)

    return int(num_imgenc_blocks)


# .....................................................................................................................


def get_image_encoder_stage_count(state_dict):
    """
    The state dict is expected to contain relative positional encodings
    within the image encoder at multiple layers, whose shape depends on
    whether it is a windowed or 'global' attention layer

    The model is assumed to be made of sequences of windowed layers followed by
    a single global attention layer. Each of these sequences is considered 1 stage.
    We can count the number of stages (which is 4 for all known models) by
    counting how many of these larger windowing sizes we find in the relative
    positioning layer entries.
    """

    # Get all 'number of relative window indices' entries
    # -> We don't care about the literal value, just the count of small-to-larger entries
    is_relpos_layer = lambda k: k.startswith("image_encoder.blocks") and k.endswith("rel_pos_h")
    window_indices = [data.shape[0] for key, data in state_dict.items() if is_relpos_layer(key)]

    # Infer the number of stages by counting how many 'global' windowing layers are present
    # -> The global windows are recognized due to larger window sizing!
    indices_counts = Counter(window_indices)
    size_of_global_layers = max(indices_counts.keys())
    num_stages = indices_counts[size_of_global_layers]

    return int(num_stages)


# .....................................................................................................................


def get_image_encoder_heads(state_dict, features_per_image_token):
    """
    The state dict contains entries for relative positional encodings,
    which have keys ending in '...attn.rel_pos_h' for example.

    The shape of the position encodings is based off of the number of
    features per head. Given the number of features per token, we can
    infer the number of heads, based on the ratio of the
    per-token to per-head feature counts.
    """

    # Warning if we can't find the target key
    target_key = "image_encoder.blocks.0.attn.rel_pos_h"
    assert target_key in state_dict.keys(), f"Error determining image encoder heads! Couldn't find key: {target_key}"

    # Compute number of heads from feature counts
    _, features_per_head = state_dict[target_key].shape
    num_heads = features_per_image_token // features_per_head

    return int(num_heads)


# .....................................................................................................................


def get_image_encoder_features_per_token(state_dict):
    """
    The state dict is expected to contain weights for the patch embedding layer.
    This is a convolutional layer responsible for 'chopping up the image' into image patch tokens.
    The number of output channels from this convolution layer determines the number of features per token
    for the image encoder (which immediately follows the patch embedding).
    """

    # Make sure there is a patch embedding key in the given state dict
    target_key = "image_encoder.patch_embed.proj.weight"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining features per token in image encoder! Couldn't find key: {target_key}"

    # Expecting weights with shape: Fx3xPxP
    # -> F is num features per token in transformer
    # -> 3 is expected channel count of input RGB images
    # -> P is patch size in pixels
    features_per_image_token, _, _, _ = state_dict[target_key].shape

    return int(features_per_image_token)


# .....................................................................................................................


def get_patch_size_px(state_dict):
    """
    The state dict is expected to contain weights for the patch embedding layer.
    This is a convolutional layer responsible for 'chopping up the image' into image patch tokens.
    The kernel size (and stride) of this convolution layer corresponds to the patch sizing (in pixels)
    that we're after. We assume the kernel is square, so patch width & height are the same.
    """

    # Make sure there is a patch embedding key in the given state dict
    target_key = "image_encoder.patch_embed.proj.weight"
    assert target_key in state_dict.keys(), f"Error determining patch size! Couldn't find key: {target_key}"

    # Expecting weights with shape: Fx3xPxP
    # -> F is num features per token in transformer
    # -> 3 is expected channel count of input RGB images
    # -> P is patch size in pixels
    _, _, _, patch_size_px = state_dict[target_key].shape

    return int(patch_size_px)


# .....................................................................................................................


def get_image_encoder_base_patch_grid_size(state_dict):
    """
    The state dict is expected to contain a positional embedding for the
    image encoder which has a shape of: 1xHxWxF
    -> Where 1 is the batch dimension
    -> H & W are the base grid height & width
    -> F is the number of features per token of the image encoder
    -> Where N is the number of patch grid tokens + 1 (to account for class token)
    """

    # Make sure we have the target key
    target_key = "image_encoder.pos_embed"
    assert target_key in state_dict.keys(), f"Error determining base patch grid size. Couldn't find key: {target_key}"

    # Expecting shape: 1xHxWxF
    _, base_grid_h, base_grid_w, _ = state_dict[target_key].shape
    base_grid_hw = (int(base_grid_h), int(base_grid_w))

    return base_grid_hw


# .....................................................................................................................


def get_image_encoder_window_size(state_dict):
    """
    The state dict is expected to contain relative positional encodings
    within the image encoder at multiple layers, whose shape depends on
    the window sizing of the model. More specifically, the shape is: Sxf
    -> Where S is the number of relative (windowed) indices
    -> f is the number of features per head of the encoder

    For a window size of say 4, the range of possible relative indices
    is from -3 to +3. This is because a token within a window of size 4
    could be at most +/- 3 entries away from any other token. In this case,
    the number of indices, S, would be (2*4 - 1) = 7. This is the value we read from
    the model weight, but we can then work backwards to get the window size!
    """

    # Make sure we can find the target keys
    relpos_h_key = "image_encoder.blocks.0.attn.rel_pos_h"
    relpos_w_key = "image_encoder.blocks.0.attn.rel_pos_w"
    assert (relpos_h_key in state_dict.keys()) and (
        relpos_w_key in state_dict.keys()
    ), f"Error determining window size! Couldn't find {relpos_h_key} and {relpos_w_key} keys"

    # Use larger of height & width sizing just in case (should be the same)
    num_window_h, _ = state_dict["image_encoder.blocks.0.attn.rel_pos_h"].shape
    num_window_w, _ = state_dict["image_encoder.blocks.0.attn.rel_pos_w"].shape
    window_size = max((num_window_h + 1) // 2, (num_window_w + 1) // 2)

    return int(window_size)


# .....................................................................................................................


def get_features_per_prompt_token(state_dict):
    """
    The state dict is expected to contain many weights which encode the number of
    features per prompt, a simple weight to target is to find the learned
    'not-a-point' embedding, which is sized to match the features per prompt.
    """

    # Make sure the target key is in the given state dict
    target_key = "prompt_encoder.not_a_point_embed.weight"
    assert target_key in state_dict.keys(), f"Error determining features per prompt token! Couldn't find: {target_key}"

    # Expecting weights of shape: 1xF
    # -> F is the features per prompt token
    _, features_per_prompt_token = state_dict[target_key].shape

    return int(features_per_prompt_token)


# .....................................................................................................................


def get_num_output_mask_tokens(state_dict):

    target_key = "mask_decoder.mask_tokens.weight"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining the number of output mask tokens! Couldn't find key: {target_key}"

    num_mask_tokens, _ = state_dict[target_key].shape

    return int(num_mask_tokens)


# .....................................................................................................................


def get_mask_decoder_features_per_token(state_dict):

    target_key = "mask_decoder.transformer.layers.0.cross_attn_token_to_image.q_proj.weight"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining mask decoder feature count! Couldn't find key: {target_key}"

    downsample_channels, _ = state_dict[target_key].shape

    return int(downsample_channels)


# .....................................................................................................................


def get_mask_decoder_block_count(state_dict):

    # Get indexing of every mask decoder layer entry
    is_block_key = lambda key: key.startswith("mask_decoder.transformer.layers")
    block_idxs = [get_nth_integer(key, 0) for key in state_dict.keys() if is_block_key(key)]

    # Take the max block index (+1 due to zero-indexing) to determine number of blocks
    assert len(block_idxs) > 0, "Error determining number of mask decoder blocks! Could not find any blocks"
    num_maskdec_blocks = 1 + max(block_idxs)

    return int(num_maskdec_blocks)
