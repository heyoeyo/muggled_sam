#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from .key_regex import get_nth_integer


# ---------------------------------------------------------------------------------------------------------------------
# %% Main function


def get_model_config_from_state_dict(state_dict):

    # Get total block count, which is used to infer the 'size' of the model, needed to guess other configs
    total_block_count = get_image_encoder_total_block_count(state_dict)

    # Not clear if this value can be interpretted from model weights...
    hardcoded_num_decoder_heads = 8

    # Get model config from state dict
    config_dict = {
        "patch_size_px": get_patch_size_px(state_dict),
        "features_per_image_token": get_image_encoder_features_per_token(state_dict),
        "features_per_prompt_token": get_features_per_prompt_token(state_dict),
        "features_per_decoder_token": get_mask_decoder_features_per_token(state_dict),
        "features_per_memory_token": get_features_per_memory_token(state_dict),
        "imgencoder_heads": get_image_encoder_heads(total_block_count),
        "imgencoder_blocks_per_stage": get_blocks_per_stage(total_block_count),
        "imgencoder_window_size_per_stage": get_window_size_per_stage(total_block_count),
        "imgencoder_window_tile_posenc_hw": get_image_encoder_window_tile_positional_encoding_hw(state_dict),
        "imgencoder_global_attn_spacing_per_stage": get_global_attention_spacing_per_stage(total_block_count),
        "base_patch_grid_hw": get_base_patch_grid_hw(state_dict),
        "num_output_mask_tokens": get_num_output_mask_tokens(state_dict),
        "num_decoder_blocks": get_mask_decoder_block_count(state_dict),
        "num_decoder_heads": hardcoded_num_decoder_heads,
        "is_version_2p1": check_is_version_2p1(state_dict),
    }

    return config_dict


# ---------------------------------------------------------------------------------------------------------------------
# %% Component functions

# .....................................................................................................................


def get_image_encoder_heads(total_block_count):
    """
    It's not clear if the image encoder head count can be determined directly
    from the model weights. Instead it is found through a hard-coded mapping,
    based on the the known head counts for different model sizes!
    """

    if total_block_count <= 16:
        num_heads = 1
    elif total_block_count <= 48:
        num_heads = 2
    else:
        ValueError(f"Cannot determine image encoder head count! Unrecognized model block count: {total_block_count}")

    return num_heads


def get_global_attention_spacing_per_stage(total_block_count):

    # Hard-code the known mapping between block counts & global attention spacing
    # -> Isn't part of model weights
    # -> Doesn't seem to follow a clear pattern...
    spacing_by_blockcount_lut = {
        12: (None, None, 2, None),
        16: (None, None, 3, None),
        24: (None, None, 4, None),
        48: (None, None, 10, None),
    }

    # Warn if we don't know what to do with block count
    global_attn_spacing = spacing_by_blockcount_lut.get(total_block_count, None)
    if global_attn_spacing is None:
        raise ValueError(
            f"Cannot determine global attention spacing! Unrecognized model block count: {total_block_count}"
        )

    return global_attn_spacing


def get_window_size_per_stage(total_block_count):

    # Hard-code the known mapping between block counts & per-stage window sizing
    # -> Isn't part of model weights
    # -> Doesn't seem to follow a clear pattern...
    window_sizes_per_stage_by_blockcount_lut = {
        12: (8, 4, 14, 7),
        16: (8, 4, 14, 7),
        24: (8, 4, 14, 7),
        48: (8, 4, 16, 8),
    }

    # Warn if we don't know what to do with block count
    winsizes_per_stage = window_sizes_per_stage_by_blockcount_lut.get(total_block_count, None)
    if winsizes_per_stage is None:
        raise ValueError(f"Cannot determine window size per stage! Unrecognized model block count: {total_block_count}")

    return winsizes_per_stage


def get_blocks_per_stage(total_block_count):

    # Hard-code the known mapping between block counts & blocks per-stage
    # -> Isn't part of model weights
    # -> Doesn't seem to follow a clear pattern...
    blocks_per_stage_by_blockcount_lut = {
        12: (1, 2, 7, 2),
        16: (1, 2, 11, 2),
        24: (2, 3, 16, 3),
        48: (2, 6, 36, 4),
    }

    # Warn if we don't know what to do with block count
    blocks_per_stage = blocks_per_stage_by_blockcount_lut.get(total_block_count, None)
    if blocks_per_stage is None:
        raise ValueError(f"Cannot determine blocks per stage! Unrecognized model block count: {total_block_count}")

    return blocks_per_stage


def get_features_per_memory_token(state_dict):

    target_key = "memory_encoder.out_proj.weight"
    assert target_key in state_dict.keys(), f"Error determining memory token features! Couldn't find key: {target_key}"

    # Expecting weights with shape: FxPx1x1
    # -> F is num features per memory token
    # -> P is num features per prompt token
    features_per_memory_token, _, _, _ = state_dict[target_key].shape

    return int(features_per_memory_token)


def get_base_patch_grid_hw(state_dict):

    target_key = "image_encoder.trunk.pos_embed"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining base patch grid hw in image encoder! Couldn't find key: {target_key}"

    # Expecting weights with shape: 1xFxHxW
    # -> F is num features per token in transformer
    # -> H, W are the sizes we're looking for
    _, _, h, w = state_dict[target_key].shape

    return (int(h), int(w))


def get_image_encoder_total_block_count(state_dict):
    """
    State dict contains keys like:
        'image_encoder.trunk.blocks.0.norm1.weight'
        'image_encoder.trunk.blocks.4.attn.qkv.bias',
        'image_encoder.trunk.blocks.10.mlp.layers.0.weight',
        ... etc
    This function tries to find the largest number from the '...blocks.#...' part of these keys,
    since this determines how many total blocks are in the image encoder.
    """

    # Get indexing of every image encoder 'block' entry
    is_block_key = lambda key: key.startswith("image_encoder.trunk.blocks")
    block_idxs = [get_nth_integer(key, 0) for key in state_dict.keys() if is_block_key(key)]

    # Take the max block index (+1 due to zero-indexing) to determine number of transformer blocks
    assert len(block_idxs) > 0, "Error determining number of image encoder blocks! Could not find any blocks"
    num_imgenc_blocks = 1 + max(block_idxs)

    return int(num_imgenc_blocks)


# .....................................................................................................................


def get_image_encoder_features_per_token(state_dict):
    """
    The state dict is expected to contain weights for the patch embedding layer.
    This is a convolutional layer responsible for 'chopping up the image' into image patch tokens.
    The number of output channels from this convolution layer determines the number of features per token
    for the image encoder (which immediately follows the patch embedding).
    """

    # Make sure there is a patch embedding key in the given state dict
    target_key = "image_encoder.trunk.patch_embed.proj.weight"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining features per token in image encoder! Couldn't find key: {target_key}"

    # Expecting weights with shape: Fx3xPxP
    # -> F is num features per token in transformer
    # -> 3 is expected channel count of input RGB images
    # -> P is patch size in pixels
    features_per_image_token, _, _, _ = state_dict[target_key].shape

    return int(features_per_image_token)


def get_image_encoder_window_tile_positional_encoding_hw(state_dict):
    """
    The state dict is expected to contain weights for a 'tiled' position
    encoding/embedding value, used in the positional encoding of the
    image encoder. This 'tile' is repeated to match the size of the
    image token patch grid size, and was (in the original implementation)
    tied directly to the first (image encoder) stage window size. However
    it has been split out here, to allow for the possibility of adjusting
    windows sizes without breaking the positional encoding tiling!
    """

    # Make sure there is a patch embedding key in the given state dict
    target_key = "image_encoder.trunk.pos_embed_window"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining features per token in image encoder! Couldn't find key: {target_key}"

    # Expecting weights with shape: 1x3xHxW
    # -> F is num features per image token
    # -> H & W are the height & width we're looking for
    _, _, h, w = state_dict[target_key].shape

    return (int(h), int(w))


# .....................................................................................................................


def get_patch_size_px(state_dict):
    """
    The state dict is expected to contain weights for the patch embedding layer.
    This is a convolutional layer responsible for 'chopping up the image' into image patch tokens.
    The kernel size (and stride) of this convolution layer corresponds to the patch sizing (in pixels)
    that we're after. We assume the kernel is square, so patch width & height are the same.
    """

    # Make sure there is a patch embedding key in the given state dict
    target_key = "image_encoder.trunk.patch_embed.proj.weight"
    assert target_key in state_dict.keys(), f"Error determining patch size! Couldn't find key: {target_key}"

    # Expecting weights with shape: Fx3xPxP
    # -> F is num features per token in transformer
    # -> 3 is expected channel count of input RGB images
    # -> P is patch size in pixels
    _, _, _, patch_size_px = state_dict[target_key].shape

    return int(patch_size_px)


# .....................................................................................................................


def get_features_per_prompt_token(state_dict):
    """
    The state dict is expected to contain many weights which encode the number of
    features per prompt, a simple weight to target is to find the learned
    'not-a-point' embedding, which is sized to match the features per prompt.
    """

    # Make sure the target key is in the given state dict
    target_key = "sam_prompt_encoder.not_a_point_embed.weight"
    assert target_key in state_dict.keys(), f"Error determining features per prompt token! Couldn't find: {target_key}"

    # Expecting weights of shape: 1xF
    # -> F is the features per prompt token
    _, features_per_prompt_token = state_dict[target_key].shape

    return int(features_per_prompt_token)


# .....................................................................................................................


def get_num_output_mask_tokens(state_dict):

    target_key = "sam_mask_decoder.mask_tokens.weight"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining the number of output mask tokens! Couldn't find key: {target_key}"

    num_mask_tokens, _ = state_dict[target_key].shape

    return int(num_mask_tokens)


# .....................................................................................................................


def get_mask_decoder_features_per_token(state_dict):

    target_key = "sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.q_proj.weight"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining mask decoder feature count! Couldn't find key: {target_key}"

    downsample_channels, _ = state_dict[target_key].shape

    return int(downsample_channels)


# .....................................................................................................................


def get_mask_decoder_block_count(state_dict):

    # Get indexing of every mask decoder layer entry
    is_block_key = lambda key: key.startswith("sam_mask_decoder.transformer.layers")
    block_idxs = [get_nth_integer(key, 0) for key in state_dict.keys() if is_block_key(key)]

    # Take the max block index (+1 due to zero-indexing) to determine number of blocks
    assert len(block_idxs) > 0, "Error determining number of mask decoder blocks! Could not find any blocks"
    num_maskdec_blocks = 1 + max(block_idxs)

    return int(num_maskdec_blocks)


# .....................................................................................................................


def check_is_version_2p1(state_dict) -> bool:
    """
    An updated version of SAMv2, called v2.1, was released a couple months
    after the initial release. It contains some additional parameters, and
    slight differences in model structure (to use these parameters) compared
    to v2.

    This function checks for these extra parameters and returns a boolean
    flag which is True if version 2.1 (aka 2p1) is loaded.
    """

    target_keys_list = ["no_obj_embed_spatial", "obj_ptr_tpos_proj.weight", "obj_ptr_tpos_proj.bias"]
    return all(key in state_dict.keys() for key in target_keys_list)
