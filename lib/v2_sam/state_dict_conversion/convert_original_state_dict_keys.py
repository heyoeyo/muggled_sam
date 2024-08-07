#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from .key_regex import get_nth_integer, get_suffix_terms, find_match_by_lut


# ---------------------------------------------------------------------------------------------------------------------
# %% Main function


def convert_state_dict_keys(config_dict: dict, original_state_dict: dict) -> dict[str, dict]:
    """
    Function which converts original Segment-Anything V2 model weights
    into the new format needed by the model implementation in this repo (MuggledSAM)
    (layer names are renamed to make the model easier to understand, some are deleted or re-arranged)

    Returns:
        new_state_dict

    Note: The new state dict has keys corresponding the the model components:
        "imgencoder", "coordencoder", "promptencoder", "maskdecoder", "memoryencoder", "memoryattention"
    """

    # Pre-compute mapping between old block indexing & new stage + block indexing
    # - This is used to convert from: trunk.blocks.9.... -to-> trunk.stages.2.blocks.4
    # - For example, for the tiny model which has blocks_per_stage: (2, 3, 7, 2),
    #   'new' block indexing would look like: 0, 1 | 0, 1, 2 | 0, 1, 2, 3, 4, 5, 6 | 0, 1
    #   where each | ... | represents the block indexing within a stage
    imgenc_blocks_per_stage = config_dict["imgencoder_blocks_per_stage"]
    block_idx_to_stage_idx = []
    block_idx_offset_by_stage = []
    for stage_idx, num_blocks in enumerate(imgenc_blocks_per_stage):
        block_idx_to_stage_idx += [stage_idx for _ in range(num_blocks)]
        block_idx_offset_by_stage.append(sum(imgenc_blocks_per_stage[:stage_idx]))

    # Allocate storage for state dict of each (new) model component
    imgenc_sd = {}
    coordencoder_sd = {}
    promptencoder_sd = {}
    maskdecoder_sd = {}
    memencoder_sd = {}
    memattn_sd = {}

    # Loop over all midas state dict keys and convert them to new formatting
    found_key = lambda key: key is not None
    for orig_key, orig_data in original_state_dict.items():

        # For sanity, make sure the key is definitely a string
        orig_key = str(orig_key)

        new_key = _convert_imgenc_keys(orig_key, block_idx_to_stage_idx, block_idx_offset_by_stage)
        if found_key(new_key):

            # Correct layernorm2d weight shapes
            layernorm_key_hints = ("output_projection.1", "output_projection.3")
            mod_data = _reshape_layernorm2d(new_key, orig_data, *layernorm_key_hints)

            imgenc_sd[new_key] = mod_data
            continue

        new_key = _convert_coordencoder_keys(orig_key)
        if found_key(new_key):
            coordencoder_sd[new_key] = orig_data
            continue

        new_key = _convert_promptencoder_keys(orig_key)
        if found_key(new_key):
            promptencoder_sd[new_key] = orig_data
            continue

        new_key = _convert_maskdecoder_keys(orig_key)
        if found_key(new_key):

            # Correct layernorm2d weight shapes
            layernorm_key_hints = ("downscaler.1", "downscaler.4", "img_patch_upscaler.norm")
            mod_data = _reshape_layernorm2d(new_key, orig_data, *layernorm_key_hints)

            maskdecoder_sd[new_key] = mod_data
            continue

        new_key = _convert_memencoder_keys(orig_key)
        if found_key(new_key):

            # Correct layernorm2d weight shapes
            layernorm_key_hints = [f"mask_downsampler.{idx}." for idx in (1, 4, 7, 10)]
            layernorm_key_hints.extend(["fuser.0.norm", "fuser.1.norm"])
            mod_data = _reshape_layernorm2d(new_key, orig_data, *layernorm_key_hints)

            memencoder_sd[new_key] = mod_data
            continue

        new_key = _convert_memattention_keys(orig_key)
        if found_key(new_key):
            memattn_sd[new_key] = orig_data
            continue

    # Bundle new state dict model components together for easier handling
    new_state_dict = {
        "imgencoder": imgenc_sd,
        "coordencoder": coordencoder_sd,
        "promptencoder": promptencoder_sd,
        "maskdecoder": maskdecoder_sd,
        "memoryencoder": memencoder_sd,
        "memoryattention": memattn_sd,
    }

    return new_state_dict


# ---------------------------------------------------------------------------------------------------------------------
# %% Component functions


def _reshape_layernorm2d(key, data, *key_hints):
    """
    Helper used to re-shape weight tensors for layernorm layers. The original
    model used weights that had a single dimension (i.e. a vector), but then
    internally added the missing batch/height/width dimensions to make them
    image-like, with a shape of: 1xFx1x1
    (where F is the length of the original weight 'vector')

    Rather than re-shaping every time the model runs, we can reshape on load
    and just use the result directly!
    """

    # If we get a key with a matching hint, reshape it
    for hint in key_hints:
        if hint in key:
            return data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    return data


# .....................................................................................................................


def _convert_imgenc_keys(
    key: str, block_idx_to_stage_idx: list[int], block_idx_offset_by_stage: list[int]
) -> None | str:
    """Converts keys associated with the image encoder component of the model"""

    # Capture oddly placed hi-res convolution layers on the mask decoder
    if key.startswith("sam_mask_decoder.conv_s0"):
        return key.replace("sam_mask_decoder.conv_s0", "proj_x4")
    if key.startswith("sam_mask_decoder.conv_s1"):
        return key.replace("sam_mask_decoder.conv_s1", "proj_x2")

    # Capture 'no_mem_embed' which originally belonged to parent SAM model
    if key == "no_mem_embed":
        return key

    # Re-arrange/name positional encoding weights to a different model component
    if key == "image_encoder.trunk.pos_embed":
        return "posenc.base_embedding"
    if key == "image_encoder.trunk.pos_embed_window":
        return "posenc.base_window_tile"

    # Bail on non-image encoder keys
    if not key.startswith("image_encoder"):
        return None

    # Remove prefix from all keys, since this isn't present in new implementation
    new_key = key.removeprefix("image_encoder.")

    # Patch embedding is no longer stored on the trunk
    if new_key.startswith("trunk.patch_embed"):
        return new_key.removeprefix("trunk.")

    # Re-format block keys into stage-based keys
    if new_key.startswith("trunk.blocks"):

        # Rename transformer layernorm layers
        if "norm1" in new_key:
            new_key = new_key.replace("norm1", "norm_preattn")
        if "norm2" in new_key:
            new_key = new_key.replace("norm2", "norm_premlp")

        # Fix block MLP layer indexing
        orig_block_idx = get_nth_integer(new_key, 0)
        if ".mlp.layers." in new_key:
            # block_idx = get_nth_integer(new_key, 0)
            mlp_layer_idx = get_nth_integer(new_key, 1)
            new_idx = 2 * mlp_layer_idx
            weight_or_bias = get_suffix_terms(new_key, 1)
            new_key = f"trunk.blocks.{orig_block_idx}.mlp.layers.{new_idx}.{weight_or_bias}"

        # Create new stage-block indexing (from older  'global block indexing')
        stage_idx = block_idx_to_stage_idx[orig_block_idx]
        new_block_idx = orig_block_idx - block_idx_offset_by_stage[stage_idx]
        old_prefix = f"trunk.blocks.{orig_block_idx}"
        new_prefix = f"trunk.stages.{stage_idx}.{new_block_idx}"
        new_key = new_key.replace(old_prefix, new_prefix)

        return new_key

    # Fix weird FPN convolution layers
    if "neck.convs" in new_key:
        seq_idx = get_nth_integer(new_key, 0)
        weight_or_bias = get_suffix_terms(new_key, 1)
        return f"output_projection.multires_projs.{seq_idx}.{weight_or_bias}"

    # Return all other keys with just prefix removed
    return new_key


# .....................................................................................................................


def _convert_coordencoder_keys(key: str) -> None | str:

    # Handle position embedding
    if key.startswith("sam_prompt_encoder.pe_layer"):
        return "gaussian_matrix"

    return None


# .....................................................................................................................


def _convert_promptencoder_keys(key: str) -> None | str:

    # Bail on non-prompt encoder keys
    if not key.startswith("sam_prompt_encoder"):
        return None

    # Handle special 'not a point' token (used for padding prompt)
    if key.startswith("sam_prompt_encoder.not_a_point_embed"):
        return "point_encoder.not_a_point_embed"

    # Handle the point & bounding box embeddings
    if key.startswith("sam_prompt_encoder.point_embeddings"):

        # The point embeddings are stored in a single matrix
        # -> Rows of the matrix correspond to different point encodings
        # -> This LUT maps the row index of the original weights to the new named weights
        point_type_lut = {
            0: "bg_embed",
            1: "fg_embed",
            2: "tl_embed",
            3: "br_embed",
        }

        # Map old indexing to new individually named parameters
        pt_idx = get_nth_integer(key, 0)
        is_box_point = pt_idx > 1
        enc_type = "box_encoder" if is_box_point else "point_encoder"
        pt_type = point_type_lut[pt_idx]
        new_key = f"{enc_type}.{pt_type}"

        return new_key

    return None


# .....................................................................................................................


def _convert_maskdecoder_keys(key: str) -> None | str:

    # Convert some weights that belonged to the prompt encoder originally over to the mask decoder
    if key.startswith("sam_prompt_encoder"):

        # Handle special 'no mask' token
        if key.startswith("sam_prompt_encoder.no_mask_embed"):
            return "maskhint_encoder.no_mask_embed"

        # Handle downscaling layers for mask prompts
        if key.startswith("sam_prompt_encoder.mask_downscaling"):
            return key.replace("sam_prompt_encoder.mask_downscaling", "maskhint_encoder.downscaler")

    # Bail on non-decoder keys
    if not key.startswith("sam_mask_decoder"):
        return None

    if key.startswith("sam_mask_decoder.obj_score_token"):
        return "cls_obj_token"

    if key.startswith("sam_mask_decoder.iou_token"):
        return "cls_iou_token"

    if key.startswith("sam_mask_decoder.mask_tokens"):
        return "cls_mask_tokens"

    if key.startswith("sam_mask_decoder.transformer"):

        # Remove mask_decoder prefix from all transformer keys
        new_key = key.replace("sam_mask_decoder.", "")

        # All transformer key changes are handled using find-and-replace lookup
        find_and_replace_lut = {
            "self_attn": "prompt_selfattn.attn",
            "cross_attn_token_to_image": "prompt_crossattn.attn",
            "cross_attn_image_to_token": "image_crossattn.attn",
            "mlp.layers.0": "prompt_mlpnorm.mlp.0",
            "mlp.layers.1": "prompt_mlpnorm.mlp.2",
            "norm1": "prompt_selfattn.norm",
            "norm2": "prompt_crossattn.norm",
            "norm3": "prompt_mlpnorm.norm",
            "norm4": "image_crossattn.norm",
            "final_attn_token_to_image": "final_prompt_crossattn.attn",
            "norm_final_attn": "final_prompt_crossattn.norm",
        }
        has_attn_match, targ_str, match_str = find_match_by_lut(new_key, find_and_replace_lut)
        if has_attn_match:
            return new_key.replace(targ_str, match_str)
        return new_key

    if key.startswith("sam_mask_decoder.output_upscaling"):

        # All upscaling keys have a different prefix
        new_key = key.replace("sam_mask_decoder", "maskgen")
        # new_key = key.replace("sam_mask_decoder.output_upscaling", "maskgen.img_patch_upscaler")

        # Adjust nested key names, which were original stored in a sequential model, now stored in a separate module
        find_and_replace_lut = {
            "output_upscaling.0": "img_patch_upscaler.upscale_1",
            "output_upscaling.1": "img_patch_upscaler.norm",
            "output_upscaling.3": "img_patch_upscaler.upscale_2",
        }

        has_attn_match, targ_str, match_str = find_match_by_lut(new_key, find_and_replace_lut)
        if has_attn_match:
            return new_key.replace(targ_str, match_str)

        return new_key

    # Convert 'object score head' keys to MLP naming scheme
    if key.startswith("sam_mask_decoder.pred_obj_score_head"):
        layer_idx = get_nth_integer(key, 0)
        new_idx = 2 * layer_idx
        weight_or_bias = get_suffix_terms(key, 1)
        return f"obj_token_mlp.layers.{new_idx}.{weight_or_bias}"

    # Convert 'iou prediction head' keys to MLP naming scheme
    if key.startswith("sam_mask_decoder.iou_prediction_head"):
        layer_idx = get_nth_integer(key, 0)
        new_idx = 2 * layer_idx
        weight_or_bias = get_suffix_terms(key, 1)
        return f"iou_token_mlp.layers.{new_idx}.{weight_or_bias}"

    # Convert 'hypernetworks' keys to mask token MLPs naming scheme
    if key.startswith("sam_mask_decoder.output_hypernetworks_mlps"):
        seq_idx = get_nth_integer(key, 0)
        layer_idx = get_nth_integer(key, 1)
        new_idx = 2 * layer_idx
        weight_or_bias = get_suffix_terms(key, 1)
        return f"maskgen.mask_token_mlps.{seq_idx}.layers.{new_idx}.{weight_or_bias}"

    # Ignore conv_s0/conv_s1 layers, which now belong to the image encoder!
    if key.startswith("sam_mask_decoder.conv"):
        return None

    return None


# .....................................................................................................................


def _convert_memencoder_keys(key: str) -> None | str:

    # Update mask downsampler to sequential indexing scheme
    if key.startswith("memory_encoder.mask_downsampler.encoder"):
        return key.replace("memory_encoder.mask_downsampler.encoder", "mask_downsampler")

    # Update fuser layers to sequential indexing scheme
    if key.startswith("memory_encoder.fuser.layers"):
        return key.replace("memory_encoder.fuser.layers", "fuser")

    # Remove model name prefix
    if key.startswith("memory_encoder"):
        return key.removeprefix("memory_encoder.")

    return None


# .....................................................................................................................


def _convert_memattention_keys(key: str) -> None | str:

    # Remove model name prefix
    if key.startswith("memory_attention"):
        return key.removeprefix("memory_attention.")

    return None
