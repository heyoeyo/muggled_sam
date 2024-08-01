#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from .key_regex import get_nth_integer, get_suffix_terms, replace_prefix, find_match_by_lut


# ---------------------------------------------------------------------------------------------------------------------
# %% Main function


def convert_state_dict_keys(config_dict: dict, original_state_dict: dict) -> dict[str, dict]:
    """
    Function which converts original Segment-Anything model weights
    into the new format needed by the model implementation in this repo (MuggledSAM)
    (layer names are renamed to make the model easier to understand, some are deleted or re-arranged)

    Returns:
        new_state_dict

    Note: The new state dict has keys corresponding the the model components:
        "imgencoder", "coordencoder", "promptencoder", "maskdecoder"
    """

    # Grab model config info for properly interpreting model structure
    num_imgenc_stages = config_dict["num_encoder_stages"]
    num_imgenc_blocks = config_dict["num_encoder_blocks"]
    imgenc_blocks_per_stage = num_imgenc_blocks // num_imgenc_stages

    # Allocate storage for state dict of each (new) model component
    imgenc_sd = {}
    coordencoder_sd = {}
    promptencoder_sd = {}
    maskdecoder_sd = {}

    # Loop over all midas state dict keys and convert them to new formatting
    found_key = lambda key: key is not None
    for orig_key, orig_data in original_state_dict.items():

        # For sanity, make sure the key is definitely a string
        orig_key = str(orig_key)

        new_key = _convert_imgenc_keys(orig_key, imgenc_blocks_per_stage)
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
            layernorm_key_hints = ("downscaler.1", "downscaler.4", "img_patch_upscaler.1")
            mod_data = _reshape_layernorm2d(new_key, orig_data, *layernorm_key_hints)

            maskdecoder_sd[new_key] = mod_data
            continue

    # Bundle new state dict model components together for easier handling
    new_state_dict = {
        "imgencoder": imgenc_sd,
        "coordencoder": coordencoder_sd,
        "promptencoder": promptencoder_sd,
        "maskdecoder": maskdecoder_sd,
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
            res = data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            return res

    return data

# .....................................................................................................................

def _convert_imgenc_keys(key: str, blocks_per_stage: int) -> None | str:
    """
    Converts keys associated with the image encoder component of the model
    Takes care of:
        - position embeddings
        - transformer blocks (including re-structuring as stages)
        - output projection ('neck') layers
    """

    # Bail on non-image encoder keys
    if not key.startswith("image_encoder"):
        return None

    # Handle patch embeddings
    is_patch_embed = key.startswith("image_encoder.patch_embed")
    if is_patch_embed:
        return key.removeprefix("image_encoder.")

    # Handle position embedding
    if key.startswith("image_encoder.pos_embed"):
        return key.replace("image_encoder.pos_embed", "posenc.base_embedding")

    # Handle output ('neck') layers
    if key.startswith("image_encoder.neck"):
        return key.replace("image_encoder.neck", "output_projection")

    # Handle transformer blocks (bulk of the model)
    if key.startswith("image_encoder.blocks"):

        # We need the block index to figure out which stage we're in
        # (the original model does not have 'stages')
        orig_block_idx = get_nth_integer(key, 0)
        stage_idx = orig_block_idx // blocks_per_stage

        # Figure out if the block is a global attention block (otherwise windowed attention)
        block_idx_within_stage = orig_block_idx % blocks_per_stage
        global_block_idx_within_stage = blocks_per_stage - 1
        is_global_block = block_idx_within_stage == global_block_idx_within_stage

        # Update key to account for stage indexing
        # image_encoder.blocks.6.norm1.weight -> stages.2.windowed_attn_blocks.0.norm1.weight
        # image_encoder.blocks.11.mlp.lin2.bias -> stages.3.global_attn_block.mlp.layers.2.bias
        target_prefix = "image_encoder.blocks.#"
        global_prefix = f"stages.{stage_idx}.global_attn_block"
        windowed_prefix = f"stages.{stage_idx}.windowed_attn_blocks.{block_idx_within_stage}"
        new_key = replace_prefix(key, target_prefix, global_prefix if is_global_block else windowed_prefix)

        # Further handle updates to specific layer names
        if "rel_pos" in new_key:
            new_key = new_key.replace("rel_pos_h", "relpos.rel_pos_h")
            new_key = new_key.replace("rel_pos_w", "relpos.rel_pos_w")
            return new_key

        # Handle mlp linear layers
        if "mlp.lin" in new_key:
            new_key = new_key.replace("lin1", "layers.0")
            new_key = new_key.replace("lin2", "layers.2")
            return new_key

        return new_key

    return None


# .....................................................................................................................


def _convert_coordencoder_keys(key: str) -> None | str:

    # Handle position embedding
    if key.startswith("prompt_encoder.pe_layer"):
        return "gaussian_matrix"

    return None


# .....................................................................................................................


def _convert_promptencoder_keys(key: str) -> None | str:

    # Bail on non-prompt encoder keys
    if not key.startswith("prompt_encoder"):
        return None

    # Handle special 'not a point' token (used for padding prompt)
    if key.startswith("prompt_encoder.not_a_point_embed"):
        return "point_encoder.not_a_point_embed"

    # Handle the point & bounding box embeddings
    if key.startswith("prompt_encoder.point_embeddings"):

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
    if key.startswith("prompt_encoder"):

        # Handle special 'no mask' token
        if key.startswith("prompt_encoder.no_mask_embed"):
            return "maskhint_encoder.no_mask_embed"

        # Handle downscaling layers for mask prompts
        if key.startswith("prompt_encoder.mask_downscaling"):
            return key.replace("prompt_encoder.mask_downscaling", "maskhint_encoder.downscaler")

    # Bail on non-decoder keys
    if not key.startswith("mask_decoder"):
        return None

    if key.startswith("mask_decoder.iou_token"):
        return "cls_iou_token"

    if key.startswith("mask_decoder.mask_tokens"):
        return "cls_mask_tokens"

    if key.startswith("mask_decoder.transformer"):

        # Remove mask_decoder prefix from all transformer keys
        new_key = key.replace("mask_decoder.", "")

        # All transformer key changes are handled using find-and-replace lookup
        find_and_replace_lut = {
            "self_attn": "prompt_selfattn.attn",
            "cross_attn_token_to_image": "prompt_crossattn.attn",
            "cross_attn_image_to_token": "image_crossattn.attn",
            "mlp.lin1": "prompt_mlpnorm.mlp.0",
            "mlp.lin2": "prompt_mlpnorm.mlp.2",
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

    if key.startswith("mask_decoder.output_upscaling"):
        return key.replace("mask_decoder.output_upscaling", "maskgen.img_patch_upscaler")

    if key.startswith("mask_decoder.iou_prediction_head"):
        layer_idx = get_nth_integer(key, 0)
        new_idx = 2 * layer_idx
        weight_or_bias = get_suffix_terms(key, 1)
        return f"iou_token_mlp.layers.{new_idx}.{weight_or_bias}"

    if key.startswith("mask_decoder.output_hypernetworks_mlps"):
        seq_idx = get_nth_integer(key, 0)
        layer_idx = get_nth_integer(key, 1)
        new_idx = 2 * layer_idx
        weight_or_bias = get_suffix_terms(key, 1)
        return f"maskgen.mask_token_mlps.{seq_idx}.layers.{new_idx}.{weight_or_bias}"

    return None
