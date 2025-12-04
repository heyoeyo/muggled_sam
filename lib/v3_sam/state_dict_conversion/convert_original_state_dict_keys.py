#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from enum import StrEnum

from .key_regex import get_nth_integer, get_suffix_terms, replace_prefix, find_match_by_lut


# ---------------------------------------------------------------------------------------------------------------------
# %% Stage type


class SAM3StageType(StrEnum):
    image_encoder = "imgenc"
    image_projection = "imgproj"
    coordinate_encoder = "coordenc"
    prompt_encoder = "promptenc"
    mask_decoder = "maskdec"
    memory_encoder = "memenc"
    memory_image_fusion = "memimgfusion"
    text_encoder = "textenc"
    geometry_encoder = "geoenc"
    detector_fusion = "detfusion"
    detector_segmentation = "detseg"
    not_used = "not_used"
    unknown = "unknown"


# ---------------------------------------------------------------------------------------------------------------------
# %% Main function


def convert_state_dict_keys(
    config_dict: dict, original_state_dict: dict, warn_missing: bool = True, report_unused: bool = False
) -> dict[SAM3StageType, dict]:
    """
    Function which converts original SAM3 model weights into the new format
    needed by the model implementation in this repo (MuggledSAM).
    (layer names are renamed to make the model easier to understand, some are deleted or re-arranged)

    Returns:
        new_state_dict

    Note: The new state dict has keys corresponding the the model components:
        "image_encoder", "image_projection", "coordinate_encoder",
        "prompt_encoder", "memory_encoder", "memoryencoder", "memoryfusion",
        "text_encoder", "geometry_encoder", "detector_fusion", "detector_segmentation"
    """

    # Grab model config info for properly interpreting model structure
    num_imgenc_stages = config_dict["imgencoder_num_stages"]
    num_imgenc_blocks = config_dict["imgencoder_num_blocks"]
    imgenc_blocks_per_stage = num_imgenc_blocks // num_imgenc_stages

    # Allocate storage for state dict of each (newly organized) model component
    imgenc_sd = {}
    imgproj_sd = {}
    coordencoder_sd = {}
    promptencoder_sd = {}
    maskdecoder_sd = {}
    memencoder_sd = {}
    memimgfusion_sd = {}

    # Loop over all state dict keys and convert them to new formatting
    for orig_key, orig_data in original_state_dict.items():

        # For sanity, make sure the key is definitely a string
        orig_key = str(orig_key)
        sam_stage_type = get_stage_type(orig_key)
        new_key, new_data = orig_key, orig_data

        if sam_stage_type == SAM3StageType.image_encoder:

            # Skip unused keys
            new_key = _convert_imgenc_keys(orig_key, imgenc_blocks_per_stage)
            if new_key is None:
                continue

            # Reshape base-position encoding
            if new_key.startswith("posenc.tile_embedding"):
                # Original is shape: 1x(N+1)xC (N is number tokens + 1 cls token, C is features per token)
                # -> Want in image format: 1xCxHxW, without the cls token (which isn't used)
                new_data = new_data[0, 1:, :].reshape(1, 24, 24, 1024).permute(0, 3, 1, 2)

            # Store new key/weight data
            imgenc_sd[new_key] = new_data

        elif sam_stage_type == SAM3StageType.image_projection:

            # Skip unused keys
            new_key = _convert_imgproj_keys(orig_key)
            if new_key is None:
                continue
            imgproj_sd[new_key] = new_data

        elif sam_stage_type == SAM3StageType.coordinate_encoder:

            # Skip unused keys
            new_key = _convert_coordencoder_keys(orig_key)
            if new_key is None:
                continue
            coordencoder_sd[new_key] = new_data

        elif sam_stage_type == SAM3StageType.prompt_encoder:

            # Skip unused keys
            new_key = _convert_promptencoder_keys(orig_key)
            if new_key is None:
                continue
            promptencoder_sd[new_key] = new_data

        elif sam_stage_type == SAM3StageType.mask_decoder:

            # Skip unused keys
            new_key = _convert_maskdecoder_keys(orig_key)
            if new_key is None:
                continue

            # Correct layernorm2d weight shapes
            layernorm_key_hints = ("downscaler.1", "downscaler.4", "img_patch_upscaler.norm")
            new_data = _reshape_layernorm2d(new_key, orig_data, *layernorm_key_hints)

            maskdecoder_sd[new_key] = new_data

        elif sam_stage_type == SAM3StageType.memory_encoder:

            # Skip unused keys
            new_key = _convert_memencoder_keys(orig_key)
            if new_key is None:
                continue

            # Correct layernorm2d weight shapes
            layernorm_key_hints = [f"mask_downsampler.downsample.{idx}." for idx in (1, 4, 7, 10)]
            layernorm_key_hints.extend(["channel_mixer.0.norm", "channel_mixer.1.norm"])
            new_data = _reshape_layernorm2d(new_key, orig_data, *layernorm_key_hints)

            # Convert linear 'point-wise' weights to 1x1 convolution shapes
            # -> Original shape is: DxC (D output channels, C input channels)
            # -> Want 1x1 conv shape: DxCx1x1
            if "inverted_bottleneck" in new_key and "weight" in new_key:
                new_data = new_data.unsqueeze(-1).unsqueeze(-1)

            # Add broadcasting dimensions to channel scaling term to match BxCxHxW shape
            # -> Original assumed shape: BxHxWxC, so didn't need broadcast dimensions
            # -> This is required due to changing to 1x1 convolutions!
            if "per_channel_scale" in new_key:
                new_data = new_data.reshape(1, -1, 1, 1)

            memencoder_sd[new_key] = new_data

        elif sam_stage_type == SAM3StageType.memory_image_fusion:

            # Skip unused keys
            new_key = _convert_memimgfusion_keys(orig_key)
            if new_key is None:
                continue
            memimgfusion_sd[new_key] = new_data

        elif sam_stage_type == SAM3StageType.text_encoder:
            continue
        elif sam_stage_type == SAM3StageType.geometry_encoder:
            continue
        elif sam_stage_type == SAM3StageType.detector_fusion:
            continue
        elif sam_stage_type == SAM3StageType.detector_segmentation:
            continue
        elif sam_stage_type == SAM3StageType.not_used:
            if report_unused:
                print("", f"Unused key: {orig_key}", f"     Shape: {tuple(orig_data.shape)}", sep="\n")
            pass
        elif sam_stage_type == SAM3StageType.unknown:
            print(f"Unknown key: {orig_key}")
        elif warn_missing:
            # Warn about any missed weights
            print(
                "",
                "Warning, model weight not handled on load:",
                f"    Key: '{orig_key}'",
                f"  Shape: {tuple(orig_data.shape)}",
                sep="\n",
            )

    # Bundle new state dict model components together for easier handling
    new_state_dict = {
        SAM3StageType.image_encoder: imgenc_sd,
        SAM3StageType.image_projection: imgproj_sd,
        SAM3StageType.coordinate_encoder: coordencoder_sd,
        SAM3StageType.prompt_encoder: promptencoder_sd,
        SAM3StageType.mask_decoder: maskdecoder_sd,
        SAM3StageType.memory_encoder: memencoder_sd,
        SAM3StageType.memory_image_fusion: memimgfusion_sd,
    }

    return new_state_dict


def get_stage_type(key: str) -> SAM3StageType:
    """
    Function used to figure out which part of the (mugsam)
    model implementation a given weight key belongs to. This
    helps prevent every model component trying to parse each key.
    """

    # Handle image encoder + new SAM3 components
    if key.startswith("detector"):
        if key.startswith("detector.backbone.vision"):
            # Re-direct certain backbone keys to image projection
            if "convs." in key:
                return SAM3StageType.image_projection
            return SAM3StageType.image_encoder
        elif key.startswith("detector.backbone.language"):
            return SAM3StageType.text_encoder
        elif key.startswith("detector.geometry_encoder"):
            return SAM3StageType.geometry_encoder
        elif key.startswith("detector.transformer.encoder"):
            return SAM3StageType.detector_fusion
        elif key.startswith("detector.transformer.decoder"):
            return SAM3StageType.detector_fusion
        elif key.startswith("detector.segmentation_head"):
            return SAM3StageType.detector_segmentation
        elif key.startswith("detector.dot_prod_scoring"):
            return SAM3StageType.detector_fusion

    # Handle 'old SAM2' components
    if key.startswith("tracker"):
        if key.startswith("tracker.transformer"):
            return SAM3StageType.memory_image_fusion
        elif key.startswith("tracker.maskmem_backbone"):
            return SAM3StageType.memory_encoder
        elif key.startswith("tracker.sam_prompt_encoder"):
            # Re-direct to subcomponents as needed
            if "pe_layer" in key:
                return SAM3StageType.coordinate_encoder
            if "point" in key:
                return SAM3StageType.prompt_encoder
            return SAM3StageType.mask_decoder
        elif key.startswith("tracker.sam_mask_decoder"):
            # Re-direct certain mask decoder keys to image projection
            if key.startswith("tracker.sam_mask_decoder.conv"):
                return SAM3StageType.image_projection
            return SAM3StageType.mask_decoder
        elif key.startswith("tracker.obj_ptr_proj"):
            return SAM3StageType.mask_decoder
        elif key.startswith("tracker.mask_downsample"):
            return SAM3StageType.not_used
        elif key.startswith("tracker.no_mem"):
            # Catches 'no_mem_embed', 'no_mem_pos_enc'
            return SAM3StageType.memory_image_fusion
        elif key.startswith("tracker.no_obj"):
            # Catches 'no_obj_ptr' and 'no_obj_embed_spatial'
            if key == "tracker.no_obj_ptr":
                return SAM3StageType.mask_decoder
            return SAM3StageType.memory_encoder
        elif key.startswith("tracker.obj_ptr_tpos_proj"):
            return SAM3StageType.memory_image_fusion
        elif key == "tracker.maskmem_tpos_enc":
            return SAM3StageType.memory_image_fusion

    return SAM3StageType.unknown


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
        - channel projection ('neck') layers
    """

    # Remove original prefix for simplicity
    key = key.removeprefix("detector.backbone.vision_backbone.")
    if "trunk" in key:
        key = key.removeprefix("trunk.")

    # Handle transformer blocks (bulk of the model)
    if key.startswith("blocks"):

        # Drop the stored rope position encodings (these are generated at run-time, not learned)
        if "freqs_cis" in key:
            return None

        # We need the block index to figure out which stage we're in
        # (the original model does not have 'stages')
        orig_block_idx = get_nth_integer(key, 0)
        stage_idx = orig_block_idx // blocks_per_stage

        # Figure out if the block is a global attention block (otherwise windowed attention)
        block_idx_within_stage = orig_block_idx % blocks_per_stage
        last_block_idx_within_stage = blocks_per_stage - 1
        is_global_block = block_idx_within_stage == last_block_idx_within_stage

        # Update key to account for stage indexing
        # blocks.6.norm1.weight -> stages.2.windowed_attn_blocks.0.global_attn.norm1.weight
        # blocks.11.mlp.lin2.bias -> stages.3.global_attn_block.mlp.layers.2.bias
        target_prefix = "blocks.#"
        new_prefix = f"stages.{stage_idx}.windowed_attn_blocks.{block_idx_within_stage}"
        if is_global_block:
            new_prefix = f"stages.{stage_idx}.global_attn_block"
        new_key = replace_prefix(key, target_prefix, new_prefix)

        # Rename attention block norm layers to be more descriptive
        if ".norm" in new_key:
            new_key = new_key.replace("norm1", "norm_preattn")
            new_key = new_key.replace("norm2", "norm_premlp")
            return new_key

        # Handle mlp linear layer re-structuring
        if "mlp.fc" in new_key:
            new_key = new_key.replace("fc1", "layers.0")
            new_key = new_key.replace("fc2", "layers.2")
            return new_key

        return new_key

    # Handle patch embeddings
    if key.startswith("patch_embed"):
        return key

    # Handle position embedding
    if key.startswith("pos_embed"):
        return key.replace("pos_embed", "posenc.tile_embedding_bhwc")

    # Handle layernorm prior to transformer blocks
    if key.startswith("ln_pre"):
        return key.replace("ln_pre", "pre_layernorm")

    # Handle special post-transformer up-/down-scaling layers
    # (e.g. ...vision_backbone.convs.3.conv_3x3.bias or ...vision_backbone.sam2_convs.0.dconv_2x2_0.weight)
    if "convs." in key:

        # Throw away final (unused) downscaling projection layer
        if "convs.3.conv" in key:
            return None

        # Swap out prefix for new component names
        new_key = key.replace("sam2_convs.", "multires_proj_sam2.")
        new_key = new_key.replace("convs.", "multires_proj_sam3.")

        # Re-map the output projection layers (e.g. 'neck' convolutions)
        find_and_replace_lut = {
            "0.dconv_2x2_0": "proj_x4.0",
            "0.dconv_2x2_1": "proj_x4.2",
            "0.conv_1x1": "proj_x4.3",
            "0.conv_3x3": "proj_x4.4",
            "1.dconv_2x2": "proj_x2.0",
            "1.conv_1x1": "proj_x2.1",
            "1.conv_3x3": "proj_x2.2",
            "2.conv_1x1": "proj_x1.0",
            "2.conv_3x3": "proj_x1.1",
        }
        is_lut_match, targ_str, match_str = find_match_by_lut(new_key, find_and_replace_lut)
        if is_lut_match:
            return new_key.replace(targ_str, match_str)

        pass

    return None


# .....................................................................................................................


def _convert_imgproj_keys(key: str) -> None | str:

    # Handle keys that are part of the mask decoder
    if key.startswith("tracker.sam_mask_decoder"):
        # -> There should only be 4 of these total (2x '.weights' and 2x '.bias')
        # ex: tracker.sam_mask_decoder.conv_s0 -> multires_proj_v2.proj_x4.5
        # ex: tracker.sam_mask_decoder.conv_s1 -> multires_proj_v2.proj_x2.3
        new_key = key.replace("tracker.sam_mask_decoder.conv_s0", "multires_proj_v2.proj_x4.5")
        new_key = new_key.replace("tracker.sam_mask_decoder.conv_s1", "multires_proj_v2.proj_x2.3")
        return new_key

    # Handle keys that are part of the detector weights
    key = key.removeprefix("detector.backbone.vision_backbone.")

    # Throw away final (unused) downscaling projection layer
    if "convs.3.conv" in key:
        return None

    # Swap out prefix for new component names
    new_key = key.replace("sam2_convs.", "multires_proj_v2.")
    new_key = new_key.replace("convs.", "multires_proj_v3.")

    # Re-map the output projection layers (e.g. 'neck' convolutions)
    find_and_replace_lut = {
        "0.dconv_2x2_0": "proj_x4.0",
        "0.dconv_2x2_1": "proj_x4.2",
        "0.conv_1x1": "proj_x4.3",
        "0.conv_3x3": "proj_x4.4",
        "1.dconv_2x2": "proj_x2.0",
        "1.conv_1x1": "proj_x2.1",
        "1.conv_3x3": "proj_x2.2",
        "2.conv_1x1": "proj_x1.0",
        "2.conv_3x3": "proj_x1.1",
    }
    is_lut_match, targ_str, match_str = find_match_by_lut(new_key, find_and_replace_lut)
    if is_lut_match:
        return new_key.replace(targ_str, match_str)

    return None


# .....................................................................................................................


def _convert_coordencoder_keys(key: str) -> None | str:

    # Handle position embedding
    if key.startswith("tracker.sam_prompt_encoder.pe_layer"):
        return "gaussian_matrix"

    return None


# .....................................................................................................................


def _convert_promptencoder_keys(key: str) -> None | str:

    # Remove original prefix for simplicity
    key = key.removeprefix("tracker.sam_prompt_encoder.")

    # Handle special 'not a point' token (used for padding prompt)
    if key.startswith("not_a_point_embed"):
        return "point_encoder.not_a_point_embed"

    # Handle the point & bounding box embeddings
    if key.startswith("point_embeddings"):

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
    if key.startswith("tracker.sam_prompt_encoder"):

        # Handle special 'no mask' token
        if "no_mask_embed" in key:
            return "maskhint_encoder.no_mask_embed"

        # Handle downscaling layers for mask prompts
        if key.startswith("tracker.sam_prompt_encoder.mask_downscaling"):
            return key.replace("tracker.sam_prompt_encoder.mask_downscaling", "maskhint_encoder.downscaler")

    # Capture object pointer weights (now part of mask decoder, instead of 'parent' model)
    if key == "tracker.no_obj_ptr":
        return "objptrgen.no_ptr"
    elif key.startswith("tracker.obj_ptr_proj"):
        layer_idx = get_nth_integer(key, 0)
        new_idx = 2 * layer_idx
        weight_or_bias = get_suffix_terms(key)
        return f"objptrgen.pointer_mlp.layers.{new_idx}.{weight_or_bias}"

    # Remove original prefix for remaining mask decoder keys
    key = key.removeprefix("tracker.sam_mask_decoder.")

    # Rename transformer layers (bulk of model)
    if key.startswith("transformer"):

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
        has_attn_match, targ_str, match_str = find_match_by_lut(key, find_and_replace_lut)
        if has_attn_match:
            return key.replace(targ_str, match_str)
        return key

    # Rename output mlp layers (another large part of the model)
    if key.startswith("output_hypernetworks_mlps"):
        seq_idx = get_nth_integer(key, 0)
        layer_idx = get_nth_integer(key, 1)
        new_idx = 2 * layer_idx
        weight_or_bias = get_suffix_terms(key, 1)
        return f"maskgen.mask_token_mlps.{seq_idx}.layers.{new_idx}.{weight_or_bias}"

    # Rename built-in tokens
    if key.startswith("obj_score_token"):
        return "cls_obj_token"
    if key.startswith("iou_token"):
        return "cls_iou_token"
    if key.startswith("mask_tokens"):
        return "cls_mask_tokens"

    if key.startswith("output_upscaling"):

        # Adjust nested key names, which were originally stored in a sequential model, now stored in a separate module
        find_and_replace_lut = {
            "output_upscaling.0": "maskgen.img_patch_upscaler.upscale_1",
            "output_upscaling.1": "maskgen.img_patch_upscaler.norm",
            "output_upscaling.3": "maskgen.img_patch_upscaler.upscale_2",
        }

        has_attn_match, targ_str, match_str = find_match_by_lut(key, find_and_replace_lut)
        if has_attn_match:
            return key.replace(targ_str, match_str)

        return None

    if key.startswith("iou_prediction_head"):
        layer_idx = get_nth_integer(key, 0)
        new_idx = 2 * layer_idx
        weight_or_bias = get_suffix_terms(key, 1)
        return f"iou_token_mlp.layers.{new_idx}.{weight_or_bias}"

    # Convert 'object score head' keys to MLP naming scheme
    if key.startswith("pred_obj_score_head"):
        layer_idx = get_nth_integer(key, 0)
        new_idx = 2 * layer_idx
        weight_or_bias = get_suffix_terms(key, 1)
        return f"objptrgen.score_mlp.layers.{new_idx}.{weight_or_bias}"

    return None


# .....................................................................................................................


def _convert_memencoder_keys(key: str) -> None | str:

    # Check for 'no object embedding' which is named differently than other memory-encoder keys
    if key == "tracker.no_obj_embed_spatial":
        return "missing_obj_encoder.no_object_embed"

    # Remove original prefix for simplicity
    key = key.removeprefix("tracker.maskmem_backbone.")

    # Update mask downsampler to sequential indexing scheme
    if key.startswith("mask_downsampler.encoder"):

        # Rename downsampler layers
        new_key = key.replace("mask_downsampler.encoder", "mask_downsampler.downsample")

        # Special renaming of final 'downsample' layer, which is actually a projection layer
        if "downsample.12" in new_key:
            return new_key.replace("downsample.12", "out_proj")

        return new_key

    # Rename the image projection layer
    if key.startswith("pix_feat_proj"):
        return key.replace("pix_feat_proj", "image_proj")

    # Update fuser layers to sequential indexing scheme
    if key.startswith("fuser.layers"):

        # Rename prefix
        new_key = key.replace("fuser.layers", "channel_mixer")

        # All transformer key changes are handled using find-and-replace lookup
        find_and_replace_lut = {
            "dwconv": "per_channel_conv",
            "gamma": "per_channel_scale",
            "pwconv1": "inverted_bottleneck.0",
            "pwconv2": "inverted_bottleneck.2",
        }
        has_match, targ_str, match_str = find_match_by_lut(new_key, find_and_replace_lut)
        if has_match:
            return new_key.replace(targ_str, match_str)

        return new_key

    # Keep output projection as-is
    if key.startswith("out_proj"):
        return key

    return None


# .....................................................................................................................


def _convert_memimgfusion_keys(key: str) -> None | str:

    # Ignore position encoding when no memory is present (not used)
    if key == "tracker.no_mem_pos_enc":
        return None

    # Capture 'no_mem_embed' which originally belonged to parent SAM model
    if key == "tracker.no_mem_embed":
        return key.removeprefix("tracker.")

    # Rename frame position offset embedding
    if key == "tracker.maskmem_tpos_enc":
        return "memconcat.memposenc.base_memposenc_offsets"

    # Rename object pointer projection weights (only present on SAMv2.1)
    if key.startswith("tracker.obj_ptr_tpos_proj"):
        return key.replace("tracker.obj_ptr_tpos_proj", "memconcat.ptrposenc.pointer_pos_proj")

    # Remove original prefix for remaining 'transformer' keys
    key = key.removeprefix("tracker.transformer.encoder.")

    # Re-name the layers of the transformer (bulk of this model)
    if key.startswith("layers"):

        # Handle re-structuring of the fusion transformer layers
        find_and_replace_lut = {
            "norm1": "image_selfattn.norm",
            "norm2": "image_crossattn.norm",
            "self_attn": "image_selfattn.attn",
            "cross_attn_image": "image_crossattn.attn",
            "norm3": "image_mlp.mlp.0",
            "linear1": "image_mlp.mlp.1",
            "linear2": "image_mlp.mlp.3",
        }
        has_attn_match, targ_str, match_str = find_match_by_lut(key, find_and_replace_lut)
        if has_attn_match:
            return key.replace(targ_str, match_str)

    # Rename final norm layers for clarity
    # (these are different from the norm layers 'norm1/2/3' inside the transformer)
    if key.startswith("norm"):
        return key.replace("norm", "out_norm")

    return None
