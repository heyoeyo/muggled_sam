#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from collections import Counter

from .key_regex import get_nth_integer


# ---------------------------------------------------------------------------------------------------------------------
# %% Main function


def get_model_config_from_state_dict(state_dict):

    # Check for special config that is stored in weights that are not needed, so may not be present
    imgenc_num_stages, imgenc_num_heads, imgenc_window_size = get_image_encoder_freqs_data(state_dict)

    # Generally it's not possible to determine the number of heads from weights directly
    hardcoded_maskdec_num_heads = 8
    hardcoded_txtenc_num_heads = 16
    hardcoded_samplingenc_num_heads = 8
    hardcoded_imgexmfuse_num_heads = 8
    hardcoded_exmdet_num_heads = 8
    hardcoded_exmseg_num_heads = 8

    return {
        "features_per_prompt_token": get_features_per_prompt_token(state_dict),
        "features_per_decoder_token": get_mask_decoder_features_per_token(state_dict),
        "features_per_memory_token": get_features_per_memory_token(state_dict),
        "features_per_detection_token": get_exemplar_detector_features_per_token(state_dict),
        "imgencoder_features": get_image_encoder_features_per_token(state_dict),
        "imgencoder_num_stages": imgenc_num_stages,
        "imgencoder_num_blocks": get_image_encoder_block_count(state_dict),
        "imgencoder_num_heads": imgenc_num_heads,
        "imgencoder_patch_size_px": get_patch_size_px(state_dict),
        "imgencoder_posenc_tile_hw": get_image_encoder_posembed_hw(state_dict),
        "imgencoder_window_size": imgenc_window_size,
        "maskdecoder_num_blocks": get_mask_decoder_block_count(state_dict),
        "maskdecoder_num_heads": hardcoded_maskdec_num_heads,
        "maskdecoder_num_mask_tokens": get_num_output_mask_tokens(state_dict),
        "memencoder_num_downsample_layers": get_memory_encoder_downsample_layer_count(state_dict),
        "memencoder_num_mixer_layers": get_memory_encoder_mixer_layer_count(state_dict),
        "memimgfusion_num_fusion_layers": get_memory_image_fusion_layer_count(state_dict),
        "txtencoder_features": get_text_encoder_features_per_token(state_dict),
        "txtencoder_num_blocks": get_text_encoder_block_count(state_dict),
        "txtencoder_num_heads": hardcoded_txtenc_num_heads,
        "txtencoder_vocab_size": get_text_encoder_vocab_size(state_dict),
        "samplingenc_num_blocks": get_sampling_encoder_block_count(state_dict),
        "samplingenc_num_heads": hardcoded_samplingenc_num_heads,
        "imgexmfuse_num_blocks": get_image_exemplar_fusion_block_count(state_dict),
        "imgexmfuse_num_heads": hardcoded_imgexmfuse_num_heads,
        "exmdetector_num_detections": get_exemplar_detector_num_detections(state_dict),
        "exmdetector_num_blocks": get_exemplar_detector_block_count(state_dict),
        "exmdetector_num_heads": hardcoded_exmdet_num_heads,
        "exmsegment_num_heads": hardcoded_exmseg_num_heads,
    }


# ---------------------------------------------------------------------------------------------------------------------
# %% Image encoder functions


def _find_num_seq_elems(state_dict: dict, key_prefix: str) -> int:
    """
    Helper function used to find the number of sequence elements in a model
    Keys are assume to have a shared prefix like:
        'model.components.blocks.0.linear.weight',
        'model.components.blocks.0.attn.qkv.weight',
        'model.components.blocks.0.attn.proj.bias'
        ... etc
    The shared prefix would be: 'model.components.blocks'

    This function reports the maximum index + 1 (as this is typically the count of sequence elements)
    """

    # Get indexing of entry with the target prefix
    is_target_key = lambda key: key.startswith(key_prefix)
    idx_list = [get_nth_integer(key, 0) for key in state_dict.keys() if is_target_key(key)]
    if len(idx_list) == 0:
        return 0

    # Take the max index (+1 due to zero-indexing) to determine count
    return int(1 + max(idx_list))


def get_image_encoder_block_count(state_dict):
    """
    State dict contains keys like:
        'detector.backbone.vision_backbone.trunk.blocks.0.norm1.weight'
        'detector.backbone.vision_backbone.trunk.blocks.7.attn.qkv.weight',
        'detector.backbone.vision_backbone.trunk.blocks.29.mlp.fc1.weight',
        ... etc
    This returns number of blocks
    """

    num_blocks = _find_num_seq_elems(state_dict, "detector.backbone.vision_backbone.trunk.blocks")
    assert num_blocks > 0, "Couldn't find any image encoder blocks..."
    return num_blocks


def get_image_encoder_features_per_token(state_dict):
    """
    The state dict is expected to contain weights for the patch embedding layer.
    This is a convolutional layer responsible for 'chopping up the image' into image patch tokens.
    The number of output channels from this convolution layer determines the number of features per token
    for the image encoder (which immediately follows the patch embedding).
    """

    # Make sure there is a patch embedding key in the given state dict
    target_key = "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight"
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

    # Make sure there is a patch embedding key in the given state dict
    target_key = "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight"
    assert target_key in state_dict.keys(), f"Error determining patch size! Couldn't find key: {target_key}"

    # Expecting weights with shape: Fx3xPxP
    # -> F is num features per token in transformer
    # -> 3 is expected channel count of input RGB images
    # -> P is patch size in pixels
    _, _, _, patch_size_px = state_dict[target_key].shape

    return int(patch_size_px)


# .....................................................................................................................


def get_image_encoder_posembed_hw(state_dict):

    target_key = "detector.backbone.vision_backbone.trunk.pos_embed"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining tiled position embedding size! Couldn't find key: {target_key}"

    # Expecting weights with shape: 1x(1+H*W)xF
    # -> H & W are what we're after. We assume that are equal. The '1+' part is due to an unused cls token
    # -> F is num features per token in transformer
    _, num_tokens, _ = state_dict[target_key].shape
    hw_product = num_tokens - 1
    size = int(round(hw_product**0.5))

    return (size, size)


# .....................................................................................................................


def get_image_encoder_freqs_data(
    state_dict,
    default_num_stages: int = 4,
    default_num_heads: int = 16,
    default_window_size: int = 24,
) -> tuple[int, int, int]:
    """
    The original SAMv3 model contains weights for layers with the name 'freqs_cis'.
    These aren't learned values and probably shouldn't be loaded from weights,
    however they include information that can be used to infer the number of stages,
    the number of heads and the window sizing used in the image encoder.

    The 'freqs' weights have a shape of: NxC
    -> Where N is the number of tokens processed during attention calculations
    -> C is the channel count of the tokens

    These values are influenced by whether a block is using windowed or global
    attention. For example, with windowed attention the number of tokens is
    greatly reduced (576 originally) compared to with global attention (5184 originally).

    The number of 'large' token counts indicates the number of stages.
    The window sizing is found as the square root of the smaller token count

    The channel count can be used to infer the number of heads, as the 'freqs'
    values are actually used after tokens are split into a multi-headed shape.
    Specifically, the channel count is given by:
        C = 2 * (F / num_heads) / 4
        -> where F is features per image token (1024 default)
        num_heads = F / (2*C)

    All this being said, the inclusion of the freqs_cis weights is actually
    problematic for how the model works! So it's possible they may be
    missing in any releases/fine-tunes that drop them. So this function
    has to have a fallback option in case these weights aren't present.

    Returns:
        num_stages, num_heads, window_size
    """

    # Default values if freqs data is missing
    num_stages = default_num_stages
    num_heads = default_num_heads
    window_size = default_window_size

    # Look for 'freqs_cis' keys (which aren't actually needed, so may not be available)
    target_key_piece = "attn.freqs_cis"
    freqs_num_tokens = [val.shape[0] for key, val in state_dict.items() if target_key_piece in key]
    if len(freqs_num_tokens) == 0:
        print(
            "",
            "Warning:",
            "Couldn't find 'freqs_cis' weights in model!",
            "Cannot determine stage count, heads or window size of image encoder.",
            f"Will assume defaults: {num_stages}, {num_heads}, {window_size} respectively",
            sep="\n",
        )
        return num_stages, num_heads, window_size

    # Figure out min/max 'num tokens' since these correspond to windowed & global blocks
    count_of_num_tokens = Counter(freqs_num_tokens)
    windowed_num_tokens = min(count_of_num_tokens)
    global_num_tokens = max(count_of_num_tokens)

    # Compute config values from freqs_cis results
    num_stages = max(count_of_num_tokens[global_num_tokens], 1)
    window_size = int(round(windowed_num_tokens**0.5))

    return num_stages, num_heads, window_size


# ---------------------------------------------------------------------------------------------------------------------
# %% Mask decoder functions


def get_features_per_prompt_token(state_dict):
    # Make sure the target key is in the given state dict
    target_key = "tracker.sam_prompt_encoder.not_a_point_embed.weight"
    assert target_key in state_dict.keys(), f"Error determining features per prompt token! Couldn't find: {target_key}"

    # Expecting weights of shape: 1xF
    # -> F is the features per prompt token
    _, features_per_prompt_token = state_dict[target_key].shape

    return int(features_per_prompt_token)


# .....................................................................................................................


def get_num_output_mask_tokens(state_dict):

    target_key = "tracker.sam_mask_decoder.mask_tokens.weight"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining the number of output mask tokens! Couldn't find key: {target_key}"

    num_mask_tokens, _ = state_dict[target_key].shape

    return int(num_mask_tokens)


# .....................................................................................................................


def get_mask_decoder_features_per_token(state_dict):

    target_key = "tracker.sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.q_proj.weight"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining mask decoder feature count! Couldn't find key: {target_key}"

    downsample_channels, _ = state_dict[target_key].shape

    return int(downsample_channels)


# .....................................................................................................................


def get_mask_decoder_block_count(state_dict):
    num_layers = _find_num_seq_elems(state_dict, "tracker.sam_mask_decoder.transformer.layers")
    assert num_layers > 0, "Couldn't find any mask decoder layers..."
    return num_layers


# ---------------------------------------------------------------------------------------------------------------------
# %% Memory functions


def get_features_per_memory_token(state_dict):

    target_key = "tracker.maskmem_backbone.out_proj.weight"
    assert target_key in state_dict.keys(), f"Error determining memory token features! Couldn't find key: {target_key}"

    # Expecting weights with shape: FxPx1x1
    # -> F is num features per memory token
    # -> P is num features per prompt token
    features_per_memory_token, _, _, _ = state_dict[target_key].shape

    return int(features_per_memory_token)


# .....................................................................................................................


def get_memory_encoder_downsample_layer_count(state_dict):
    """
    This one is a bit weird. The layers look like:
        'tracker.maskmem_backbone.mask_downsampler.encoder.0.weight'
        'tracker.maskmem_backbone.mask_downsampler.encoder.0.bias'
        'tracker.maskmem_backbone.mask_downsampler.encoder.1.weight'
        'tracker.maskmem_backbone.mask_downsampler.encoder.1.bias'
        ...
        'tracker.maskmem_backbone.mask_downsampler.encoder.12.bias'
        etc.

    Each index appears twice, but every third entry is missing,
    so the original index sequence looks like:
        0,0,1,1,3,3,4,4,6,6,7,7,9,9,10,10,12,12

    Each sequential index pair is considered one layer.
    For example (0,0,1,1) and (3,3,4,4) are each a layer.
    The last two entries (12,12) are actually a projection module and are not really part
    of the layering sequence. So in the given example, the number of layers is 4
    """

    # Get indexing of entry with the target prefix
    is_target_key = lambda key: key.startswith("tracker.maskmem_backbone.mask_downsampler.encoder")
    idx_list = [key for key in state_dict.keys() if is_target_key(key)]

    # Count layers by removing final 2 projection entries and account for doubling of entries per layer
    target_layer_count = max(len(idx_list) - 2, 0)
    assert target_layer_count > 0, "Couldn't find any memory encoder downsample layers..."
    num_layers = target_layer_count // 4
    return num_layers


# .....................................................................................................................


def get_memory_encoder_mixer_layer_count(state_dict):
    num_layers = _find_num_seq_elems(state_dict, "tracker.maskmem_backbone.fuser.layers")
    assert num_layers > 0, "Couldn't find any memory encoder mixer layers..."
    return num_layers


# .....................................................................................................................


def get_memory_image_fusion_layer_count(state_dict):
    num_layers = _find_num_seq_elems(state_dict, "tracker.transformer.encoder.layers")
    assert num_layers > 0, "Couldn't find any memory-image fusion layers..."
    return num_layers


# ---------------------------------------------------------------------------------------------------------------------
# %% Text encoder functions


def get_text_encoder_features_per_token(state_dict):

    target_key = "detector.backbone.language_backbone.encoder.positional_embedding"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining text encoder feature count! Couldn't find key: {target_key}"

    _, textenc_feature_count = state_dict[target_key].shape

    return int(textenc_feature_count)


# .....................................................................................................................


def get_text_encoder_vocab_size(state_dict):

    target_key = "detector.backbone.language_backbone.encoder.token_embedding.weight"
    assert (
        target_key in state_dict.keys()
    ), f"Error determining text encoder vocab size! Couldn't find key: {target_key}"

    vocab_size, _ = state_dict[target_key].shape

    return int(vocab_size)


# .....................................................................................................................


def get_text_encoder_block_count(state_dict):
    num_blocks = _find_num_seq_elems(state_dict, "detector.backbone.language_backbone.encoder.transformer.resblocks")
    assert num_blocks > 0, "Couldn't find any text encoder blocks..."
    return num_blocks


# ---------------------------------------------------------------------------------------------------------------------
# %% Exemplar functions


def get_sampling_encoder_block_count(state_dict):
    num_blocks = _find_num_seq_elems(state_dict, "detector.geometry_encoder.encode.")
    assert num_blocks > 0, "Couldn't find any sampling encoder blocks..."
    return num_blocks


# .....................................................................................................................


def get_image_exemplar_fusion_block_count(state_dict):
    num_layers = _find_num_seq_elems(state_dict, "detector.transformer.encoder.layers")
    assert num_layers > 0, "Couldn't find any image-exemplar fusion layers..."
    return num_layers


# .....................................................................................................................


def get_exemplar_detector_features_per_token(state_dict):

    target_key = "detector.transformer.decoder.query_embed.weight"
    assert target_key in state_dict.keys(), f"Error determining detector feature count! Couldn't find key: {target_key}"

    _, exmdet_feature_count = state_dict[target_key].shape

    return int(exmdet_feature_count)


# .....................................................................................................................


def get_exemplar_detector_num_detections(state_dict):

    target_key = "detector.transformer.decoder.query_embed.weight"
    assert target_key in state_dict.keys(), f"Error determining detector query count! Couldn't find key: {target_key}"

    num_queries, _ = state_dict[target_key].shape

    return int(num_queries)


# .....................................................................................................................


def get_exemplar_detector_block_count(state_dict):
    num_layers = _find_num_seq_elems(state_dict, "detector.transformer.decoder.layers")
    assert num_layers > 0, "Couldn't find any exemplar detector blocks..."
    return num_layers
