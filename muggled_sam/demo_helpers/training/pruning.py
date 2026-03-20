#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def copy_samv3_features(
    original_sd_mugsam: dict,
    new_sd_mugsam: dict,
    original_sd: dict,
    reverse_key_lut: dict[str, str],
) -> tuple[bool, dict]:
    """
    Function used for pruning the feature count of a model, by copying
    a subset of weights from a larger original model into a small (pruned) version.
    This is only tested for SAMv3 so far!

    This is somewhat of an awkward procedure, as this function tries to maintain
    the original SAM weight naming scheme, but expects to be given muggled sam
    weights as a reference (easier to produce a correctly sized/configured model this way).

    The inputs should be:
        original_sd_mugsam:
            state dict from the 'reference' model, in muggled sam format
        new_sd_mugsam:
            state dict in muggled sam format of a model created with the
            desired (pruned) feature sizing
        original_sd:
            state dict of the original model (in original format), weights
            will be taken from here to create the output state dict
        reverse_key_lut:
            A dictionary containing mappings of mugsam weight keys to
            the original model weight keys. This comes from the mugsam
            state dict conversion functions

    Returns:
        is_pruning_ok, new_state_dict
    """

    # Copy all original weights for output, as we're going to overwrite only some of them
    assert len(original_sd_mugsam) == len(new_sd_mugsam), "Error, mismatching sizes of new/original model weights"
    out_sd = {**original_sd}

    is_ok = True
    for name, old_weight in original_sd_mugsam.items():

        # Skip mugsam config entry (will be a mismatch generally...)
        if name.startswith("config"):
            continue

        # Skip weights if there's already an exact match
        old_shape = old_weight.shape
        new_shape = new_sd_mugsam[name].shape
        if old_shape == new_shape:
            continue

        # Make sure we know which (original) weight name to overwrite
        orig_name = reverse_key_lut.get(name, None)
        if orig_name is None:
            print("Error, cannot find name of weight in original model:", name, sep="\n")
            is_ok = False
            continue

        # Check if we have a weight shape mismatch (may be able to correct it, otherwise will need warning)
        orig_shape = original_sd[orig_name].shape
        is_orig_shape_mismatch = orig_shape != old_shape

        # Decimate original weights into shape that fits new feature sizing
        num_dims = len(new_shape)
        idx_list = [slice(None)] * num_dims
        for dim_idx, (og_n, ss_n) in enumerate(zip(old_shape, new_shape)):
            is_same_size = og_n == ss_n
            idx_array = torch.arange(ss_n) if is_same_size else torch.linspace(0, og_n - 1, ss_n).round().int()
            array_shape = [ss_n if idx == dim_idx else 1 for idx in range(num_dims)]
            idx_list[dim_idx] = idx_array.view(array_shape)
        new_weights = original_sd_mugsam[name][tuple(idx_list)]
        assert new_weights.shape == new_shape, f"Shape mismatch: {new_shape} vs. {new_weights.shape}"

        # Special handling of tile position encoding, which uses a BxNxC weight shape in original model
        # -> It also has an unused 'cls' token in the 0th position, which needs to be added back in for compatability
        if name.endswith("tile_embedding_bhwc"):
            _, num_tile_feats, tile_h, tile_w = new_weights.shape
            new_weights = new_weights.reshape(1, num_tile_feats, tile_h * tile_w).permute(0, 2, 1).contiguous()
            fake_cls = torch.zeros((1, 1, num_tile_feats), dtype=new_weights.dtype, device=new_weights.device)
            new_weights = torch.concat((fake_cls, new_weights), dim=1)
            is_orig_shape_mismatch = False

        # Don't store results if shape isn't matched to original model (won't load properly)
        if is_orig_shape_mismatch:
            print("Re-shaping error! Cannot handle weight correctly, skipping...")
            print("->", name)
            is_ok = False
            continue
        out_sd[orig_name] = new_weights

    # Special handling of 'text_projection' weight that isn't used in the original model (not stored in mugsam)
    # -> The unused weight has a shape of: (txt_features, 'out_dim') where out_dim is not used elsewhere
    # -> We can copy the out_dim from the original (unused) weight and get the new feature count from the vocab embed.
    unused_txt_proj_key = "detector.backbone.language_backbone.encoder.text_projection"
    orig_txtproj = out_sd[unused_txt_proj_key]
    _, new_width = out_sd["detector.backbone.language_backbone.encoder.token_embedding.weight"].shape
    orig_width, orig_outdim = orig_txtproj.shape
    if orig_width != new_width:
        out_sd[unused_txt_proj_key] = torch.zeros(
            (new_width, orig_outdim), dtype=orig_txtproj.dtype, device=orig_txtproj.device
        )

    # Special handling of ViT-neck 4th convolution layers, which aren't used (not stored in mugsam)
    # -> The unused 4th convs have the same shape as the 3rd layer convs
    ref_conv_key = "detector.backbone.vision_backbone.convs.2.conv_1x1.weight"
    unused_sam2_conv_key = "detector.backbone.vision_backbone.sam2_convs.3.conv_1x1.weight"
    unused_sam3_conv_key = "detector.backbone.vision_backbone.convs.3.conv_1x1.weight"
    new_weight = out_sd[ref_conv_key] * 0.0
    old_weight = out_sd[unused_sam2_conv_key]
    if old_weight.shape != new_weight.shape:
        out_sd[unused_sam2_conv_key] = new_weight.clone()
        out_sd[unused_sam3_conv_key] = new_weight.clone()

    return is_ok, out_sd
