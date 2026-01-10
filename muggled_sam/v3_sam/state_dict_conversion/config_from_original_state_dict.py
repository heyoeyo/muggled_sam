#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports


# ---------------------------------------------------------------------------------------------------------------------
# %% Main function


def get_model_config_from_state_dict(state_dict):

    # Hard-coded for now, to get implementation working
    # -> Also, SAMv3 has only 1 model size configuration (as of Dec 2025)

    return {
        "features_per_prompt_token": 256,
        "features_per_decoder_token": 128,
        "features_per_memory_token": 64,
        "features_per_detection_token": 256,
        "imgencoder_features": 1024,
        "imgencoder_num_stages": 4,
        "imgencoder_num_blocks": 32,
        "imgencoder_num_heads": 16,
        "imgencoder_patch_size_px": 14,
        "imgencoder_posenc_tile_hw": (24, 24),
        "imgencoder_window_size": 24,
        "maskdecoder_num_blocks": 2,
        "maskdecoder_num_heads": 8,
        "maskdecoder_num_mask_tokens": 4,
        "memencoder_num_downsample_layers": 4,
        "memencoder_num_mixer_layers": 2,
        "memimgfusion_num_fusion_layers": 4,
        "txtencoder_features": 1024,
        "txtencoder_num_blocks": 24,
        "txtencoder_num_heads": 16,
        "txtencoder_vocab_size": 49408,
        "samplingenc_num_blocks": 3,
        "samplingenc_num_heads": 8,
        "imgexmfuse_num_blocks": 6,
        "imgexmfuse_num_heads": 8,
        "exmdetector_num_detections": 200,
        "exmdetector_num_blocks": 6,
        "exmdetector_num_heads": 8,
        "exmsegment_num_heads": 8,
    }
