# usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

# Needed to make this script work from outside the root project folder (without requiring install)
try:
    import muggled_sam  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "muggled_sam" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to muggled_sam folder!")

import argparse
from pathlib import Path
from collections import defaultdict
import json

import torch

from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import ask_for_model_path_if_missing, select_from_options
from muggled_sam.demo_helpers.text_input import confirm_prompt
from muggled_sam.demo_helpers.training.default_data import make_default_image_encoder_block_mapping, save_unnested_json
from muggled_sam.v3_sam.state_dict_conversion.config_from_original_state_dict import get_model_config_from_state_dict

# Only used for feature pruning
from muggled_sam.demo_helpers.training.pruning import copy_samv3_features
from muggled_sam.v3_sam.make_sam_v3 import make_sam_v3, make_samv3_from_original_state_dict
from muggled_sam.v3_sam.state_dict_conversion.convert_original_state_dict_keys import convert_state_dict_keys


# ---------------------------------------------------------------------------------------------------------------------
# %% Script args

# Set argparse defaults (useful for hard-coding overrides)
default_model_path = None
default_mapping_path = "block_mappings_image_encoder.json"

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to prune the image encoder of a SAMv3 model")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Path to SAMv3 model weights")
parser.add_argument(
    "-map",
    "--mapping_path",
    default=default_mapping_path,
    type=str,
    help="Path to a json file containing block mappings (run script to generate initial file)",
)
parser.add_argument("-n", "--map_name", default=None, type=str, help="Name of block mapping to use")
parser.add_argument(
    "-f",
    "--feature_count",
    default=None,
    type=int,
    help="Set to a number to choose how many features to use (prune) from original model",
)
parser.add_argument("-debug", default=False, action="store_true", help="Enable debug printouts")

# For convenience
args = parser.parse_args()
arg_model_path = args.model_path
mapping_path = args.mapping_path
enable_debug_printouts = args.debug
map_name = args.map_name
target_prune_features = args.feature_count

# Info printout
print(
    "",
    "This script can be used to reduce the size of the SAMv3 image encoder.",
    "",
    "It works by keeping only a subset of the original transformer model",
    "which has 32 blocks grouped into 4 stages. This script uses a block",
    "mapping to decide which blocks to keep and in what order:",
    f"@ {mapping_path}",
    "",
    "Removal of 8 or more blocks degrades accuracy.",
    "The choice of which blocks are kept can have a significant impact",
    "on the way in which the model degrades!",
    "Fine-tuning/distillation can recover some of the lost performance.",
    sep="\n",
)

# Create history to re-use selected inputs
root_path = Path(__file__).parent.parent
history = HistoryKeeper(root_path)
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)

# Store history for use on repeat pruning
history.store(model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Get block mapping

# Force relative path to be relative to this script
mapping_path = Path(mapping_path)
if mapping_path.parent == Path("."):
    mapping_path = Path(__file__).parent / mapping_path

# Load mapping (create if missing)
if not mapping_path.exists():
    default_samv3_block_mapping_dict = make_default_image_encoder_block_mapping()
    save_unnested_json(mapping_path, default_samv3_block_mapping_dict)
assert mapping_path.exists(), f"Error! Couldn't load mapping file: {mapping_path}"
with open(mapping_path, "r") as infile:
    block_mappings_dict = json.load(infile)
assert isinstance(block_mappings_dict, dict), "Error loading mapping, expecting a dictionary!"

# Assume we're using samv3 (not support for other models for now)
block_mappings_dict = block_mappings_dict.get("samv3", None)
if block_mappings_dict is None:
    raise KeyError("Error, couldn't find 'samv3' key in block mapping")

# Ask user for mapping if not given
if map_name is None:
    map_name = select_from_options(block_mappings_dict.keys(), default_option=None, title_message="Select mapping:")

# Try to load block index mapping
idx_to_keep_per_stage_list = block_mappings_dict.get(map_name, None)
if idx_to_keep_per_stage_list is None:
    raise ValueError(f"Invalid mapping name: {map_name}")

# Make sure each stage has the same number of blocks
num_new_blocks_per_stage = [len(idx_list) for idx_list in idx_to_keep_per_stage_list]
same_sized_stages = all(num_new_blocks_per_stage[0] == stage_size for stage_size in num_new_blocks_per_stage)
assert same_sized_stages, f"Stages must all have the same number of blocks! Got sizes: {num_new_blocks_per_stage}"

# Warn if number of blocks doesn't match key selection
# Create flattened copy of indexes for easier handling
flat_idx_to_keep_list = []
for idx_list in idx_to_keep_per_stage_list:
    flat_idx_to_keep_list.extend(idx_list)
actual_num_blocks_to_keep = len(flat_idx_to_keep_list)
blocks_per_stage = num_new_blocks_per_stage[0]


# ---------------------------------------------------------------------------------------------------------------------
# %% Load model

# Load base model
print("", "Loading base model...", sep="\n")
state_dict = torch.load(model_path)

# Sanity check. Make sure we're dealing with SAMv3
mugsam_key = "config_muggled_samv3"
if mugsam_key in state_dict.keys():
    raise TypeError("MuggledSAM model detected! Only original SAMv3 models are supported")
sam3_required_key = "detector.backbone.vision_backbone.trunk.pos_embed"
if sam3_required_key not in state_dict.keys():
    raise TypeError("Error! Only SAMv3 models are supported")

# Get model config for reporting
orig_config = get_model_config_from_state_dict(state_dict)
num_img_feats = orig_config["imgencoder_features"]
num_img_heads = orig_config["imgencoder_num_heads"]


# ---------------------------------------------------------------------------------------------------------------------
# %% Prune features

is_prune_enabled = target_prune_features is not None
if is_prune_enabled:

    # Get state dict of original model in mugsam format
    orig_config, orig_model_mugsam = make_samv3_from_original_state_dict(state_dict)
    _, reverse_key_lut = convert_state_dict_keys(orig_config, state_dict)
    orig_sd_mugsam = orig_model_mugsam.state_dict()
    del orig_model_mugsam

    # Sanity check, make sure we're actually doing something
    num_orig_feats, num_orig_heads = orig_config["imgencoder_features"], orig_config["imgencoder_num_heads"]
    num_orig_feats_per_head = round(num_orig_feats / num_orig_heads)
    assert (
        target_prune_features < num_orig_feats
    ), f"Error no features are being pruned (orig feature count: {num_orig_feats})"

    # Create new model config (with features pruned)
    new_img_feats = round(target_prune_features / num_orig_feats_per_head) * num_orig_feats_per_head
    new_img_heads = round(new_img_feats / num_orig_feats_per_head)
    new_config = {**orig_config}
    new_config["imgencoder_features"] = new_img_feats
    new_config["imgencoder_num_heads"] = new_img_heads

    # Get state dict of new mugsam model (with reduced feature count)
    new_model_mugsam = make_sam_v3(**new_config)
    new_sd_mugsam = new_model_mugsam.state_dict()
    del new_model_mugsam

    # Try to copy features into pruned model
    print("", f"Pruning image encoder features ({num_orig_feats} down to {new_img_feats})...")
    is_ok_prune, pruned_sd = copy_samv3_features(orig_sd_mugsam, new_sd_mugsam, state_dict, reverse_key_lut)
    assert set(pruned_sd.keys()) == set(state_dict.keys()), "Error, mismatching keys after feature pruning!"
    if not is_ok_prune:
        print(
            "",
            "Warning",
            "Feature pruning produced errors!",
            "The pruned model may not be useable...",
            sep="\n",
        )
    state_dict = pruned_sd
    num_img_feats = new_img_feats
    num_img_heads = new_img_heads


# ---------------------------------------------------------------------------------------------------------------------
# %% Get block indexing

# Hard-code block prefix for SAMv3
target_block_key_prefix = "detector.backbone.vision_backbone.trunk.blocks"
target_freqs_key_component = "freqs_cis"

# Get all model keys for every block
total_orig_layers = 0
block_keys_by_idx_lut = defaultdict(list)
max_block_idx = -1
freqs_keys = []
for key in state_dict.keys():

    # Skip non-block keys
    if not key.startswith(target_block_key_prefix):
        continue

    # Figure out the block index
    # Expecting keys like: detector.backbone.vision_backbone.trunk.blocks.5.attn.qkv.weight
    # -> Want to figure out the number '5' in this case (but done for all keys)
    components = key.split(".")
    integer_components = [int(comp) for comp in key.split(".") if comp.isnumeric()]
    assert len(integer_components) > 0, "Expecting at least 1 index in block key"

    # Record keys by index
    block_index = int(integer_components[0])
    block_keys_by_idx_lut[block_index].append(key)
    max_block_idx = max(max_block_idx, block_index)
    total_orig_layers += 1

    # Record freqs size info, which can be used to indicate windowed vs. global blocks
    if target_freqs_key_component in key:
        freqs_keys.append(key)

# Record examples of minimum and maximum 'freqs_cis' weights for overwriting windowed vs. global blocks on saving
windowed_freqs_weight, global_freqs_weight = torch.zeros((576, 32)), torch.zeros((5184, 32))
freqs_sizes = [state_dict[key].shape[0] for key in freqs_keys]
for key in freqs_keys:
    freqs_weight = state_dict[key]
    if freqs_weight.shape[0] == min(freqs_sizes):
        windowed_freqs_weight = freqs_weight
        break
for key in freqs_keys:
    freqs_weight = state_dict[key]
    if freqs_weight.shape[0] == max(freqs_sizes):
        global_freqs_weight = freqs_weight
        break


# ---------------------------------------------------------------------------------------------------------------------
# %% Remove/rename layers

# For convenience
debug_print = lambda *args, **kwargs: print(*args, **kwargs) if enable_debug_printouts else None

# Report blocks for inspection
print("", f"Keeping {actual_num_blocks_to_keep} blocks. New indexing:", "", sep="\n")
for stage_idx, idx_to_keep_list in enumerate(idx_to_keep_per_stage_list):
    print(f"Stage {stage_idx}:", ", ".join(str(idx).rjust(2) for idx in idx_to_keep_list))

# Remove keys from the state dict that we're not saving
debug_print("", "Removing layers:", sep="\n")
num_layers_removed = 0
for idx in range(1 + max_block_idx):

    # Skip keys we're keeping
    if idx in flat_idx_to_keep_list:
        continue

    # Remove keys that we don't want to save
    keys_to_remove = block_keys_by_idx_lut[idx]
    for key in keys_to_remove:
        rem_key = state_dict.pop(key, None)
        if rem_key is not None:
            debug_print(" ", key)
            num_layers_removed += 1
print("", f"Removing {num_layers_removed} of {total_orig_layers} layers", sep="\n")

# Store copies of weights to keep/rename while removing all others.
# -> Need to do this separately from renaming, in case user tries to repeat/re-orders layers
weights_to_keep_sd = {}
for old_idx, old_keys_per_idx in block_keys_by_idx_lut.items():

    for old_key in old_keys_per_idx:
        old_weights = state_dict.pop(old_key, None)
        if old_weights is None:
            continue
        if old_idx in flat_idx_to_keep_list:
            weights_to_keep_sd[old_key] = old_weights

# Rename keys that we're keeping to have proper sequential indexing
# (we 'rename' all keys, even if they don't change, so that they're together in the model weights file)
debug_print("", "Renaming layers:", sep="\n")
for new_idx, old_idx in enumerate(flat_idx_to_keep_list):
    keys_to_rename = block_keys_by_idx_lut[old_idx]
    for old_key in keys_to_rename:
        if old_key not in weights_to_keep_sd:
            debug_print(" ", "Missing:", old_key)
            continue
        new_key = old_key.replace(f"blocks.{old_idx}", f"blocks.{new_idx}")
        state_dict[new_key] = weights_to_keep_sd[old_key]
        debug_print(" ", old_key)

        # Update freqs_cis (if present) to indicate global blocks
        if target_freqs_key_component in new_key:
            is_global_layer = ((1 + new_idx) % blocks_per_stage) == 0
            state_dict[new_key] = global_freqs_weight if is_global_layer else windowed_freqs_weight

# Sanity check, make sure we're changing something
is_identical_order = tuple(flat_idx_to_keep_list) == tuple(range(max_block_idx + 1))
if is_identical_order and num_layers_removed == 0:
    print("", "*" * 32, "WARNING: Blocks are unchanged!", "*" * 32, sep="\n")


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Save model***

# Figure out name of new model
old_model_path = Path(model_path)
new_model_name = f"{old_model_path.stem}_imgenc_{map_name}"
if is_prune_enabled:
    new_model_name = f"{old_model_path.stem}_imgenc_{num_img_feats}feats_{map_name}"
new_model_path = old_model_path.with_stem(new_model_name)
for rename_idx in range(2, 100):
    if not new_model_path.exists():
        break
    assert rename_idx <= 20, f"Error cannot determine a name for saving without overwriting ({new_model_name})"
    new_model_path = old_model_path.with_stem(f"{new_model_name}_v{rename_idx}")

# Get user to confirm before saving (large) file
print("", "Created new model:", new_model_path.name, "", sep="\n", flush=True)
user_confirm_save = confirm_prompt("Save model")

# Only save when confirmed
if user_confirm_save:

    # Save data with some feedback
    print("", "Saving new model:", str(new_model_path).replace(str(Path.home()), "~"), sep="\n")
    torch.save(state_dict, new_model_path)

    # Provide info about how to use with original codebase
    new_global_idxs = [n + sum(num_new_blocks_per_stage[:idx]) - 1 for idx, n in enumerate(num_new_blocks_per_stage)]
    print(
        "",
        "*" * 64,
        "",
        "Using this model with the original SAM3 codebase requires (minor) modifications:",
        f"  1 - Set the ViT embedding dimension to: {num_img_feats}",
        f"  2 - Set the depth to: {actual_num_blocks_to_keep}",
        f"  3 - Set the number of heads to: {num_img_heads}",
        f"  4 - Set the global attention block indices to: {tuple(new_global_idxs)}",
        "",
        "See:",
        "https://github.com/facebookresearch/sam3/blob/f6e51f59500a87c576c2df2323ce56b9fd7a12de/sam3/model_builder.py#L78-L80",
        "https://github.com/facebookresearch/sam3/blob/f6e51f59500a87c576c2df2323ce56b9fd7a12de/sam3/model_builder.py#L87",
        sep="\n",
    )
