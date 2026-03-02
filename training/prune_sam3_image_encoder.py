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
from muggled_sam.demo_helpers.loading import ask_for_model_path_if_missing


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
parser.add_argument("-debug", default=False, action="store_true", help="Enable debug printouts")

# For convenience
args = parser.parse_args()
arg_model_path = args.model_path
mapping_path = args.mapping_path
enable_debug_printouts = args.debug
map_name = args.map_name

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


# ---------------------------------------------------------------------------------------------------------------------
# %% Get block mapping

# Set up a default mapping for saving (user can then modify file)
default_samv3_block_mapping_dict = {
    "samv3": {
        "4_blocks": [[7], [15], [23], [31]],
        "8_blocks": [[0, 7], [8, 15], [16, 23], [24, 31]],
        "12_blocks": [[0, 1, 7], [8, 9, 15], [16, 17, 23], [24, 25, 31]],
        "16_blocks": [[0, 1, 2, 7], [8, 9, 10, 15], [16, 17, 18, 23], [24, 25, 26, 31]],
        "20_blocks": [[0, 1, 2, 3, 7], [8, 9, 10, 11, 15], [16, 17, 18, 19, 23], [24, 25, 26, 27, 31]],
        "2_stages": [[0, 2, 4, 6, 8, 10, 12, 14, 15], [16, 18, 20, 22, 24, 26, 28, 30, 31]],
        "6_stages": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
        "reference": [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31],
        ],
    },
}

# Load mapping (create if missing)
mapping_path = Path(mapping_path)
if not mapping_path.exists():
    # Try to format json a bit nicer on save
    json_str = json.dumps(default_samv3_block_mapping_dict)
    json_str = json_str.replace(' "', '\n "').replace("{", "{\n ").replace("}", "\n}")
    with open(mapping_path, "w") as outfile:
        outfile.write(json_str)
assert mapping_path.exists(), f"Error! Couldn't load mapping file: {mapping_path}"
with open(mapping_path, "r") as infile:
    block_mappings_dict = json.load(infile)
assert isinstance(block_mappings_dict, dict), "Error loading mapping, expecting a dictionary!"

# Assume we're using samv3 (not support for other models for now)
block_mappings_dict = block_mappings_dict.get("samv3", None)
if block_mappings_dict is None:
    raise KeyError("Error, couldn't find 'samv3' key in block mapping")

# Ask user for mapping if not given
num_mapping_entries = len(block_mappings_dict)
assert num_mapping_entries > 0, f"Error empty block mapping ({mapping_path.name})"
if map_name is None:
    map_name = list(block_mappings_dict.keys())[0] if num_mapping_entries == 1 else None
    if num_mapping_entries > 1:
        mapping_choices_list = list(block_mappings_dict.keys())
        strs_to_print = []
        for idx, choice_str in enumerate(mapping_choices_list):
            strs_to_print.append(f"  {1+idx:>2}: {choice_str}")
        print("", "Select index:", "", *strs_to_print, "", sep="\n", flush="")
        map_name = input("Enter selection: ")
        if map_name.isnumeric():
            map_idx = int(map_name) - 1
            assert 0 <= map_idx < num_mapping_entries, f"Bad index, must be between 1 and {num_mapping_entries}"
            map_name = mapping_choices_list[map_idx]
        pass
        if map_name.strip() == "":
            print("Selection cancelled...", "", sep="\n")
            quit()
    pass

# If we get a name that isn't in the mappings, try to match to the start of the mapping names
if map_name not in block_mappings_dict.keys():
    for key in block_mappings_dict.keys():
        if key.startswith(map_name):
            map_name = key
            break

# Try to load block index mapping
idx_to_keep_per_stage_list = block_mappings_dict.get(map_name, None)
if idx_to_keep_per_stage_list is None:
    raise ValueError(f"Invalid mapping name: {map_name}")

# Provide feedback about selected option
print(f"             --> {map_name}")

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
    print("", "*" * 32, "WARNING: Blocks are unchanged!", "  -> Model won't be changed", "*" * 32, sep="\n")


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Save model***

# Figure out name of new model
old_model_path = Path(model_path)
new_model_name = f"{old_model_path.stem}_imgenc_{map_name}"
new_model_path = old_model_path.with_stem(new_model_name)
for rename_idx in range(2, 100):
    if not new_model_path.exists():
        break
    assert rename_idx <= 20, f"Error cannot determine a name for saving without overwriting ({new_model_name})"
    new_model_path = old_model_path.with_stem(f"{new_model_name}_v{rename_idx}")

# Get user to confirm before saving (large) file
user_confirm_save = False
try:
    print("", "Created new model:", new_model_path.name, "", sep="\n", flush=True)
    user_confirm_str = input("Save model? [y/N] ")
    user_confirm_save = user_confirm_str.lower().strip() == "y"

except KeyboardInterrupt:
    pass

# Only save when confirmed
if user_confirm_save:

    # Save data with some feedback
    print("", "Saving new model:", str(new_model_path).replace(str(Path.home()), "~"), sep="\n")
    torch.save(state_dict, new_model_path)

    # Provide info about how to use with original codebase
    new_global_idxs = [n + sum(num_new_blocks_per_stage[:idx]) - 1 for idx, n in enumerate(num_new_blocks_per_stage)]
    print(
        "",
        "Using this model with the original SAM3 codebase requires (minor) modifications.",
        f"  1 - Set the model depth to: {actual_num_blocks_to_keep}",
        f"  2 - Set the global attention block indices to: {tuple(new_global_idxs)}",
        "",
        "See:",
        "https://github.com/facebookresearch/sam3/blob/f6e51f59500a87c576c2df2323ce56b9fd7a12de/sam3/model_builder.py#L79",
        "https://github.com/facebookresearch/sam3/blob/f6e51f59500a87c576c2df2323ce56b9fd7a12de/sam3/model_builder.py#L87",
        sep="\n",
    )
