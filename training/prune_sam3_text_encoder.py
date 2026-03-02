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
default_mapping_path = "layer_mappings_text_encoder.json"

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to prune the text encoder of a SAMv3 model")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Path to SAMv3 model weights")
parser.add_argument(
    "-map",
    "--mapping_path",
    default=default_mapping_path,
    type=str,
    help="Path to a json file containing layer mappings (run script to generate initial file)",
)
parser.add_argument("-n", "--map_name", default=None, type=str, help="Name of layer mapping to use")
parser.add_argument("-debug", default=False, action="store_true", help="Enable debug printouts")

# For convenience
args = parser.parse_args()
arg_model_path = args.model_path
mapping_path = args.mapping_path
enable_debug_printouts = args.debug
map_name = args.map_name

# Info print out
print(
    "",
    "This script can be used to reduce the size of the SAMv3 text encoder.",
    "",
    "It works by keeping only a subset of the original transformer model",
    "which has 24 block/layers. This script uses a block mapping to",
    "decide which of the layers to keep and in what order:",
    f"@ {mapping_path}",
    "",
    "Model accuracy is expected to degrade as layers are removed,",
    "though the choice of which layers are kept can have a",
    "significant impact on the degradation!",
    "Fine-tuning/distillation can recover some of the lost performance.",
    "",
    "Note:",
    "The text encoder makes up around 40% of the original SAMv3",
    "model file size but less than 1% of the inference time.",
    "So the main benefit of pruning is to reduce the file size.",
    "It may also speed up fine-tuning by having fewer parameters to train.",
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
default_textenc_block_mapping_dict = {
    "samv3": {
        "4_layers": tuple(range(4)),
        "8_layers": tuple(range(8)),
        "12_layers": tuple(range(12)),
        "16_layers": tuple(range(16)),
        "20_layers": tuple(range(20)),
        "reference": tuple(range(24)),
    }
}

# Load mapping (create if missing)
mapping_path = Path(mapping_path)
if not mapping_path.exists():
    # Try to format json a bit nicer on save
    json_str = json.dumps(default_textenc_block_mapping_dict)
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
idx_to_keep_list = block_mappings_dict.get(map_name, None)
if idx_to_keep_list is None:
    raise ValueError(f"Invalid mapping name: {map_name}")
num_to_keep = len(idx_to_keep_list)

# Provide feedback about selected option
print(f"             --> {map_name}")


# ---------------------------------------------------------------------------------------------------------------------
# %% Load model

# Load base model
print("", "Loading base model...", sep="\n")
state_dict = torch.load(model_path)

# Sanity check. Make sure we're dealing with SAMv3
mugsam_key = "config_muggled_samv3"
if mugsam_key in state_dict.keys():
    raise TypeError("MuggledSAM model detected! Only original SAMv3 models are supported")
sam3_required_key = "detector.backbone.language_backbone.encoder.token_embedding.weight"
if sam3_required_key not in state_dict.keys():
    raise TypeError("Error! Only SAMv3 models are supported")

# Hard-code block prefix for the text encoder
target_block_key_prefix = "detector.backbone.language_backbone.encoder.transformer.resblocks"

# Get all model keys for every block
total_orig_layers = 0
block_keys_by_idx_lut = defaultdict(list)
max_block_idx = -1
for key in state_dict.keys():

    # Skip non-block keys
    if not key.startswith(target_block_key_prefix):
        continue

    # Figure out the block index
    # Expecting keys like: detector.backbone.language_backbone.encoder.transformer.resblocks.4.attn.out_proj.weight
    # -> Want to figure out the number '4' in this case (but done for all keys)
    components = key.split(".")
    integer_components = [int(comp) for comp in key.split(".") if comp.isnumeric()]
    assert len(integer_components) > 0, "Expecting at least 1 index in block key"

    # Record keys by index
    block_index = int(integer_components[0])
    block_keys_by_idx_lut[block_index].append(key)
    max_block_idx = max(max_block_idx, block_index)
    total_orig_layers += 1


# ---------------------------------------------------------------------------------------------------------------------
# %% Remove/rename layers

# For convenience
debug_print = lambda *args, **kwargs: print(*args, **kwargs) if enable_debug_printouts else None

# Report blocks for inspection
print(
    "",
    f"Keeping {num_to_keep} blocks",
    f"New indexing: {', '.join(str(idx) for idx in idx_to_keep_list)}",
    "",
    sep="\n",
)

# Remove keys from the state dict that we're not saving
debug_print("", "Removing layers:", sep="\n")
num_layers_removed = 0
for idx in range(1 + max_block_idx):

    # Skip keys we're keeping
    if idx in idx_to_keep_list:
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
        if old_idx in idx_to_keep_list:
            weights_to_keep_sd[old_key] = old_weights

# Rename keys that we're keeping to have proper sequential indexing
# (we 'rename' all keys, even if they don't change, so that they're together in the model weights file)
debug_print("", "Renaming layers:", sep="\n")
for new_idx, old_idx in enumerate(idx_to_keep_list):
    keys_to_rename = block_keys_by_idx_lut[old_idx]
    for old_key in keys_to_rename:
        if old_key not in weights_to_keep_sd:
            debug_print(" ", "Missing:", old_key)
            continue
        new_key = old_key.replace(f"blocks.{old_idx}", f"blocks.{new_idx}")
        state_dict[new_key] = weights_to_keep_sd[old_key]
        debug_print(" ", old_key)

# Sanity check, make sure we're changing something
is_identical_order = tuple(idx_to_keep_list) == tuple(range(max_block_idx + 1))
if is_identical_order and num_layers_removed == 0:
    print("", "*" * 32, "WARNING: Blocks are unchanged!", "  -> Model won't be changed", "*" * 32, sep="\n")


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Save model***

# Figure out name of new model
old_model_path = Path(model_path)
new_model_name = f"{old_model_path.stem}_txtenc_{map_name}"
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
    print(
        "",
        "Using this model with the original SAM3 codebase requires",
        f"modifying the text encoder layer count to be: {num_to_keep}",
        "",
        "See:",
        "https://github.com/facebookresearch/sam3/blob/f6e51f59500a87c576c2df2323ce56b9fd7a12de/sam3/model_builder.py#L497",
        sep="\n",
    )
