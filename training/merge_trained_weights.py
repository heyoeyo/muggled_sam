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

import torch

from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import ask_for_model_path_if_missing, ask_for_path_if_missing, select_from_options
from muggled_sam.demo_helpers.text_input import confirm_prompt
from muggled_sam.demo_helpers.training.io import get_training_weight_paths

from muggled_sam.v3_sam.state_dict_conversion.convert_original_state_dict_keys import convert_state_dict_keys
from muggled_sam.v3_sam.state_dict_conversion.config_from_original_state_dict import get_model_config_from_state_dict


# ---------------------------------------------------------------------------------------------------------------------
# %% Script args

# Set argparse defaults (useful for hard-coding overrides)
default_model_path = None
default_train_path = None

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to merge training weights into SAMv3 models")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Path to base model weights")
parser.add_argument("-t", "--train_path", default=default_train_path, type=str, help="Path to training weights")
parser.add_argument(
    "-c",
    "--confirm",
    default=False,
    action="store_true",
    help="Enable confirmation prompt, allowing for certain weights to be excluded from merging",
)

# For convenience
args = parser.parse_args()
arg_model_path = args.model_path
train_weights_path = args.train_path
enable_confirm_prompt = args.confirm

# Create history to re-use selected inputs
root_path = Path(__file__).parent.parent
history = HistoryKeeper(root_path)
_, history_studentpath = history.read("distill_student_path")

# Get pathing to resources, if not provided already
base_model_path = ask_for_model_path_if_missing(
    root_path, arg_model_path, history_studentpath, message="Select base model:"
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Prompt for weights

# Confirm valid weights path if given directly
if train_weights_path is not None:
    train_weights_path = Path(train_weights_path)
    if not train_weights_path.exists():
        raise FileNotFoundError(f"Invalid file path: {train_weights_path}")

# Ask user for path to training weights if needed
if train_weights_path is None:

    # Try to get weights associated with the loaded model, so we can provide a menu selection
    base_model_path = Path(base_model_path)
    base_model_name = base_model_path.name
    base_model_name_no_ext = base_model_path.stem
    train_file_paths_list = get_training_weight_paths(root_path, base_model_name_no_ext)

    # Ask user for file path if we can't find weights for the model, otherwise give a menu
    num_entries = len(train_file_paths_list)
    if num_entries == 0:
        # Ask user for file path directly
        print("", f"No training files found for model ({base_model_name})", "", sep="\n", flush=True)
        train_weights_path = ask_for_path_if_missing(file_type="training weights")

    else:

        # Figure out which file is newest (will use as default)
        newest_ctime = max(train_file_paths_list, key=lambda path_item: path_item.stat().st_ctime)
        newest_entry_idx = train_file_paths_list.index(newest_ctime)
        default_file_path = train_file_paths_list[newest_entry_idx]

        # Build menu names like: 'parent_folder / name_of_weights.pt' for menu
        menu_names_list = [" / ".join(file_path.parts[-2:]) for file_path in train_file_paths_list]
        train_weights_path = select_from_options(
            menu_names_list,
            default_option=menu_names_list[newest_entry_idx],
            response_list=train_file_paths_list,
            title_message="Select training file:",
            allow_path_response=True,
        )
    pass


# ---------------------------------------------------------------------------------------------------------------------
# %% Load model & weights

print("", "Loading training weights...", f"@ {train_weights_path}", sep="\n")
sd_train = torch.load(train_weights_path)
sd_weights = sd_train.get("weights", None)
if sd_weights is None:
    raise IndexError("Error! Expecting 'weights' key inside of training weights data")

print("", "Loading base model...", f"@ {base_model_path}", sep="\n")
sd_model = torch.load(base_model_path)

# Get model weights/info
model_config = get_model_config_from_state_dict(sd_model)
model_components_to_sd, new_to_old_key_lut = convert_state_dict_keys(model_config, sd_model)

# Warn if we get a name mismatch
train_student_name = sd_train.get("student_name", None)
if train_student_name is not None and train_student_name != base_model_name:
    print(
        "",
        "Warning:",
        f"Base model name ({base_model_name}) doesn't match the training weights ({train_student_name})",
        "This may lead to unexpected performance of the merged model...",
        sep="\n",
    )
    pass

# Warn if configs mismatch
train_student_config = sd_train.get("student_config", None)
if train_student_config is not None:
    mismatch_strs = []
    for key, model_value in model_config.items():
        student_value = train_student_config.get(key, None)
        if student_value != model_value:
            mismatch_strs.append(f"  {key}: {model_value} vs. {student_value}")
    if len(mismatch_strs) > 0:
        print(
            "",
            "Warning:",
            "  Config mismatch found between training weights and base model!",
            "  This may lead to issues merging weights...",
            "",
            "Mismatched entries: (base model vs model used in training)",
            *mismatch_strs,
            sep="\n",
            flush=True,
        )


# ---------------------------------------------------------------------------------------------------------------------
# %% Handle lora-embeddings

lora_embed_rotmat_and_bias_list = []
for layer_name, lora_state_dict in sd_weights.get("lora_embedding", {}).items():

    # Get original weight name corresponding to saved lora weights
    layer_weight_name = f"{layer_name}.weight"
    orig_weight_name = new_to_old_key_lut.get(layer_weight_name, None)
    assert orig_weight_name, "Error, couldn't find embedding weight key: {orig_key_weight}"

    # Get lora weights
    a_weight = lora_state_dict["A"]
    b_weight = lora_state_dict["B"]
    b_bias = lora_state_dict.get("bias", None)
    lora_eye = torch.eye(a_weight.shape[0])
    lora_embed_rotmat = lora_eye + torch.matmul(a_weight, b_weight)

    # Store components for updating existing weights
    lora_embed_rotmat_and_bias_list.append((orig_weight_name, lora_embed_rotmat, b_bias))

num_embeddings = len(lora_embed_rotmat_and_bias_list)
print("", f"Found {num_embeddings} embedding layers", sep="\n")
if num_embeddings > 0 and enable_confirm_prompt:
    user_confirm = confirm_prompt("Merge embeddings")
    if not user_confirm:
        lora_embed_rotmat_and_bias_list = []
        print("Ignoring embeddings...")


# ---------------------------------------------------------------------------------------------------------------------
# %% Handle lora-linears

lora_linear_weights_dict = {}
lora_linear_bias_dict = {}
for layer_name, lora_state_dict in sd_weights.get("lora_linear", {}).items():

    # Get original weight name corresponding to saved lora weights
    layer_weight_name = f"{layer_name}.weight"
    orig_weight_name = new_to_old_key_lut.get(layer_weight_name, None)
    assert orig_weight_name, "Error, couldn't find linear weight key: {orig_key_weight}"

    # Store lora weights
    a_weight = lora_state_dict["A.weight"]
    b_weight = lora_state_dict["B.weight"]
    lora_linear_weights_dict[orig_weight_name] = torch.matmul(b_weight, a_weight)

    # Store bias if present
    b_bias = lora_state_dict.get("B.bias", None)
    if b_bias is not None:
        layer_bias_name = f"{layer_name}.bias"
        orig_bias_name = new_to_old_key_lut.get(layer_bias_name, None)
        assert orig_bias_name is not None, f"Error, missing original bias values for layer: {layer_bias_name}"
        lora_linear_bias_dict[orig_bias_name] = b_bias

num_linears = len(lora_linear_weights_dict)
print("", f"Found {num_linears} linear layers", sep="\n")
if num_linears > 0 and enable_confirm_prompt:
    user_confirm = confirm_prompt("Merge linears")
    if not user_confirm:
        lora_linear_weights_dict = {}
        lora_linear_bias_dict = {}
        print("Ignoring linear layers...")


# ---------------------------------------------------------------------------------------------------------------------
# %% Handle offset-layernorms

offset_layernorm_weights_dict = {}
offset_layernorm_bias_dict = {}
for layer_name, offset_state_dict in sd_weights.get("offset_layernorm", {}).items():

    # Get original weight name corresponding to saved weights
    layer_weight_name = f"{layer_name}.weight"
    orig_weight_name = new_to_old_key_lut.get(layer_weight_name, None)
    assert orig_weight_name, "Error, couldn't find layernorm weight key: {orig_key_weight}"

    # Store lora weights
    weight_offset = offset_state_dict["weight"]
    offset_layernorm_weights_dict[orig_weight_name] = weight_offset

    # Store bias if present
    bias_offset = offset_state_dict.get("bias", None)
    if bias_offset is not None:
        layer_bias_name = f"{layer_name}.bias"
        orig_bias_name = new_to_old_key_lut.get(layer_bias_name, None)
        assert orig_bias_name is not None, f"Error, missing original bias values for layer: {layer_bias_name}"
        offset_layernorm_bias_dict[orig_bias_name] = bias_offset

num_layernorms = len(offset_layernorm_weights_dict)
print("", f"Found {num_layernorms} layernorm layers", sep="\n")
if num_layernorms > 0 and enable_confirm_prompt:
    user_confirm = confirm_prompt("Merge layernorms")
    if not user_confirm:
        offset_layernorm_weights_dict = {}
        offset_layernorm_bias_dict = {}
        print("Ignoring layernorms...")


# ---------------------------------------------------------------------------------------------------------------------
# %% Handle lora-conv2d

lora_conv2d_weights_dict = {}
lora_conv2d_bias_dict = {}
for layer_name, lora_state_dict in sd_weights.get("lora_conv2d", {}).items():

    # Get original weight name corresponding to saved lora weights
    layer_weight_name = f"{layer_name}.weight"
    orig_weight_name = new_to_old_key_lut.get(layer_weight_name, None)
    assert orig_weight_name, "Error, couldn't find conv2d weight key: {orig_key_weight}"

    # Store lora weights
    a_weight = lora_state_dict["A.weight"]
    b_weight = lora_state_dict["B.weight"]
    lora_conv2d_weights_dict[orig_weight_name] = torch.einsum("OrHW,rIHW->OIHW", b_weight, a_weight)

    # Store bias if present
    b_bias = lora_state_dict.get("B.bias", None)
    if b_bias is not None:
        layer_bias_name = f"{layer_name}.bias"
        orig_bias_name = new_to_old_key_lut.get(layer_bias_name, None)
        assert orig_bias_name is not None, f"Error, missing original bias values for layer: {layer_bias_name}"
        lora_conv2d_bias_dict[orig_bias_name] = b_bias

num_conv2ds = len(lora_conv2d_weights_dict)
print("", f"Found {num_conv2ds} conv2d layers", sep="\n")
if num_conv2ds > 0 and enable_confirm_prompt:
    user_confirm = confirm_prompt("Merge conv2d")
    if not user_confirm:
        lora_conv2d_weights_dict = {}
        lora_conv2d_bias_dict = {}
        print("Ignoring conv2d layers...")


# ---------------------------------------------------------------------------------------------------------------------
# %% Handle lora-convtranspose2d

lora_convtranspose2d_weights_dict = {}
lora_convtranspose2d_bias_dict = {}
for layer_name, lora_state_dict in sd_weights.get("lora_convtranspose2d", {}).items():

    # Get original weight name corresponding to saved lora weights
    layer_weight_name = f"{layer_name}.weight"
    orig_weight_name = new_to_old_key_lut.get(layer_weight_name, None)
    assert orig_weight_name, "Error, couldn't find convtranspose2d weight key: {orig_key_weight}"

    # Store lora weights
    a_weight = lora_state_dict["A.weight"]
    b_weight = lora_state_dict["B.weight"]
    lora_convtranspose2d_weights_dict[orig_weight_name] = torch.einsum("Oryx,IrHW->IOHW", b_weight, a_weight)

    # Store bias if present
    b_bias = lora_state_dict.get("B.bias", None)
    if b_bias is not None:
        layer_bias_name = f"{layer_name}.bias"
        orig_bias_name = new_to_old_key_lut.get(layer_bias_name, None)
        assert orig_bias_name is not None, f"Error, missing original bias values for layer: {layer_bias_name}"
        lora_convtranspose2d_bias_dict[orig_bias_name] = b_bias

num_convtranspose2ds = len(lora_convtranspose2d_weights_dict)
print("", f"Found {num_convtranspose2ds} convtranspose2d layers", sep="\n")
if num_conv2ds > 0 and enable_confirm_prompt:
    user_confirm = confirm_prompt("Merge convtranspose2d")
    if not user_confirm:
        lora_convtranspose2d_weights_dict = {}
        lora_convtranspose2d_bias_dict = {}
        print("Ignoring convtranspose2d layers...")


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Save model***

# Copy original weights
new_sd = {**sd_model}

# Handle additive updates [Updated value = old_value + new_value]
for additive_weights_dict in (
    lora_linear_weights_dict,
    lora_linear_bias_dict,
    offset_layernorm_weights_dict,
    offset_layernorm_bias_dict,
    lora_conv2d_weights_dict,
    lora_conv2d_bias_dict,
    lora_convtranspose2d_weights_dict,
    lora_convtranspose2d_bias_dict,
):
    for weight_key, additive_weight in additive_weights_dict.items():
        new_sd[weight_key] += additive_weight

# Handle embedding updates [Updated weight = old_weight * (identity + A * B) + bias]
for weight_key, new_rotmat, new_bias in lora_embed_rotmat_and_bias_list:
    new_sd[weight_key] = torch.matmul(sd_model[weight_key], new_rotmat)
    if new_bias is not None:
        new_sd[weight_key] += new_bias

# Set up name of new model, with renaming to avoid overwriting existing models
new_model_name_no_ext = f"{base_model_name_no_ext}-merged"
new_model_path = base_model_path.with_stem(new_model_name_no_ext)
for rename_idx in range(2, 100):
    if not new_model_path.exists():
        break
    assert rename_idx <= 20, f"Error cannot determine a name for saving without overwriting ({new_model_name_no_ext})"
    new_model_path = base_model_path.with_stem(f"{new_model_name_no_ext}_v{rename_idx}")

# Get user to confirm before saving (large) file
print("", "Created new model:", new_model_path.name, "", sep="\n", flush=True)
user_confirm_save = confirm_prompt("Save merged model")

# # Only save when confirmed
if user_confirm_save:
    print("", "Saving new model:", f"@ {new_model_path}", sep="\n")
    torch.save(new_sd, new_model_path)
else:
    print("Cancelled...")
