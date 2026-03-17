#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from pathlib import Path
from datetime import datetime as dt
import json

import torch
import torch.nn as nn
import numpy as np

# For type hints
from typing import Any
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class TrainModulesDict:
    """
    Data structure used to hold modules for training.
    It is a two-level dictionary-like structure.
    The top level holds a reference to different training layer types (e.g. Linear/Embedding/Layernorm etc.).
    The second level holds a reference to layer names and corresponding modules.
    For example:
        {"lora_linear": {
                "layer.mlp.0": <module instance>,
                "layer.mlp.1": <module instance>,
                etc.
            }
        }
    """

    # .................................................................................................................

    def __init__(self, data_dict: dict[str, dict[str, nn.Module]] | None = None):
        self._data_dict = {} if data_dict is None else data_dict

    def __iter__(self):
        for layer_type, layer_module_dict in self._data_dict.items():
            for layer_name, module_ref in layer_module_dict.items():
                yield layer_type, layer_name, module_ref
        return

    def __getitem__(self, index):
        return self._data_dict[index]

    def get_dict(self, key, default=None):
        return self._data_dict.get(key, default)

    def get_layer_types(self):
        return self._data_dict.keys()

    def get_layer_names(self, layer_type: str):
        return self._data_dict[layer_type].keys()

    def get_module(self, layer_type: str, layer_name: str) -> tuple[bool, nn.Module | None]:
        """Get a module by layer type/name, if it exists. Returns: module_exists, module"""
        module_ref = self._data_dict.get(layer_type, {}).get(layer_name, None)
        is_ok = module_ref is not None
        return is_ok, self._data_dict[layer_type][layer_name] if is_ok else None

    # .................................................................................................................

    def reset_all_weights(self):
        """Reset the weights of all modules"""
        for _, _, module_ref in self:
            module_ref.reset_weights()
        return

    def record_state_dict(self, layer_name_prefix: str | None = None) -> dict[str, dict[str, Tensor]]:
        """
        Output a dictionary containing the state_dict of all training weights.
        This dictionary has a two-layer structure, where the first key
        represents the type of layer (e.g. 'lora_linear' vs. 'lora_embedding'),
        while the second key represents the name of the layer being trained. The
        value associated with the second key is itself a state_dict.
        Returns:
            train_weights_state_dicts
        """
        out_data_dict = {layer_type: {} for layer_type in self.get_layer_types()}
        for layer_type, layer_name, module_ref in self:
            if layer_name_prefix is not None:
                layer_name = f"{layer_name_prefix}.{layer_name}"
            out_data_dict[layer_type][layer_name] = module_ref.record_weights()
        return out_data_dict

    def load_state_dict(self, recorded_data_dict: dict[str, dict[str, Tensor]]) -> None:
        """Load training weights from a previously recorded state_dict. See 'record_state_dict(...)' function"""
        for layer_type, layer_state_dict in recorded_data_dict.items():
            for layer_name, state_dict in layer_state_dict.items():
                self._data_dict[layer_type][layer_name].load_weights(state_dict)
        return

    def store_training_modules(
        self, layer_type: str, layer_module_dict: dict[str, nn.Module], replace: bool = True
    ) -> None:
        """
        Helper used to bulk-store training modules. Training modules are
        expected to be provided as a dictionary with keys presenting layer names,
        and values representing the training modules.
        """
        if layer_type not in self._data_dict:
            self._data_dict[layer_type] = {}
        if replace:
            self._data_dict[layer_type] = {**layer_module_dict}
        else:
            self._data_dict[layer_type].update(layer_module_dict)
        return

    # .................................................................................................................


class ShuffleList:
    """
    Helper class which implements a list that is read sequentially,
    when all entries in the list have been read, the list is shuffled
    and reading resets to the beginning of the list.
    """

    # .................................................................................................................

    def __init__(self, initial_data: list[Any] | None = None, shuffle_on_init: bool = True):

        # Handle input options
        if initial_data is None:
            initial_data = []
        if not isinstance(initial_data, list):
            initial_data = list(initial_data)

        self._data = initial_data
        self._curr_item_idx = -1
        self._idx_list = list(range(len(initial_data)))
        if shuffle_on_init and len(initial_data) > 0:
            self.force_shuffle()
        pass

    # .................................................................................................................

    def __str__(self) -> str:
        return [self._data[idx] for idx in self._idx_list].__str__()

    def __repr__(self) -> str:
        return [self._data[idx] for idx in self._idx_list].__repr__()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        while True:
            is_reshuffle, data = self.get_next()
            yield data
            if is_reshuffle:
                break
        return

    def __getitem__(self, index: int):
        data_idx = self._idx_list[index]
        return self._data[data_idx]

    # .................................................................................................................

    def append(self, new_data) -> None:
        self._data.append(new_data)
        num_items = len(self._data)
        min_idx = max(0, self._curr_item_idx + 1)
        max_idx = max(num_items, min_idx + 1)
        new_rand_idx = np.random.randint(min_idx, max_idx)
        self._idx_list.insert(new_rand_idx, num_items - 1)

    def clear(self) -> None:
        self._data = []
        self._idx_list = []

    # .................................................................................................................

    def get_next(self, repeat: bool = False) -> tuple[bool, Any]:

        # Don't update item indexing if we're repeating an entry
        if not repeat:
            self._curr_item_idx += 1

        data_idx = self._idx_list[self._curr_item_idx]
        result = self._data[data_idx]
        is_reshuffle = False

        is_reshuffle = self._curr_item_idx >= len(self._idx_list) - 1
        if is_reshuffle:
            self.force_shuffle()

        return is_reshuffle, result

    def remove_previous(self) -> Any:
        if len(self._idx_list) == 0:
            return None

        data_idx = self._idx_list.pop()
        result = self._data.pop(data_idx)
        assert len(self._idx_list) == len(self._data), "Error, mismatch data/indexing length!"
        self._curr_item_idx = min(self._curr_item_idx, len(self._data) - 1)
        return result

    def force_shuffle(self) -> None:
        np.random.shuffle(self._idx_list)
        self._curr_item_idx = -1
        return

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def make_save_folder(
    root_path: str | Path, model_name, save_folder_name: str, base_folder_name: str = "saved_train_weights"
) -> Path:
    "Helper used to build a folder path for saving training weights. Returns: save_folder_path"

    # Make sure we're dealing with names/folder paths not file paths
    root_path = Path(root_path)
    if root_path.is_file():
        root_path = root_path.parent
    save_folder_name = Path(save_folder_name)
    if save_folder_name.is_file():
        save_folder_name = save_folder_name.stem

    # Remove extension from model name (and folder pathing, if present)
    model_name = Path(model_name).stem

    return root_path / base_folder_name / model_name / save_folder_name


# .....................................................................................................................


def save_training_weights(
    save_folder_path: Path, data_dict: dict, train_iterations: int, ext: str = "pt"
) -> tuple[str, float]:
    """
    Helper used to save training weight data.
    Generates a time-stamped file name to help ensure uniqueness
    Returns:
        save_path, size_of_saved_file_mb
    """

    # Remove leading period from file extension
    if ext.startswith("."):
        ext = ext[1:]

    # Create name and final save path
    dt_now = dt.now()
    date_prefix_str = dt_now.date().isoformat().replace("-", "")
    seconds_of_day = (dt_now - dt_now.replace(hour=0, minute=0, second=0)).total_seconds()
    timestamp_str = hex(int(seconds_of_day)).removeprefix("0x").rjust(5, "0")
    save_name = f"{date_prefix_str}_{timestamp_str}_{train_iterations}its.{ext}"

    # Save data and get file size for reporting
    save_folder_path.mkdir(parents=True, exist_ok=True)
    save_path = save_folder_path / save_name
    torch.save(data_dict, save_path)
    save_size_mb = save_path.stat().st_size / 1_000_000

    return save_path, save_size_mb


# .....................................................................................................................


def get_training_weight_paths(
    root_folder_path: str | Path,
    base_model_name: str | Path,
    weight_folder_name: str | Path = "saved_train_weights",
    reverse_sort_paths: bool = True,
) -> list[Path]:
    """
    Helper used to list out all training weights associated with a given model
    Returns:
        weight_file_paths_list
    """

    # Build pathing to where we expect training weights to be stored for the given model
    base_model_name = Path(base_model_name).stem
    weight_folder_name = Path(weight_folder_name).stem
    model_train_folder = Path(root_folder_path) / weight_folder_name / base_model_name

    # List out all saved torch files under the training folder
    paths_list = []
    for parent_path, folders_list, files_list in model_train_folder.walk():
        for file_name in files_list:
            is_pytorch_file = file_name.endswith("pt") or file_name.endswith("pth")
            if is_pytorch_file:
                file_path = parent_path / file_name
                paths_list.append(file_path)
            pass
        pass

    # Sort paths (for datetime-named files, this puts newest file first)
    if reverse_sort_paths:
        return sorted(paths_list, reverse=True)
    return paths_list


# .....................................................................................................................


def load_prior_weights(
    path_to_weights: str,
    student_name: str,
    train_modules: dict[str, dict[str, Tensor]],
    prefix_to_remove: str | None = None,
) -> bool:
    """
    Helper used to load previously saved weights into lora modules
    Assumes we've already set up 'new' lora modules and we load into them.
    If the loaded data contains weights for layers we aren't using, then
    they won't be loaded!

    Returns:
        needed_module_resizing
        -> Typically this is True if the LoRA rank of loaded data doesn't match the existing modules
    """

    # Load weights and check if student name matches
    loaded_weights = torch.load(path_to_weights)
    target_student_name = loaded_weights.get("student_name", "unknown")
    if target_student_name != student_name:
        print(
            "",
            "Warning:",
            f"Loaded weights are for a different student model ({target_student_name})",
            "This may result in unexpected behavior...",
            sep="\n",
            flush=True,
        )
    assert "weights" in loaded_weights.keys(), "Error interpretting loaded weights, expecting 'weights' key..."

    # Loop over all loaded weights and try to load into existing modules
    existing_layer_types = train_modules.get_layer_types()
    need_module_resizing = False
    for layer_type, layer_to_state_dict in loaded_weights["weights"].items():

        # Skip any layers that are loaded but not in existing training modules (e.g. they weren't enabled)
        if layer_type not in existing_layer_types:
            print(f"  -> Skipping loaded '{layer_type}' weights (disabled)")
            continue

        # Load each layer state dict into existing modules
        num_loaded_modules = 0
        for layer_name, state_dict in layer_to_state_dict.items():

            # Remove layer prefix, which may be added on saving
            if prefix_to_remove is not None:
                layer_name = layer_name.removeprefix(prefix_to_remove)

            # Skip any layers that aren't part of existing training modules
            is_mod_ok, module_ref = train_modules.get_module(layer_type, layer_name)
            if not is_mod_ok:
                print("  -> Skipping unused layer:", layer_name)
                continue
            need_module_resizing |= module_ref.load_weights(state_dict)
            num_loaded_modules += 1
        print(f"  -> {layer_type}: Loaded {num_loaded_modules} modules")

    return need_module_resizing


# .....................................................................................................................


def load_text_list_file(file_path: str | Path) -> list[str]:
    """
    Simple helper used to a list of text data from a plain text
    file, assuming each line is a unique entry. Also supports
    loading from a json file (expecting a lsit of strings).
    Returns:
        list_of_text_entries
    """
    is_json_file = Path(file_path).suffix.lower() == ".json"
    with open(file_path, "r") as infile:
        all_text_list = json.load(infile) if is_json_file else infile.read().splitlines()

    return all_text_list
