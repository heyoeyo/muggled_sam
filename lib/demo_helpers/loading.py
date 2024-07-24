#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import os.path as osp


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions

# .....................................................................................................................


def clean_path_str(path=None):
    """
    Helper used to interpret user-given pathes correctly
    Import for Windows, since 'copy path' on file explorer includes quotations!
    """

    path_str = "" if path is None else str(path)
    return osp.expanduser(path_str).strip().replace('"', "").replace("'", "")


# .....................................................................................................................


def ask_for_path_if_missing(path=None, file_type="file"):

    # Bail if we get a good path
    path = clean_path_str(path)
    if osp.exists(path):
        return path

    # Keep asking for path until it points to something
    try:
        while True:
            print("", flush=True)
            path = clean_path_str(input(f"Enter path to {file_type}: "))
            if osp.exists(path):
                break
            print("", "", f"Invalid {file_type} path!", sep="\n", flush=True)

    except KeyboardInterrupt:
        quit()

    return path


# .....................................................................................................................


def ask_for_model_path_if_missing(file_dunder, model_path=None):

    # Bail if we get a good path
    path_was_given = model_path is not None
    model_path = clean_path_str(model_path)
    if osp.exists(model_path):
        return model_path

    # If we're given a path that doesn't exist, use it to match to similarly named model files
    # -> This allows the user to select models using substrings, e.g. 'large_5' to match to 'beit_large_512'
    model_file_paths = get_model_weights_paths(file_dunder)
    if path_was_given:
        model_file_paths = list(filter(lambda p: model_path in osp.basename(p), model_file_paths))

    # If we have model files remaining (after filtering), pick the one with the smallest file size to auto-load
    if len(model_file_paths) > 0:
        model_path = min(model_file_paths, key=osp.getsize)

    # If we still don't have a model path, ask the user
    model_path = ask_for_path_if_missing(model_path, "model weights")

    return model_path


# .....................................................................................................................


def get_model_weights_paths(file_dunder, model_weights_folder_name="model_weights"):

    # Build path to model weight folder (and create if missing)
    script_caller_folder_path = osp.dirname(file_dunder) if osp.isfile(file_dunder) else file_dunder
    model_weights_path = osp.join(script_caller_folder_path, model_weights_folder_name)
    os.makedirs(model_weights_path, exist_ok=True)

    # Get only the paths to files with specific extensions
    valid_exts = {".pt", ".pth"}
    all_files_list = os.listdir(model_weights_path)
    model_files_list = [file for file in all_files_list if osp.splitext(file)[1].lower() in valid_exts]
    model_file_paths = [osp.join(model_weights_path, file) for file in model_files_list]

    return model_file_paths
