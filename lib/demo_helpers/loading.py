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


def ask_for_path_if_missing(path=None, file_type="file", default_path=None):

    # Bail if we get a good path
    path = clean_path_str(path)
    if osp.exists(path):
        return path

    # Wipe out bad default paths
    if default_path is not None:
        if not osp.exists(default_path):
            default_path = None

    # Set up prompt text and default if needed
    prompt_txt = f"Enter path to {file_type}: "
    default_msg_spacing = " " * (len(prompt_txt) - len("(default:") - 1)
    default_msg = "" if default_path is None else f"{default_msg_spacing}(default: {default_path})"

    # Keep asking for path until it points to something
    try:
        while True:

            # Print empty line for spacing and default hint if available
            print("", flush=True)
            if default_path is not None:
                print(default_msg, flush=True)

            # Ask user for path, and fallback to default if nothing is given
            path = clean_path_str(input(prompt_txt))
            if path == "" and default_path is not None:
                path = default_path

            # Stop asking once we get a valid path
            if osp.exists(path):
                break
            print("", "", f"Invalid {file_type} path!", sep="\n", flush=True)

    except KeyboardInterrupt:
        quit()

    return path


# .....................................................................................................................


def ask_for_model_path_if_missing(file_dunder, model_path=None, default_prompt_path=None):

    # Bail if we get a good path
    path_was_given = model_path is not None
    model_path = clean_path_str(model_path)
    if osp.exists(model_path):
        return model_path

    # If we're given a path that doesn't exist, use it to match to similarly named model files
    # -> This allows the user to select models using substrings, e.g. 'large_5' to match to 'beit_large_512'
    model_file_paths = get_model_weights_paths(file_dunder)
    if path_was_given:

        # If there is exactly 1 model that matches the given string, then load it
        filtered_paths = list(filter(lambda p: model_path in osp.basename(p), model_file_paths))
        if len(filtered_paths) == 1:
            return filtered_paths[0]

    # If there are no files in the model weights folder, as the user to enter a path to load a model
    if len(model_file_paths) == 0:
        model_path = ask_for_path_if_missing(model_path, "model weights", default_prompt_path)
    else:
        # If more than 1 file is available, provide a menu to select from the models
        model_path = ask_for_model_from_menu(model_file_paths, default_prompt_path)

    return model_path


# .....................................................................................................................


def ask_for_model_from_menu(model_files_paths, default_path=None):

    # Wipe out bad default paths
    if default_path is not None:
        if not osp.exists(default_path):
            default_path = None

    # Generate list of model selections (ordered by size)
    model_files_paths = sorted(model_files_paths, key=osp.getsize)
    model_names = [osp.basename(filepath) for filepath in model_files_paths]
    menu_item_strs = []
    for idx, (path, name) in enumerate(zip(model_files_paths, model_names)):
        menu_str = f" {1+idx:>2}: {name}"
        is_default = path == default_path
        if is_default:
            menu_str += " (default)"
        menu_item_strs.append(menu_str)

    # Keep giving menu until user selects something
    selected_model_path = None
    try:
        while True:

            # Provide prompt to ask user to select from a list of model files
            print("", "Select model file:", "", *menu_item_strs, "", sep="\n")
            user_selection = clean_path_str(input("Enter selection: "))

            # User the default if the user didn't enter anything (and a default is available)
            if user_selection == "" and default_path is not None:
                selected_model_path = default_path
                break

            # Check if user entered a number matching an item in the list
            try:
                idx_select = int(user_selection) - 1
                selected_model_path = model_files_paths[idx_select]
                break
            except (ValueError, IndexError):
                # Happens is user didn't input an integer selecting an item in the menu
                # -> We'll just assume they entered something else otherwise
                pass

            # Check if the user entered a path to a valid file
            if osp.exists(user_selection):
                selected_model_path = user_selection
                break

            # Check if the user entered a string that matches to some part of one of the entries
            filtered_names = list(filter(lambda p: user_selection in osp.basename(p), model_names))
            if len(filtered_names) == 1:
                user_selected_name = filtered_names[0]
                idx_select = model_names.index(user_selected_name)
                selected_model_path = model_files_paths[idx_select]
                break

            # If we get here, we didn't get a valid input. So warn user and repeat prompt
            print("", "", "Invalid selection!", sep="\n", flush=True)

    except KeyboardInterrupt:
        quit()

    return selected_model_path


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
