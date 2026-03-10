#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import os.path as osp
from pathlib import Path
import json
from time import sleep


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions

# .....................................................................................................................


def clean_path_str(path=None):
    """
    Helper used to interpret user-given paths correctly
    Import for Windows, since 'copy path' on file explorer includes quotations!
    """

    path_str = "" if path is None else str(path)
    return osp.expanduser(path_str).strip().replace('"', "").replace("'", "")


# .....................................................................................................................


def ask_for_value_if_missing(
    input_value=None,
    message="Enter value: ",
    default_value=None,
    return_type=str,
    allow_empty_input: bool = False,
):
    """
    Function used to provide a cli-based prompt to ask for an input value
    This is just a wrapper around the built-in 'input(...)' function,
    but includes support for providing a default value and looping
    on invalid input.
    """

    # Bail if we already have an input value
    if input_value is not None:
        return input_value

    # Fix bad defaults
    if default_value == "":
        default_value = None

    # Set up prompt text and default if needed
    default_txt_prefix = "(default: "
    max_msg_spacing = max(len(message), len(default_txt_prefix))
    padded_default_str = default_txt_prefix.rjust(max_msg_spacing, " ")
    default_msg = "" if default_value is None else f"{padded_default_str}{default_value})"
    prompt_msg = message.rjust(max_msg_spacing, " ")

    # Print empty line for spacing and default hint if available

    # Ask user for missing input
    while True:
        try:
            print("", flush=True)
            if default_value is not None:
                print(default_msg, flush=True)
            user_input = input(prompt_msg).strip()

        except KeyboardInterrupt:
            quit()

        # Handle 'use the default' inputs
        is_empty = len(user_input) == 0
        if is_empty and (default_value is not None):
            user_input = default_value
            break

        # Reject missing inputs if needed
        if is_empty and (not allow_empty_input) and (default_value is None):
            print("", "", "Cannot leave input empty!", sep="\n", flush=True)
            continue

        try:
            if return_type is not None:
                user_input = return_type(user_input)
        except ValueError:
            print("", "", f"Invalid input, must be of type: {return_type}", sep="\n", flush=True)
            continue

        # If we get this far, the input passed checks so break from the loop
        break

    return user_input


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


def ask_for_model_path_if_missing(file_dunder, model_path=None, default_prompt_path=None, message="Select model file:"):

    # Bail if we get a good path
    path_was_given = model_path is not None
    model_path = clean_path_str(model_path)
    if osp.exists(model_path):
        return model_path

    # If we're given a path that doesn't exist, use it to match to similarly named model files
    # -> This allows the user to select models using substrings, e.g. 'large' to match to 'large_model'
    model_file_paths = get_model_weights_paths(file_dunder)
    if path_was_given:

        # If there is exactly 1 model that matches the given string, then load it
        filtered_paths = list(filter(lambda p: model_path in osp.basename(p), model_file_paths))
        if len(filtered_paths) == 1:
            return filtered_paths[0]

    # Handle no files vs. 1 file vs. many files
    if len(model_file_paths) == 0:

        # Provide link to model listings in case user doesn't have any
        print(
            "",
            "No model path found for loading!",
            "For more info on where to get model files, see:",
            "https://github.com/heyoeyo/muggled_sam/blob/main/README.md#model-weights",
            sep="\n",
            flush=True,
        )

        # If there are no files in the model weights folder, ask the user to enter a path to load a model
        model_path = ask_for_path_if_missing(model_path, "model weights", default_prompt_path)
    elif len(model_file_paths) == 1:
        # If we have exactly one model, return that by default (no need to ask user)
        model_path = model_file_paths[0]
    else:
        # If more than 1 file is available, provide a menu to select from the models
        model_path = ask_for_model_from_menu(model_file_paths, default_prompt_path, message=message)

    return model_path


# .....................................................................................................................


def ask_for_model_from_menu(model_files_paths, default_path=None, message: str = "Select model file:"):
    """
    Function which provides a simple cli 'menu' for selecting which model to load.
    A 'default' can be provided, which will highlight a matching entry in the menu
    (if present), and will be used if the user does not enter a selection.

    Entries are 'selected' by entering their list index, or can be selected by providing
    a partial string match (or otherwise a full path can be used, if valid), looks like:

    Select model file:

      1: model_a.pth
      2: model_b.pth (default)
      3: model_c.pth

    Enter selection:
    """

    # Wipe out bad default paths
    if default_path is not None:
        if not osp.exists(default_path):
            default_path = None

    # Generate list of model selections, including the default path if it isn't in the folder listing
    model_files_paths = sorted(model_files_paths, reverse=True)
    model_names = [osp.basename(filepath) for filepath in model_files_paths]
    default_in_listing = any(default_path == path for path in model_files_paths)
    if not default_in_listing and default_path is not None:
        model_files_paths.append(default_path)
        default_name = osp.join("...", osp.basename(osp.dirname(default_path)), osp.basename(default_path))
        model_names.append(default_name)

    # Create menu listing strings for each model option for display in terminal
    menu_item_strs = []
    for idx, (path, name) in enumerate(zip(model_files_paths, model_names)):
        menu_str = f" {1+idx:>2}: {name}" if not default_in_listing else f"  {1+idx:>2}: {name}"
        is_default = path == default_path
        if is_default:
            menu_str = f" *{1+idx:>2}: {name}  (default)"
        menu_item_strs.append(menu_str)

    # Set up prompt text and feedback printing
    prompt_txt = "Enter selection: "
    feedback_prefix = " " * (len(prompt_txt) - len("-->") - 1) + "-->"
    print_selected_model = lambda index_select: print(f"{feedback_prefix} {model_names[idx_select]}")

    # Keep giving menu until user selects something
    selected_model_path = None
    try:
        while True:

            # Provide prompt to ask user to select from a list of model files
            print("", message, "", *menu_item_strs, "", sep="\n")
            user_selection = clean_path_str(input("Enter selection: "))

            # User the default if the user didn't enter anything (and a default is available)
            if user_selection == "" and default_path is not None:
                selected_model_path = default_path
                break

            # Check if user entered a number matching an item in the list
            try:
                idx_select = int(user_selection) - 1
                selected_model_path = model_files_paths[idx_select]
                print_selected_model(idx_select)
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
                print_selected_model(idx_select)
                break

            # If we get here, we didn't get a valid input. So warn user and repeat prompt
            print("", "", "Invalid selection!", sep="\n", flush=True)
            sleep(0.75)

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


# .....................................................................................................................


def load_init_prompts(path_to_json: str | None):

    # Initialize outputs
    ok_prompts = False
    prompts_dict = {}

    # Bail if we don't get a path
    if path_to_json is None:
        return ok_prompts, prompts_dict

    # Bail if the given path isn't valid
    if not osp.exists(path_to_json):
        print("", "Warning: Not using prompts JSON, path is invalid", f"@ {path_to_json}", sep="\n")
        return ok_prompts, prompts_dict

    try:
        # Try to load the given file as json data, expecting a dictionary!
        with open(path_to_json, "r") as infile:
            json_data = json.load(infile)
        if not isinstance(json_data, dict):
            raise TypeError("Prompt JSON must be a dictionary")

        # Make sure the json data has the expected keys
        req_keys = ("boxes", "fg_points", "bg_points")
        ok_prompts = all(key in json_data.keys() for key in req_keys)
        if not ok_prompts:
            raise ValueError(f"Prompt JSON must contain keys: {req_keys}")

        # If we get here, the json data is ok to use
        prompts_dict = json_data

    except Exception as err:
        print("", "Warning: Unable to load prompt json", f"@ {path_to_json}", "", str(err), sep="\n")

    return ok_prompts, prompts_dict


# .....................................................................................................................


def select_from_options(
    options_list: list[str],
    default_option: str | None = None,
    response_list: list | None = None,
    title_message: str = "Select option:",
    input_message: str = "Enter selection: ",
    allow_path_response: bool = False,
    allow_direct_response: bool = False,
    allow_text_match: bool = True,
    sleep_on_error_duration: float = 1,
) -> str:
    """
    Helper used to present a cli-based menu selector to user
    If a default option is given, the user can enter nothing to auto-select the default.

    A 'response_list' can be given, in which case when a user selects a menu item,
    the corresponding item from the response list will be returned.
    """

    # For convenience, use options as response if not given
    if response_list is None:
        response_list = tuple(options_list)

    # Sanity check, if a response list is given it must match the options
    num_options = len(options_list)
    if num_options != len(response_list):
        num_responses = len(response_list)
        raise ValueError(
            f"Selection error! Must have matching number of options ({num_options}) and responses ({num_responses})"
        )

    # Handle non-sense cases
    if num_options == 0:
        raise ValueError("Error, no options!")
    if num_options == 1:
        return response_list[0]

    # Sanity check, ignore defaults that don't appear in options
    if (default_option is not None) and (default_option not in options_list):
        default_option = None
    default_idx = 0 if default_option is None else options_list.index(default_option)

    # Build menu strings
    menu_strs_to_print = ["", title_message, ""]
    for idx, option_name in enumerate(options_list):
        is_default = option_name == default_option if default_option is not None else False
        menu_str = f"  {1+idx:>2}: {option_name}" if not is_default else f" *{1+idx:>2}: {option_name} (default)"
        menu_strs_to_print.append(menu_str)
    menu_strs_to_print.append("")

    # Build spacer text in case we need to write out what the user selected (on implicit selections)
    indicator_spacer_txt = "--> ".rjust(len(input_message))

    # Repeatedly ask user for input until they give use something valid
    out_response = "selection error"
    while True:
        # Provide menu prompt to user
        print(*menu_strs_to_print, sep="\n", flush=True)
        user_input_str = input(input_message).strip()

        # Interpret blank input as 'choose the default' if we have a default
        if user_input_str == "" and default_option is not None:
            out_response = response_list[default_idx]
            print(f"{indicator_spacer_txt}{default_option}", "", sep="\n", flush=True)
            break

        # Check if user entered a valid menu index
        if user_input_str.isnumeric():
            choice_idx = int(user_input_str) - 1
            is_bad_index = (choice_idx < 1) or (choice_idx > num_options)
            if is_bad_index:
                print("", f"Bad index, must be between 1 and {num_options}", "", sep="\n", flush=True)
                continue
            choice_option = options_list[choice_idx]
            out_response = response_list[choice_idx]
            print(f"{indicator_spacer_txt}{choice_option}", "", sep="\n", flush=True)
            break

        # Check if user input matches 1 entry in the menu
        if allow_text_match:
            is_text_match = [user_input_str in option_str for option_str in options_list]
            num_matches = sum(is_text_match)
            if num_matches == 1:
                match_idx = is_text_match.index(True)
                match_option = options_list[match_idx]
                out_response = response_list[match_idx]
                print(f"{indicator_spacer_txt}{match_option}", "", sep="\n", flush=True)
                break
            elif num_matches > 1:
                print("", f"Invalid entry! Multiple matches ({num_matches}) found...", "", sep="\n", flush=True)
                sleep(sleep_on_error_duration)
                continue
            pass

        # Allow direct string response
        if allow_direct_response:
            out_response = user_input_str
            break

        # If we get here, interpret user input as a file path if possible
        if allow_path_response and Path(user_input_str).exists():
            out_response = user_input_str
            break

        # If we can't interpret user input, warn and try continue looping
        print("", "Invalid entry! Enter an index from the list above", "", sep="\n", flush=True)
        sleep(sleep_on_error_duration)

    return out_response
