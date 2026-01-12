#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def read_user_text_input(
    prompt: str = "Enter text prompt: ",
    prompt_for_exiting: str = "",
    allow_numeric_inputs: bool = True,
) -> tuple[bool, bool, str]:
    """
    Function used to read user text input from a terminal
    Also has support for allowing the user to 'exit' by
    entering a target input (default to '' empty input),
    as well as reading floating point inputs

    Returns:
        is_user_exiting, is_user_entering_float, raw_user_text
    """

    # Initialize outputs
    is_user_exit = False
    is_user_float = False
    user_txt_input = prompt_for_exiting

    # Ask user for text input from the terminal
    print("", flush=True)
    try:
        user_txt_input = input(prompt)
    except KeyboardInterrupt:
        pass

    # Bail if user inputs exit text
    is_user_exit = user_txt_input == prompt_for_exiting
    if is_user_exit:
        return is_user_exit, is_user_float, user_txt_input

    # See if we can interpret input as a number
    try:
        _ = float(user_txt_input)
        is_user_float = True
    except ValueError:
        pass

    return is_user_exit, is_user_float, user_txt_input
