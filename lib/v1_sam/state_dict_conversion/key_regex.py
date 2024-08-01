#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import re


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def _make_hashnumber_finder_pattern(input_str):
    """
    Converts strings of the form:

        "some.text.#.more.words"

    Into a regex pattern that will match to the given string, but with
    instances of '#' replaced with a regex pattern for finding numbers
    For example, the string above is converted to:

        "some\.text\.(\d+)\.more\.words"

    -> Each dot '.' is replaced with dot literal (to avoid interpretting as regex '.' wildcard)
    -> The '#' is replaced with regex pattern: (\d+), which matches to any number of digits
    """

    # Include 'any number' check
    pattern_str = re.escape(input_str)
    pattern_str = pattern_str.replace("\#", "(\d+)")

    return pattern_str


# .....................................................................................................................


def has_prefix(input_string, prefix_str):
    """
    Search for strings starting with a given string. Equivalent
    to str.startswith(prefix_str) for simple strings.

    Can include '#' character to look for matches with 'any number'.
    For example: prefix_str = "layer.#.block" will match to:
        "layer.0.block", "layer.1.block", "layer.2.block", etc...
    """

    # Add 'startswith' specific component
    num_finder_pattern = _make_hashnumber_finder_pattern(prefix_str)
    pattern_str = "".join(["^", num_finder_pattern])
    re_pattern = re.compile(pattern_str)

    return re_pattern.match(input_string) is not None


# .....................................................................................................................


def replace_prefix(input_str, old_prefix_str, new_prefix_str):
    """
    Function used to replace a string prefix with another, however, the target prefix
    strings can have '#' placeholders to indicate arbitrary numbers.
    """

    # Check if we need to match #'s between old/new prefix
    num_hash_old = old_prefix_str.count("#")
    num_hash_new = new_prefix_str.count("#")
    if num_hash_old == num_hash_new:
        raise NotImplementedError("Haven't implemented auto-number replacement")

    elif num_hash_new > 0:
        raise ValueError("Cannot handle new prefix containing '#' -> Not sure how to match to old prefix")

    num_finder_pattern = _make_hashnumber_finder_pattern(old_prefix_str)
    pattern_str = "".join(["^", num_finder_pattern])
    re_pattern = re.compile(pattern_str)

    return re_pattern.sub(new_prefix_str, input_str)


# .....................................................................................................................


def get_nth_integer(input_str, nth_occurrence_starting_from_0=0):
    """
    Function which pulls specific integers from a given string, indexed
    by order of appearance (left-to-right). For example, we could pull
    various numbers from the following string:
        ex_str = "abc.5.xyz.2.aa.bb[0]"
          get_nth_integer(ex_str, 0) -> 5
          get_nth_integer(ex_str, 1) -> 2
          get_nth_integer(ex_str, 2) -> 0

    Raises an index error if there is no nth integer!
    """

    pattern_str = r"\d+"
    re_pattern = re.compile(pattern_str)
    matches = re_pattern.finditer(input_str)

    n_iter = range(1 + nth_occurrence_starting_from_0)
    for n, match in zip(n_iter, matches):
        if n == nth_occurrence_starting_from_0:
            return int(match.group())

    # We get here if we don't have enough matches to find the nth one!
    raise IndexError(f"Couldn't find nth ({nth_occurrence_starting_from_0}) integer: {input_str}")


# .....................................................................................................................


def find_match_by_lut(input_str, from_to_lut):
    """
    Takes an input string and a 'from-to' dictionary
    Then searches the input for each key ('from') in the dictionary,
    if a match is found, the function returns the corresponding value ('to')

    Note: Only the 'to' string is returned (i.e. none of the original input string is returned)
    """

    has_from_str = False
    target_from_str = None
    new_to_str = None

    for target_from_str, new_to_str in from_to_lut.items():
        has_from_str = target_from_str in input_str
        if has_from_str:
            return has_from_str, target_from_str, new_to_str

    return has_from_str, target_from_str, new_to_str


# .....................................................................................................................


def get_suffix_terms(input_str, num_suffix_terms=1):
    """
    Takes an input string and extras the last 'n' period-separated terms.
    For example, given the string:
        input_str = "layer.0.block.1.fc1.weight"

    Then: get_suffix_terms(input_str, 3) would return the last 3 terms:
        "1.fc1.weight"

    Note that this also works with negative 'n', in which case it returns
    all but the 'n' first terms. For example: get_suffix_terms(input_str, -2)
    will remove the first 2 terms:
        "block.1.fc1.weight"
    """

    return ".".join(input_str.split(".")[-num_suffix_terms:])


# .....................................................................................................................
