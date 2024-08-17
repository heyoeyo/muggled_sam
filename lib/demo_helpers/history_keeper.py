#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import json


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class HistoryKeeper:
    """
    Class used to manage loading/saving to a 'history' dictionary file
    The point of this being to store & re-use important settings
    (for example, re-using image or model selection paths)
    """

    def __init__(self, file_dunder=None, history_file_name=".history"):

        # Set up history save file pathing
        folder_path = None
        if file_dunder is not None:
            folder_path = os.path.dirname(file_dunder) if os.path.isfile(file_dunder) else file_dunder
        filename = str(os.path.splitext(history_file_name)[0]).lower()
        filepath = filename if folder_path is None else os.path.join(folder_path, filename)

        self._filepath = filepath
        self._history_dict = {}
        self.reload()

    def reload(self):
        """Load and store results from an existing history file"""

        try:
            with open(self._filepath, "r") as infile:
                history_dict = json.load(infile)
        except FileNotFoundError:
            history_dict = {}
        self._history_dict = history_dict

        return self

    def read(self, key):
        """Read from the current copy of history data"""
        have_key = key in self._history_dict.keys()
        loaded_key = self._history_dict.get(key, None)
        return have_key, loaded_key

    def store(self, **key_value_kwargs):
        """Update and save history data"""

        # Check if we can store the new value
        new_history_dict = {**self._history_dict, **key_value_kwargs}
        is_valid_json = False
        try:
            json.dumps(self._history_dict)
            is_valid_json = True
        except TypeError:
            is_valid_json = False
            print("", "ERROR - Cannot store history, invalid as json:", new_history_dict, sep="\n")

        # Only re-write history data if the json is valid
        if is_valid_json:
            with open(self._filepath, "w") as outfile:
                json.dump(new_history_dict, outfile, indent=2)
            self._history_dict = new_history_dict

        return self
