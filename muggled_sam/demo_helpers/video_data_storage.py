#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from collections import UserDict, deque

# For legacy warning
from time import sleep
import sys

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Data types


class SAMVideoMemoryBank(UserDict):
    """
    Simple helper used to store prompt & per-frame memory encodings needed for video segmentation,
    it is a dictionary-like wrapper around two fixed-length 'rolling' lists (deques)

    One list holds 'prompt memory' while the other holds 'frame memory', these can
    be accessed as keys (like a dictionary) but are generally not meant to be interacted with.
    Instead, use 'store_prompt_result' or 'store_frame_result' to record memory encodings,
    use 'clear' to delete prior stored memory. If direct access is desirable, it's recommended
    to instead simply use lists/deques directly.

    As this class is dictionary-like, it can be provided directly to the video segmentation function
    using python unpacking (e.g. **memory_bank_instance).
    """

    def __init__(self, max_frame_memory: int = 6, max_prompt_memory: int = 32, store_as_recent_first: bool = False):
        super().__init__()
        self.data["frame_memory_encodings"] = deque([], maxlen=max_frame_memory)
        self.data["prompt_memory_encodings"] = deque([], maxlen=max_prompt_memory)
        self.data["is_recent_first"] = store_as_recent_first

    def store_prompt_result(self, memory_encoding: list[Tensor]):
        """Used to store prompt memory encodings"""
        if self.data["is_recent_first"]:
            self.data["prompt_memory_encodings"].appendleft(memory_encoding)
        else:
            self.data["prompt_memory_encodings"].append(memory_encoding)
        return self

    def store_frame_result(self, memory_encoding: list[Tensor]):
        """Used to store per-frame memory encodings"""
        if self.data["is_recent_first"]:
            self.data["frame_memory_encodings"].appendleft(memory_encoding)
        else:
            self.data["frame_memory_encodings"].append(memory_encoding)
        return self

    def get_num_memories(self) -> tuple[int, int]:
        """Read the length of currently stored memory data. Returns: num_prompt_memory, num_frame_memory"""
        num_prompt_mems = len(self.data["prompt_memory_encodings"])
        num_frame_mems = len(self.data["frame_memory_encodings"])
        return num_prompt_mems, num_frame_mems

    def clear(self, clear_prompt_memory: bool = True, clear_frame_memory: bool = True):
        if clear_frame_memory:
            self.data["frame_memory_encodings"].clear()
        if clear_prompt_memory:
            self.data["prompt_memory_encodings"].clear()
        return self

    def check_has_prompts(self) -> bool:
        """Helper used to check if there is any stored prompt memory"""
        return len(self.data["prompt_memory_encodings"]) > 0

    def to_dict(self) -> dict:
        """Legacy support. This class already acts like a dictionary. This function may be removed in the future..."""
        _legacy_warning(
            [
                "The 'to_dict' function on SAMVideoMemoryBank will be removed by July 2026",
                "The class now directly acts as a dictionary",
                "Please replace instances of:",
                "  **memory_bank.to_dict()",
                "With simply:",
                "  **memory_bank",
            ],
        )
        return {
            "prompt_memory_encodings": self.data["prompt_memory_encodings"],
            "frame_memory_encodings": self.data["frame_memory_encodings"],
        }


# ---------------------------------------------------------------------------------------------------------------------
# %% Legacy helpers

# Global used to store names of legacy function calls (so we don't repeat warning/delay)
_LEGACY_WARN_SET = set()


def _legacy_warning(message_list, sleep_delay_sec: float = 3.0):
    """Helper used to warn about removal of legacy code"""
    msg_key = message_list[0]
    if msg_key not in _LEGACY_WARN_SET:
        _LEGACY_WARN_SET.add(msg_key)
        print(
            "",
            f"{'*' * 12} WARNING {'*' * 12}",
            *message_list,
            "",
            sep="\n",
            flush=True,
            file=sys.stderr,
        )
        sleep(sleep_delay_sec)
    return
