#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from collections import deque

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Data types


class SAMVideoMemoryBank:
    """
    Simpler helper used to store prompt & per-frame memory encodings needed for video segmentation.
    It's just a wrapper around two fixed-length lists (deques)
    """

    def __init__(self, max_frame_memory: int = 6, max_prompt_memory: int = 32):
        self.frame_memory = deque([], maxlen=max_frame_memory)
        self.prompt_memory = deque([], maxlen=max_prompt_memory)

    def store_prompt_result(self, memory_encoding: list[Tensor]):
        """Used to store prompt memory encodings"""
        self.prompt_memory.append(memory_encoding)
        return self

    def store_frame_result(self, memory_encoding: list[Tensor]):
        """Used to store per-frame memory encodings"""
        self.frame_memory.append(memory_encoding)
        return self

    def to_dict(self) -> dict:
        """Helper used to convert stored data into a dictionary, so it can be used as a 'kwargs' argument"""
        return {
            "prompt_memory_encodings": self.prompt_memory,
            "frame_memory_encodings": self.frame_memory,
        }

    def get_num_memories(self) -> tuple[int, int]:
        """Read the length of currently stored memory data. Returns: num_prompt_memory, num_frame_memory"""
        num_prompt_mems = len(self.prompt_memory)
        num_frame_mems = len(self.frame_memory)
        return num_prompt_mems, num_frame_mems

    def clear(self, clear_prompt_memory: bool = True, clear_frame_memory: bool = True):
        if clear_frame_memory:
            self.frame_memory.clear()
        if clear_prompt_memory:
            self.prompt_memory.clear()
        return self

    def check_has_prompts(self) -> bool:
        """Helper used to check if there is any stored prompt memory"""
        return len(self.prompt_memory) > 0
