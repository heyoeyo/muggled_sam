#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from collections import deque
from dataclasses import dataclass

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Data types


@dataclass
class SAMVideoBuffer:
    """Helper used to store SAM video results in a rolling buffer (avoids excessive memory usage)"""

    idx: deque[int]
    memory_history: deque[Tensor]
    pointer_history: deque[Tensor]

    @classmethod
    def create(cls, memory_history_length: int = 6, pointer_history_length: int = 15):
        idx_deck = deque([], maxlen=max(memory_history_length, pointer_history_length))
        mask_deck = deque([], maxlen=memory_history_length)
        pointer_deck = deque([], maxlen=pointer_history_length)
        return cls(idx_deck, mask_deck, pointer_deck)

    def store(self, frame_index: int, memory_encoding: Tensor, object_pointer: Tensor | None = None):
        self.idx.appendleft(frame_index)
        self.memory_history.appendleft(memory_encoding)
        if object_pointer is not None:
            self.pointer_history.appendleft(object_pointer)
        return self

    def set_memory_history(self, memory_history_length: int):

        new_mask_deck = deque([], maxlen=max(0, memory_history_length))
        new_mask_deck.extendleft(reversed(self.memory_history))
        self.memory_history = new_mask_deck

        return self

    def set_pointer_history(self, pointer_history_length: int):

        new_pointer_deck = deque([], maxlen=max(0, pointer_history_length))
        new_pointer_deck.extendleft(reversed(self.pointer_history))
        self.pointer_history = new_pointer_deck

        return self

    def clear(self, clear_memories: bool = True, clear_pointers: bool = True):

        if clear_memories:
            mem_history = self.memory_history.maxlen
            self.memory_history = deque([], maxlen=mem_history)
        if clear_pointers:
            ptr_history = self.pointer_history.maxlen
            self.pointer_history = deque([], maxlen=ptr_history)

        return self


@dataclass
class SAMVideoObjectResults:
    """Helper used to store both the prompt & per-frame buffers needed for video segmentation masking"""

    prompts_buffer: SAMVideoBuffer
    prevframe_buffer: SAMVideoBuffer

    @classmethod
    def create(cls, memory_history_length: int = 6, pointer_history_length: int = 15, prompt_history_length: int = 32):
        prompts_buffer = SAMVideoBuffer.create(prompt_history_length, prompt_history_length)
        prevframe_buffer = SAMVideoBuffer.create(memory_history_length, pointer_history_length)
        return cls(prompts_buffer, prevframe_buffer)

    def store_prompt_result(self, frame_index: int, memory_encoding: Tensor, object_pointer: Tensor | None = None):
        """Used to store (initial) prompt results"""
        self.prompts_buffer.store(frame_index, memory_encoding, object_pointer)
        return self

    def store_frame_result(self, frame_index: int, memory_encoding: Tensor, object_pointer: Tensor | None = None):
        """Used to store per-frame results history"""
        self.prevframe_buffer.store(frame_index, memory_encoding, object_pointer)
        return self

    def to_dict(self) -> dict:
        """Helper used to convert stored data into a dictionary, so it can be used as a 'kwargs' argument"""
        return {
            "prompt_memory_encodings": self.prompts_buffer.memory_history,
            "prompt_object_pointers": self.prompts_buffer.pointer_history,
            "previous_memory_encodings": self.prevframe_buffer.memory_history,
            "previous_object_pointers": self.prevframe_buffer.pointer_history,
        }

    def get_num_memories(self) -> tuple[int, int]:
        """Read the length of currently stored memory data. Returns: num_prompt_memory, num_prevframe_memory"""

        num_prompt_mems = len(self.prompts_buffer.memory_history)
        num_prevframe_mems = len(self.prevframe_buffer.memory_history)

        return num_prompt_mems, num_prevframe_mems

    def get_num_pointers(self) -> tuple[int, int]:
        """
        Read the length of currently stored object pointer data
        Returns: num_prompt_pointers, num_prevframe_pointers
        """

        num_prompt_pointers = len(self.prompts_buffer.pointer_history)
        num_prevframe_pointers = len(self.prevframe_buffer.pointer_history)

        return num_prompt_pointers, num_prevframe_pointers

    def check_has_prompts(self) -> bool:
        """Helper used to check if there is any stored prompt memory"""
        return len(self.prompts_buffer.memory_history) > 0
