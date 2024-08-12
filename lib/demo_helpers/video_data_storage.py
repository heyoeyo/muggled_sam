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
class SAM2VideoBuffer:
    """Helper used to store SAM video results in a rolling buffer (avoids excessive memory usage)"""

    idx: deque[int]
    memory_history: deque[Tensor]
    pointer_history: deque[Tensor]

    @classmethod
    def create(cls, memory_history_length=6, pointer_history_length=15):
        idx_deck = deque([], maxlen=max(memory_history_length, pointer_history_length))
        mask_deck = deque([], maxlen=memory_history_length)
        pointer_deck = deque([], maxlen=pointer_history_length)
        return cls(idx_deck, mask_deck, pointer_deck)

    def store(self, frame_index, memory_encoding, object_pointer):
        self.idx.appendleft(frame_index)
        self.memory_history.appendleft(memory_encoding)
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


@dataclass
class SAM2VideoObjectResults:
    """Helper used to store both the prompt & per-frame buffers needed for video segmentation masking"""

    prompts_buffer: SAM2VideoBuffer
    prevframe_buffer: SAM2VideoBuffer

    @classmethod
    def create(cls, memory_history_length=6, pointer_history_length=15, prompt_history_length=1):
        prompts_buffer = SAM2VideoBuffer.create(prompt_history_length, prompt_history_length)
        prevframe_buffer = SAM2VideoBuffer.create(memory_history_length, pointer_history_length)
        return cls(prompts_buffer, prevframe_buffer)

    def store_prompt_result(self, frame_index, memory_encoding, object_pointer):
        """Used to store (initial) prompt results"""
        self.prompts_buffer.store(frame_index, memory_encoding, object_pointer)
        return self

    def store_result(self, frame_index, memory_encoding, object_pointer):
        """Used to store per-frame results history"""
        self.prevframe_buffer.store(frame_index, memory_encoding, object_pointer)
        return self

    def to_dict(self):
        """Helper used to convert stored data into a dictionary, so it can be used as a 'kwargs' argument"""
        return {
            "prompt_memory_encodings": self.prompts_buffer.memory_history,
            "prompt_object_pointers": self.prompts_buffer.pointer_history,
            "previous_memory_encodings": self.prevframe_buffer.memory_history,
            "previous_object_pointers": self.prevframe_buffer.pointer_history,
        }