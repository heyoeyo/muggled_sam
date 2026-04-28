#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import json
from pathlib import Path


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def save_unnested_json(save_path: str | Path, data_dict: dict) -> None:
    """
    Normally, the json 'indent' option will cause lists to be broken into
    separate lines for each list entry, which can be hard to read.
    This function tries to save a given dictionary in a more readable format.
    It makes strong assumptions about how to do this which is specific to
    saving the default data in this script!
    """

    # Try to format json a bit nicer on save
    json_str = json.dumps(data_dict)
    json_str = json_str.replace(' "', '\n "').replace("{", "{\n ").replace("}", "\n}")
    with open(save_path, "w") as outfile:
        outfile.write(json_str)

    return


def make_default_image_encoder_block_mapping():
    """
    Returns a dictionary of example block mappings for distilling the SAMv3 image encoder.
    Each mapping is given a name (e.g. '8_blocks'), and consists of
    a list of lists of integers. The outer-most list represents the number of 'stages'
    of the resulting image encoder while the inner lists represent which transformer
    blocks should be taken from the 'teacher' model to form a student model.
    For example:
        '8_blocks': [[0, 1], [2, 3], [4, 5], [6, 7]],
    -> The outer most list has 4 entries: [ [...], [...], [...], [...] ]
       which means this forms a 4-stage image encoder
    -> The inner lists indicate which blocks should be used to build each stage
    -> The first entry: [0, 1] means to take the 0-th and 1st transformer blocks from the teacher
       to form the first stage of the student
    -> The second stage is made from: [2, 3], which is blocks 2 and 3 from the teacher and so on...

    A 'reference' entry is included which indicates the structure of the original SAMv3 image encoder.
    This isn't meant to be used for distilling (it wouldn't do anything), but shows how the blocks
    are distributed in the original model.
    """
    return {
        "samv3": {
            "1_block": [[0]],
            "4_blocks": [[0], [1], [2], [7]],
            "8_blocks": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "12_blocks": [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 15]],
            "16_blocks": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            "20_blocks": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 31]],
            "24_blocks": [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 31]],
            "2_stages": [[0, 2, 4, 6, 8, 10, 12, 14, 15], [16, 18, 20, 22, 24, 26, 28, 30, 31]],
            "6_stages": [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
                [16, 17, 18, 19],
                [20, 21, 22, 31],
            ],
            "reference": [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31],
            ],
        }
    }


def make_default_text_encoder_block_mapping():
    """
    Returns a dictionary of block mappings for distilling the SAMv3 text encoder.
    Each mapping is given a name (e.g. '4_layers'), and consists of
    a list of integer, which correspond to which layers should be taken
    from the 'teacher' model to form a student model.
    For example:
        '4_layer': [0, 2, 4, 16]
    -> This would take layer index 0, 2, 4 and 16 from the teacher model
       to form a new 'student' model with only 4 layers

    A 'reference' entry is included which indicates the structure corresponding
    to the original SAMv3 text encoder (which is just every block in sequence).
    """
    return {
        "samv3": {
            "1_layer": tuple(range(1)),
            "4_layers": tuple(range(4)),
            "8_layers": tuple(range(8)),
            "12_layers": tuple(range(12)),
            "16_layers": tuple(range(16)),
            "20_layers": tuple(range(20)),
            "reference": tuple(range(24)),
        }
    }


def make_default_training_text_list():
    """Returns a list of text prompts to use for distilling the SAMv3 text encoder"""
    return [
        "visual",
        "human",
        "arm",
        "hand",
        "leg",
        "foot",
        "shoe",
        "face",
        "hair",
        "eye",
        "nose",
        "ear",
        "white",
        "gray",
        "grey",
        "black",
        "red",
        "orange",
        "bronze",
        "yellow",
        "gold",
        "green",
        "blue",
        "teal",
        "cyan",
        "purple",
        "indigo",
        "magenta",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic",
        "light",
        "fire",
        "hydrant",
        "street",
        "sign",
        "stop sign",
        "parking",
        "meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "hat",
        "backpack",
        "umbrella",
        "glasses",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports",
        "ball",
        "kite",
        "baseball",
        "bat",
        "glove",
        "skateboard",
        "surfboard",
        "tennis",
        "racket",
        "bottle",
        "plate",
        "wine",
        "glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted",
        "plant",
        "bed",
        "mirror",
        "dining",
        "table",
        "window",
        "desk",
        "toilet",
        "door",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell",
        "phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "blender",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy",
        "drier",
        "toothbrush",
        "brush",
    ]
