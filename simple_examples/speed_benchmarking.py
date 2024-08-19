#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import lib  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to lib folder!")
import os
import torch
import numpy as np
from time import perf_counter
from lib.make_sam import make_sam_from_state_dict

# Define pathing
model_path = "/path/to/sam_v1_or_v2_model.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Image encoder settings
max_side_length = 1024
use_square_sizing = True

# Benchmarking settings
num_warmup_iterations = 5
num_image_encoder_iterations = 50
num_mask_generation_iterations = 100
if device == "cpu":
    num_warmup_iterations = 2
    num_image_encoder_iterations = num_image_encoder_iterations // 10
    num_mask_generation_iterations = num_mask_generation_iterations // 10

# Set up model
print(f"Loading model ({os.path.basename(model_path)})")
t1 = perf_counter()
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
sammodel.to(device=device, dtype=dtype)
t2 = perf_counter()
print("-> Loading took", round(1000 * (t2 - t1)), "ms")

# Model warm-up (excludes one-time VRAM/cache allocation from timing)
print("", f"Running warm-up ({device} / {dtype})", sep="\n", flush=True)
test_img = np.random.randint(0, 255, (max_side_length, max_side_length, 3), dtype=np.uint8)
for _ in range(num_warmup_iterations):
    encoded_img, _, _ = sammodel.encode_image(test_img, max_side_length, use_square_sizing)
    encoded_prompts = sammodel.encode_prompts([[(0.25, 0.25), (0.5, 0.5)]], [(0.5, 0.5)], [(0.75, 0.75)])
    _, _ = sammodel.generate_masks(encoded_img, encoded_prompts)
if torch.cuda.is_available():
    torch.cuda.synchronize()

# Time the image encoder
print("", f"Running image encoder ({num_image_encoder_iterations} iterations)", sep="\n", flush=True)
t1 = perf_counter()
for _ in range(num_image_encoder_iterations):
    encoded_img, _, _ = sammodel.encode_image(test_img, max_side_length, use_square_sizing)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
total_time_ms = round(1000 * (t2 - t1))
per_iter = total_time_ms / num_image_encoder_iterations
print("-> Image encoder took", total_time_ms, "ms", f"({per_iter} ms / iter)")

# Time prompt encoding + mask generation
print("", f"Generating masks ({num_mask_generation_iterations} iterations)", sep="\n", flush=True)
t1 = perf_counter()
for _ in range(num_mask_generation_iterations):
    encoded_prompts = sammodel.encode_prompts([], [(0.5, 0.5)], [])
    mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
total_time_ms = round(1000 * (t2 - t1))
per_iter = total_time_ms / num_mask_generation_iterations
print("-> Mask generation took", total_time_ms, "ms", f"({per_iter} ms / iter)")
