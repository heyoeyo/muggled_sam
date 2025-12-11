#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import muggled_sam  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "muggled_sam" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to muggled_sam folder!")
import os
import torch
import numpy as np
from time import perf_counter
from muggled_sam.make_sam import make_sam_from_state_dict

# Define pathing
model_path = "/path/to/model.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Image encoder settings
max_side_length = None
use_square_sizing = True

# Benchmarking settings
num_warmup_iterations = 5
num_image_encoder_iterations = 50
num_mask_generation_iterations = 100
if device == "cpu":
    num_warmup_iterations = 2
    num_image_encoder_iterations = num_image_encoder_iterations // 10
    num_mask_generation_iterations = num_mask_generation_iterations // 10

# Get initial VRAM usage
initial_vram_mb = 0
if "cuda" in device:
    free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
    initial_vram_mb = (total_vram_bytes - free_vram_bytes) // 1_000_000

# Set up model
print(f"Loading model ({os.path.basename(model_path)})")
t1 = perf_counter()
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
sammodel.to(device=device, dtype=dtype)
t2 = perf_counter()
print("-> Loading took", round(1000 * (t2 - t1)), "ms")

# Fill in missing processing size, if needed
if max_side_length is None:
    prep_img = np.zeros((10, 10, 3), dtype=np.uint8)
    prep_tensor = sammodel.image_encoder.prepare_image(prep_img, None, use_square_sizing)
    max_side_length = int(max(prep_tensor.shape[-2:]))
print("", f"Using max side length: {max_side_length}px", f"Square sizing: {use_square_sizing}", sep="\n", flush=True)

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

# Print VRAM usage if possible
if "cuda" in device:
    free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
    curr_vram_mb = (total_vram_bytes - free_vram_bytes) // 1_000_000
    vram_usage = curr_vram_mb - initial_vram_mb
    print("", f"VRAM: {vram_usage} MB", sep="\n")
