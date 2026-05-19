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
from time import perf_counter
import cv2
import torch
from muggled_sam.make_sam import make_sam_from_state_dict

# Setup
model_path = "/path/to/model.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Define all images to be batch processed & sizing
image_paths_list = ["/path/to/image_1.jpg", "/path/to/image_2.jpg", "/path/to/image_4.jpg"]  # etc.

# Define prompts using xy coordinates normalized between 0 and 1
box_xy1xy2_norm_list = [[(0.25, 0.25), (0.75, 0.75)]]  # Format is: [(top-left xy), (bottom-right xy)]
fg_xy_norm_list = []  # Example: [(0.5, 0.5)]
bg_xy_norm_list = []

# Load images & form image batch
imgs_list = []
for img_path in image_paths_list:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Error loading image from: {img_path}")
    imgs_list.append(img_bgr)
print(f"Got {len(imgs_list)} images for batching", "", sep="\n")

# Set up model
print("Loading model...")
sam_core = make_sam_from_state_dict(model_path)
interact_model = sam_core.get_interactive_context()
interact_model.to(device=device, dtype=dtype)

# Set up image batch
img_batch_tensor = interact_model.prepare_image_batch(imgs_list, max_side_length=None)
img_batch_size = img_batch_tensor.shape[0]

# Run model on batched input (equivalent to running each input separately in a for loop)
print("Processing image batch...")
t_start = perf_counter()
encoded_img = interact_model.encode_image(img_batch_tensor)
encoded_prompts = interact_model.encode_prompts(box_xy1xy2_norm_list, fg_xy_norm_list, bg_xy_norm_list)
mask_preds, iou_preds = interact_model.generate_masks(encoded_img, encoded_prompts)
t_end = perf_counter()

# Feedback
print("")
print("Results:")
if torch.cuda.is_available():
    free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
    print("VRAM Usage:", (total_vram_bytes - free_vram_bytes) // 1_000_000)
print("Pre-encoded batch shape:", tuple(img_batch_tensor.shape))
print("Mask results shape:", tuple(mask_preds.shape))
print("IoU results shape:", tuple(iou_preds.shape))
print("Time per image:", round(1000 * (t_end - t_start) / img_batch_size), "ms")
