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
import cv2
import torch
from muggled_sam.make_sam import make_sam_from_state_dict

# Setup
batch_size = 4
image_path = "/path/to/image.jpg"
model_path = "/path/to/model.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Define prompts using xy coordinates normalized between 0 and 1
box_tlbr_norm_list = [[(0.25, 0.25), (0.75, 0.75)]]  # Format is: [(top-left xy), (bottom-right xy)]
fg_xy_norm_list = []  # Example: [(0.5, 0.5)]
bg_xy_norm_list = []

# Load image
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Error loading image from: {image_path}")

# Set up model
print("Loading model...")
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
sammodel.to(device=device, dtype=dtype)

# Set up image batch by just repeating the single input image
print(f"Encoding image batch... (batch size: {batch_size})")
with torch.inference_mode():
    image_tensor = sammodel.image_encoder.prepare_image(img_bgr, max_side_length=None, use_square_sizing=True)
    image_batch = image_tensor.repeat(batch_size, 1, 1, 1)
    encoded_img = sammodel.image_encoder(image_batch)

    # SAMv3 requires a 'samv2' projection step when used for direct mask predictions
    if sammodel.name == "samv3":
        encoded_img = sammodel.image_projection.project_v2(encoded_img)

# Process data
print("Generating masks...")
encoded_prompts = sammodel.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts)

# Feedback
is_samv2 = isinstance(encoded_img, (tuple, list))
tokens_shape = encoded_img[0].shape if is_samv2 else encoded_img.shape
print("")
print("Results:")
if torch.cuda.is_available():
    free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
    print("VRAM Usage:", (total_vram_bytes - free_vram_bytes) // 1_000_000)
print("Input image shape:", img_bgr.shape)
print("Pre-encoded image shape:", tuple(image_batch.shape))
print("Image tokens shape:", tuple(tokens_shape))
print("Mask results shape:", tuple(mask_preds.shape))
print("IoU results shape:", tuple(iou_preds.shape))
