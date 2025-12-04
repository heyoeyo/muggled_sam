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

# Define pathing
image_path = "/path/to/image.jpg"
model_path = "/path/to/model.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Define prompts using xy coordinates normalized between 0 and 1
box_tlbr_norm_list = [[(0.25, 0.25), (0.75, 0.75)]]  # Format is: [(top-left xy), (bottom-right xy)]
fg_xy_norm_list = []  # Example: [(0.5, 0.5)]
bg_xy_norm_list = []
mask_hint = None  # Example: torch.randn((1, 256, 256))

# Load image
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Error loading image from: {image_path}")

# Set up model
print("Loading model...")
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
sammodel.to(device=device, dtype=dtype)

# Process data
print("Generating masks...")
encoded_img, token_hw, preencode_img_hw = sammodel.encode_image(img_bgr, max_side_length=None, use_square_sizing=True)
encoded_prompts = sammodel.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts, mask_hint)

# Feedback
print("")
print("Results:")
print("Input image shape:", img_bgr.shape)
print("Pre-encoded image height & width:", tuple(preencode_img_hw))
print("Image tokens height & width:", tuple(token_hw))
print("Mask results shape:", tuple(mask_preds.shape))
print("IoU scores:", iou_preds[0].tolist())
print("")
print("Model config:")
print(*[f"  {k}: {v}" for k, v in model_config_dict.items()], sep="\n")
