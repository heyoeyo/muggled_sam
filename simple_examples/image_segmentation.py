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
from muggled_sam.demo_helpers.model_info import get_token_hw, get_preencoding_hw

# Define pathing
image_path = "/path/to/image.jpg"
model_path = "/path/to/model.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Define prompts using xy coordinates normalized between 0 and 1
box_xy1xy2_norm_list = [[(0.25, 0.25), (0.75, 0.75)]]  # Format is: [(top-left xy), (bottom-right xy)]
fg_xy_norm_list = []  # Example: [(0.5, 0.5)]
bg_xy_norm_list = []
mask_hint = None  # Example: torch.randn((1, 256, 256))
max_side_length = None
use_square_sizing = True

# Load image
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Error loading image from: {image_path}")

# Set up model
print("Loading model...")
sam_core = make_sam_from_state_dict(model_path)
interact_model = sam_core.get_interactive_context()
interact_model.to(device=device, dtype=dtype)

# Process data
print("Generating masks...")
encoded_img = interact_model.encode_image(img_bgr, max_side_length, use_square_sizing)
encoded_prompts = interact_model.encode_prompts(box_xy1xy2_norm_list, fg_xy_norm_list, bg_xy_norm_list)
mask_preds, iou_preds = interact_model.generate_masks(encoded_img, encoded_prompts, mask_hint)

# Get some info for reporting
token_hw = get_token_hw(encoded_img)
preencode_hw = get_preencoding_hw(interact_model, img_bgr, max_side_length, use_square_sizing)
model_config_dict = sam_core.get_config()

# Feedback
print("")
print("Results:")
print("Input image shape:", img_bgr.shape)
print("Pre-encoded image height & width:", preencode_hw)
print("Image tokens height & width:", token_hw)
print("Mask results shape:", tuple(mask_preds.shape))
print("IoU scores:", iou_preds[0].tolist())
print("")
print("Model config:")
print(*[f"  {k}: {v}" for k, v in model_config_dict.items()], sep="\n")
