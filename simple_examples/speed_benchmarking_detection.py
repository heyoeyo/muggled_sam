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
model_path = "/path/to/sam3.pt"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# All coordinates are normalized between 0 and 1. Top left of image is (0,0), bottom-right is (1,1)
pos_box_xy1xy2_norm_list = [[(0.25, 0.25), (0.75, 0.75)]]  # Format is: [[(x1, y1), (x2, y2)]]
neg_box_xy1xy2_norm_list = []
pos_point_xy_norm_list = [(0.5, 0.5)]
neg_point_xy_norm_list = []
text_prompt = "visual"
max_side_length = 1008
use_square_sizing = True
enable_compilation = False

# Benchmarking settings
num_warmup_iterations = 10
num_image_encoder_iterations = 50
num_exemplar_encoding_iterations = 100
num_detection_generation_iterations = 50
if device == "cpu":
    num_warmup_iterations = 1
    num_image_encoder_iterations = max(1, num_image_encoder_iterations // 20)
    num_exemplar_encoding_iterations = max(1, num_exemplar_encoding_iterations // 20)
    num_detection_generation_iterations = max(1, num_detection_generation_iterations // 20)

# Set up model
print(f"Loading model ({os.path.basename(model_path)})")
t1 = perf_counter()
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
assert sammodel.name == "samv3", f"Error, must use a SAMv3 model! (Got: {sammodel.name})"
detmodel = sammodel.make_detector_model()
detmodel.to(device=device, dtype=dtype)
t2 = perf_counter()
print("-> Loading took", round(1000 * (t2 - t1)), "ms")

# Fill in missing processing size, if needed
if max_side_length is None:
    prep_img = np.zeros((10, 10, 3), dtype=np.uint8)
    prep_tensor = detmodel.image_encoder.prepare_image(prep_img, None, use_square_sizing)
    max_side_length = int(max(prep_tensor.shape[-2:]))
print("", f"Using max side length: {max_side_length}px", f"Square sizing: {use_square_sizing}", sep="\n", flush=True)

# Set up testing inputs
test_img = np.random.randint(0, 255, (max_side_length, max_side_length, 3), dtype=np.uint8)
imgenc_config_dict = {"max_side_length": max_side_length, "use_square_sizing": max_side_length}
prompts_dict = {
    "text": text_prompt,
    "box_xy1xy2_norm_list": pos_box_xy1xy2_norm_list,
    "point_xy_norm_list": pos_point_xy_norm_list,
    "negative_boxes_list": neg_box_xy1xy2_norm_list,
    "negative_points_list": neg_point_xy_norm_list,
}

# Run compilation
if enable_compilation:
    print("", "Compiling model... (this may take a while)", sep="\n", flush=True)
    detmodel.enable_compilation(test_img, **imgenc_config_dict)

# Model warm-up (excludes one-time VRAM/cache allocation from timing)
print("", f"Running warm-up ({device} / {dtype})", sep="\n", flush=True)
for _ in range(num_warmup_iterations):
    encoded_imgs, token_hw, preencode_hw = detmodel.encode_detection_image(test_img, **imgenc_config_dict)
    encoded_exemplars = detmodel.encode_exemplars(encoded_imgs, **prompts_dict)
    mask_preds, box_preds, detection_scores, presence_score = detmodel.generate_detections(
        encoded_imgs, encoded_exemplars
    )
if torch.cuda.is_available():
    torch.cuda.synchronize()

# Time the image encoder
print("", f"Running image encoder ({num_image_encoder_iterations} iterations)", sep="\n", flush=True)
t1 = perf_counter()
for _ in range(num_image_encoder_iterations):
    encoded_imgs, _, _ = detmodel.encode_detection_image(test_img, **imgenc_config_dict)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
total_time_ms = 1000 * (t2 - t1)
per_iter = total_time_ms / num_image_encoder_iterations
print(f"-> Image encoder took {per_iter:.1f} ms / iter (total: {total_time_ms:.0f} ms)")

# Time exemplar encoding
print("", f"Encoding exemplars ({num_exemplar_encoding_iterations} iterations)", sep="\n", flush=True)
t1 = perf_counter()
for _ in range(num_exemplar_encoding_iterations):
    encoded_exemplars = detmodel.encode_exemplars(encoded_imgs, **prompts_dict)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
total_time_ms = 1000 * (t2 - t1)
per_iter = total_time_ms / num_exemplar_encoding_iterations
print(f"-> Exemplar encoding took {per_iter:.1f} ms / iter (total: {total_time_ms:.0f} ms)")

# Time detection/mask generation
print("", f"Generating detections ({num_detection_generation_iterations} iterations)", sep="\n", flush=True)
t1 = perf_counter()
for _ in range(num_detection_generation_iterations):
    mask_preds, _, _, _ = detmodel.generate_detections(encoded_imgs, encoded_exemplars)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
total_time_ms = 1000 * (t2 - t1)
per_iter = total_time_ms / num_detection_generation_iterations
print(f"-> Generating detections took {per_iter:.1f} ms / iter (total: {total_time_ms:.0f} ms)")

# Print VRAM usage if possible
if "cuda" in device:
    torch.cuda.empty_cache()
    free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
    curr_vram_mb = (total_vram_bytes - free_vram_bytes) // 1_000_000
    print("", f"Total VRAM usage: {curr_vram_mb} MB", sep="\n")
