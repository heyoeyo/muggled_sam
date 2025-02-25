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

from time import perf_counter
from collections import deque
import cv2
import numpy as np
import torch
from lib.v2_sam.make_sam_v2 import make_samv2_from_original_state_dict

# Define pathing & device usage
mask_path = "/path/to/mask.png"
mask_image_path = "/path/to/mask_image.png"
video_path = "/path/to/video.mp4"
model_path = "/path/to/samv2_model.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Set image encoding config
imgenc_config_dict = {"max_side_length": 1024, "use_square_sizing": True}

# Load mask & corresponding image data
init_mask = cv2.imread(mask_path)
init_mask_image = cv2.imread(mask_image_path)

# Read first frame to verify video is ok, then reset playback
vcap = cv2.VideoCapture(video_path)
ok_frame, _ = vcap.read()
if not ok_frame:
    raise IOError("Bad first video frame!")
vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Set up model
print("Loading model...")
model_config_dict, sammodel = make_samv2_from_original_state_dict(model_path)
sammodel.to(device=device, dtype=dtype)

# Use initial prompt to begin segmenting an object
init_encoded_img, _, _ = sammodel.encode_image(init_mask_image, **imgenc_config_dict)
init_mem = sammodel.initialize_from_mask(init_encoded_img, init_mask)

# Set up data storage for prompted object (repeat this for each unique object)
prompt_mems = deque([init_mem])
prompt_ptrs = deque([])
prev_mems = deque([], maxlen=6)
prev_ptrs = deque([], maxlen=15)

# Process video frames
close_keycodes = {27, ord("q")}  # Esc or q to close
try:
    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(1, total_frames):

        # Read frames
        ok_frame, frame = vcap.read()
        if not ok_frame:
            break

        # Process video frames with model
        t1 = perf_counter()
        encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)
        obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
            encoded_imgs_list, prompt_mems, prompt_ptrs, prev_mems, prev_ptrs
        )
        t2 = perf_counter()
        print(f"Took {round(1000 * (t2 - t1))} ms")

        # Store object results for future frames
        if obj_score < 0:
            print("Bad object score! Implies broken tracking!")
        else:
            prev_mems.appendleft(mem_enc)
            prev_ptrs.appendleft(obj_ptr)

        # Create mask for display
        dispres_mask = torch.nn.functional.interpolate(
            mask_preds[:, best_mask_idx, :, :],
            size=frame.shape[0:2],
            mode="bilinear",
            align_corners=False,
        )
        disp_mask = ((dispres_mask > 0.0).byte() * 255).cpu().numpy().squeeze()
        disp_mask = cv2.cvtColor(disp_mask, cv2.COLOR_GRAY2BGR)

        # Show frame and mask, side-by-side
        cv2.imshow("Video Segmentation Result - q to quit", np.hstack((frame, disp_mask)))
        keypress = cv2.waitKey(1) & 0xFF
        if keypress in close_keycodes:
            break

except Exception as err:
    raise err

except KeyboardInterrupt:
    print("Closed by ctrl+c!")

finally:
    vcap.release()
    cv2.destroyAllWindows()
