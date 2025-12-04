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
from collections import deque
import cv2
import numpy as np
import torch
from muggled_sam.make_sam import make_sam_from_state_dict
from muggled_sam.demo_helpers.samurai import SimpleSamurai

# Define pathing & device usage
initial_frame_index = 0
video_path = "/path/to/video.mp4"
model_path = "/path/to/samv2_model.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Define prompts using xy coordinates normalized between 0 and 1
boxes_tlbr_norm_list = []  # Example:  [[(0.25, 0.25), (0.75, 0.75)]]
fg_xy_norm_list = [(0.5, 0.5)]
bg_xy_norm_list = []
imgenc_config_dict = {"max_side_length": None, "use_square_sizing": True}

# Control for SAMURAI tracking (value between 0 and 1)
# -> High values lead to 'lag' on SAMURAI box estimates
# -> Low values reduce the influence of SAMURAI (closer to base SAM tracking)
samurai_smoothness = 0.5

# Read first frame
vcap = cv2.VideoCapture(video_path)
vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # See: https://github.com/opencv/opencv/issues/26795
vcap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_index)
ok_frame, first_frame = vcap.read()
assert ok_frame, f"Could not read frames from video: {video_path}"

# Set up model
print("Loading model...")
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
assert sammodel.name in ("samv2", "samv3"), "Only SAMv2/v3 are supported for video segmentation"
sammodel.to(device=device, dtype=dtype)

# Use initial prompt to begin segmenting an object
init_encoded_img, _, _ = sammodel.encode_image(first_frame, **imgenc_config_dict)
init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(
    init_encoded_img, boxes_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list, mask_index_select=None
)

# Set up data storage for prompted object (repeat this for each unique object)
samurai = SimpleSamurai(init_mask, smoothness=samurai_smoothness)
prompt_mems = deque([init_mem])
prompt_ptrs = deque([init_ptr])
prev_mems = deque([], maxlen=6)
prev_ptrs = deque([], maxlen=15)

# Process video frames
stack_func = np.hstack if first_frame.shape[0] > first_frame.shape[1] else np.vstack
close_keycodes = {27, ord("q")}  # Esc or q to close
try:
    while True:

        # Read frames
        ok_frame, frame = vcap.read()
        if not ok_frame:
            break

        # Process video frames with model & added SAMURAI post-processing
        t1 = perf_counter()
        encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)
        is_mem_ok, best_mask_pred, mem_enc, obj_ptr, xy1xy2_kal = samurai.step_video_masking(
            sammodel, encoded_imgs_list, prompt_mems, prompt_ptrs, prev_mems, prev_ptrs
        )
        t2 = perf_counter()
        print(f"Took {round(1000 * (t2 - t1))} ms")

        # Only store memory that is considered valid by SAMURAI
        if is_mem_ok:
            prev_mems.appendleft(mem_enc)
            prev_ptrs.appendleft(obj_ptr)
        else:
            print("SAMURAI rejected memory!")

        # Create mask for display
        dispres_mask = torch.nn.functional.interpolate(
            best_mask_pred,
            size=frame.shape[0:2],
            mode="bilinear",
            align_corners=False,
        )
        disp_mask = ((dispres_mask > 0.0).byte() * 255).cpu().numpy().squeeze()
        disp_mask = cv2.cvtColor(disp_mask, cv2.COLOR_GRAY2BGR)

        # Overlay SAMURAI box prediction
        box_predict_color = (0, 255, 0) if is_mem_ok else (0, 0, 255)
        disp_mask = samurai.draw_box_prediction(disp_mask, xy1xy2_kal, box_predict_color)

        # Show frame and mask
        sidebyside = stack_func((frame, disp_mask))
        cv2.imshow("SAMURAI Segmentation Result - q to quit", cv2.resize(sidebyside, dsize=None, fx=0.5, fy=0.5))
        keypress = cv2.waitKey(1) & 0xFF
        if keypress in close_keycodes:
            break

except KeyboardInterrupt:
    print("Closed by ctrl+c!")

except Exception as err:
    raise err

finally:
    vcap.release()
    cv2.destroyAllWindows()
