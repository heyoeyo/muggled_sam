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

# Define pathing & device usage
initial_frame_index = 0
video_path = "/path/to/video.mp4"
model_path = "/path/to/sam_model.pt"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Define prompts using xy coordinates normalized between 0 and 1
boxes_xy1xy2_norm_list = []  # Example:  [[(0.25, 0.25), (0.75, 0.75)]]
fg_xy_norm_list = [(0.5, 0.5)]
bg_xy_norm_list = []
imgenc_config_dict = {"max_side_length": None, "use_square_sizing": True}
enable_compilation = False
display_scale = 0.5

# Read first frame
vcap = cv2.VideoCapture(video_path)
vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # See: https://github.com/opencv/opencv/issues/26795
vcap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_index)
ok_frame, first_frame = vcap.read()
assert ok_frame, f"Could not read frames from video: {video_path}"

# Set up model
print("Loading model...")
model_config_dict, sam_core = make_sam_from_state_dict(model_path)
track_model = sam_core.get_tracking_context()
track_model.to(device=device, dtype=dtype)

# Handle compilation if needed
if enable_compilation:
    print("Compiling enabled! First run may take a while...", flush=True)
    track_model.enable_compilation(first_frame, **imgenc_config_dict)

# Use initial prompt to begin segmenting an object
init_encoded_img, _, _ = track_model.encode_image(first_frame, **imgenc_config_dict)
_, init_mem = track_model.encode_prompt_memory(
    init_encoded_img, boxes_xy1xy2_norm_list, fg_xy_norm_list, bg_xy_norm_list, mask_index=None
)

# Set up 'memory bank' for tracked object (repeat this for each unique object)
prompt_mems = deque([init_mem])
frame_mems = deque([], maxlen=6)

# Process video frames
is_using_cuda = "cuda" in device
stack_func = np.hstack if first_frame.shape[0] > first_frame.shape[1] else np.vstack
close_keycodes = {27, ord("q")}  # Esc or q to close
try:
    is_webcam = isinstance(video_path, int)
    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 100_000
    for frame_idx in range(1 + initial_frame_index, total_frames):

        # Read frames
        ok_frame, frame = vcap.read()
        if not ok_frame:
            break

        # Process video frames (with timing)
        t1 = perf_counter()
        encoded_img, _, _ = track_model.encode_image(frame, **imgenc_config_dict)
        if is_using_cuda:
            torch.cuda.synchronize()
        t2 = perf_counter()
        masks_bnhw, ious_bn, ptrs_bnc, obj_score_b = track_model.step_video_masking(
            encoded_img, prompt_mems, frame_mems, return_best_only=False
        )
        if is_using_cuda:
            torch.cuda.synchronize()
        t3 = perf_counter()
        print(f"Encode: {round(1000 * (t2 - t1))} ms | Track: {round(1000 * (t3 - t2))} ms")

        # Pick the 'best' prediction. Equivalent to using 'return_best_only=True' above
        # -> Manually indexing like this gives access to all masks, which may be desirable in some cases
        # -> This approach also allows for custom mask selection logic (see: SAMURAI, for example)
        best_idx = ious_bn[0].argmax()
        best_mask = masks_bnhw[:, [best_idx], :, :]
        best_iou_float = float(ious_bn[0, best_idx])
        score_float = float(obj_score_b[0])

        # Encode and store memory for future frame tracking
        if score_float > 0:
            encoded_mem = track_model.encode_frame_memory(encoded_img, masks_bnhw, ptrs_bnc, obj_score_b, best_idx)
            frame_mems.append(encoded_mem)
        else:
            print("Bad object score! Implies broken tracking!")

        # Create mask for display
        scaled_frame = cv2.resize(frame, dsize=None, fx=display_scale, fy=display_scale)
        dispres_mask = torch.nn.functional.interpolate(
            best_mask,
            size=scaled_frame.shape[0:2],
            mode="bilinear",
            align_corners=False,
        )
        disp_mask = ((dispres_mask > 0.0).byte() * 255).cpu().numpy().squeeze()
        disp_mask = cv2.cvtColor(disp_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(disp_mask, f"ObjScore: {score_float:.1f}", (5, 16), 0, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(disp_mask, f"IoU: {best_iou_float:.3f}", (5, 34), 0, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        # Show frame and mask
        sidebyside = stack_func((scaled_frame, disp_mask))
        cv2.imshow("Video Segmentation Result - q to quit", sidebyside)
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
