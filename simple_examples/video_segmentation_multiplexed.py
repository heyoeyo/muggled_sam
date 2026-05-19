#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is a hack to make this script work from outside the root project folder (without requiring install)
import os

try:
    import muggled_sam  # NOQA
except ModuleNotFoundError:
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

# Define pathing
initial_frame_index = 0
video_path = "/path/to/video.mp4"
model_path = "/path/to/sam3.1_multiplex.pt"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Detection prompts. Top left (x,y) of image is (0,0), bottom-right is (1,1)
pos_box_xy1xy2_norm_list = []  # Format is: [[(x1, y1), (x2, y2)]]
neg_box_xy1xy2_norm_list = []
pos_point_xy_norm_list = []  # Format is: [(x1, y1)]
neg_point_xy_norm_list = []
text_prompt = "person"

# Detection & tracking config
max_side_length_detect = 1008
max_side_length_track = 504  # Reduce this to increase speed at the cost of mask quality
use_square_sizing = True
detection_score_threshold = 0.5
display_scale = 0.5
enable_compilation = False

# Read first frame, will use to begin tracking
vcap = cv2.VideoCapture(video_path)
vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # See: https://github.com/opencv/opencv/issues/26795
vcap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_index)
ok_frame, first_frame = vcap.read()
assert ok_frame, f"Could not read frames from video: {video_path}"

# Load and set up model components
print("Loading model...")
sam_core = make_sam_from_state_dict(model_path)
sam_core.to(device=device, dtype=dtype)

# Set up contexts and warn about missing support if needed
track_model = sam_core.get_tracking_context()
if not hasattr(track_model, "step_video_masking_multiplex"):
    raise TypeError("Multiplexing is not supported on this model (requires SAMv3.1)")
detect_model = sam_core.get_detector_context()

# Run initial detection to get objects to track
init_encimgs = detect_model.encode_image(first_frame, max_side_length_detect, use_square_sizing)
init_exemplars = detect_model.encode_exemplars(
    init_encimgs,
    text_prompt,
    pos_box_xy1xy2_norm_list,
    pos_point_xy_norm_list,
    neg_box_xy1xy2_norm_list,
    neg_point_xy_norm_list,
)
init_masks, _, _, _ = detect_model.generate_detections(init_encimgs, init_exemplars, detection_score_threshold)

# Sanity check
num_objects = init_masks.shape[1]
assert num_objects > 0, "No objects detected! Cannot begin tracking..."
print(f"Detected {num_objects} initial objects!")

# Re-encode image at 'tracking' resolution to set up initial memory encoding
if max_side_length_detect != max_side_length_track:
    init_encimgs = track_model.encode_image(first_frame, max_side_length_track, use_square_sizing)

# Set up 'memory bank'
init_mem = track_model.encode_prompt_memory_from_mask(init_encimgs, init_masks)
prompt_mems = deque([init_mem])
frame_mems = deque([], maxlen=6)

# Handle compilation if needed (doing this here to avoid including setup in compilation)
if enable_compilation:
    print("Compiling enabled! First run may take a while...", flush=True)
    track_model.enable_compilation(
        first_frame,
        max_side_length_track,
        use_square_sizing,
        compile_memory_encoding=False,  # Currently causes problems for multiplexing
    )

# Process video frames
is_using_cuda = "cuda" in device
stack_func = np.hstack if first_frame.shape[0] > first_frame.shape[1] else np.vstack
close_keycodes = {27, ord("q")}  # Esc or q to close
try:
    while True:

        # Read frames
        ok_frame, frame = vcap.read()
        if not ok_frame:
            break
        scaled_frame = cv2.resize(frame, dsize=None, fx=display_scale, fy=display_scale)

        # Process video frames with model
        t1 = perf_counter()
        encoded_img = track_model.encode_image(frame, max_side_length_track, use_square_sizing)
        masks_m1hw, ious_m1, ptrs_m1c, obj_scores_m = track_model.step_video_masking_multiplex(
            encoded_img, prompt_mems, frame_mems, num_multiplex_objects=num_objects
        )
        if is_using_cuda:
            torch.cuda.synchronize()
        t2 = perf_counter()
        txt_time_ms = f"{round(1000 * (t2 - t1))} ms"

        # Store object results for future frames
        ok_obj_track = obj_scores_m > 0
        total_tracked_objs = ok_obj_track.sum().int().item()
        if total_tracked_objs > 0:
            encoded_mem = track_model.encode_frame_memory(encoded_img, masks_m1hw, ptrs_m1c, obj_scores_m)
            frame_mems.append(encoded_mem)
        else:
            print("Bad object scores! No objects are being tracked!")
        txt_obj_count = f"{total_tracked_objs}/{num_objects} objects"

        # For clarity, we index out the 'best' masks while properly handling the multiplex dimension & missing objs
        best_masks_m1hw = masks_m1hw[ok_obj_track]  # Remove 'missing' masks
        if best_masks_m1hw.shape[0] == 0:
            best_masks_m1hw = torch.full((1, 1, 4, 4), -10.0, device=masks_m1hw.device, dtype=masks_m1hw.dtype)

        # Create combined mask for display
        dispres_mask_m1hw = torch.nn.functional.interpolate(
            best_masks_m1hw,
            size=scaled_frame.shape[0:2],
            mode="bilinear",
            align_corners=False,
        )
        combined_masks_binary = (dispres_mask_m1hw > 0.0).any(dim=0).squeeze(0)
        disp_mask = (combined_masks_binary.byte() * 255).cpu().numpy()
        disp_mask = cv2.cvtColor(disp_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(disp_mask, txt_time_ms, (5, 18), 0, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(disp_mask, txt_obj_count, (5, 40), 0, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        # Show frame and masks
        sidebyside = stack_func((scaled_frame, disp_mask))
        cv2.imshow("Results - q to quit", sidebyside)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress in close_keycodes:
            break

except KeyboardInterrupt:
    print("Cancelled...")

finally:
    vcap.release()
    cv2.destroyAllWindows()
