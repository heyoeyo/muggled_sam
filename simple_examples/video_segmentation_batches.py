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
tracking_model_path = "/path/to/sam.pt"  # Use SAMv2 or SAMv3
detection_model_path = "/path/to/sam3.pt"
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
max_side_length_detect = None
max_side_length_track = None
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

# Load and set up tracking model
print("Loading tracking model...")
sam_core_track = make_sam_from_state_dict(tracking_model_path)
sam_core_track.to(device=device, dtype=dtype)
track_model = sam_core_track.get_tracking_context()

# Load and detection model (if different from tracking)
sam_core_detect = sam_core_track
if tracking_model_path != detection_model_path:
    print("Loading detection model...")
    sam_core_detect = make_sam_from_state_dict(detection_model_path)
    sam_core_detect.to(device=device, dtype=dtype)
detect_model = sam_core_detect.get_detector_context()

# Run initial detection to get objects to track
init_det_encimgs = detect_model.encode_image(first_frame, max_side_length_detect, use_square_sizing)
init_exemplars = detect_model.encode_exemplars(
    init_det_encimgs,
    text_prompt,
    pos_box_xy1xy2_norm_list,
    pos_point_xy_norm_list,
    neg_box_xy1xy2_norm_list,
    neg_point_xy_norm_list,
)
init_masks, _, _, _ = detect_model.generate_detections(init_det_encimgs, init_exemplars, detection_score_threshold)

# Warning for v3.1, which assumes multiplexing and will therefore misinterpret regular batching
is_v3p1 = hasattr(track_model, "multiplex_video_masking")
if is_v3p1:
    print(
        "",
        "WARNING:",
        "SAMv3.1 does not properly support regular batching!",
        "Only 1 object will be trackable. Use multiplexing instead",
        "",
        sep="\n",
    )
    # Take only the biggest mask for tracking (v3.1 will only track the 0-th index mask when using regular batching)
    idx_of_biggest_mask = [(init_masks > 0).sum(dim=(2, 3))[0].argmax()] if init_masks.shape[1] > 0 else []
    init_masks = init_masks[:, idx_of_biggest_mask]

# Sanity check
num_objects = init_masks.shape[1]
assert num_objects > 0, "No objects detected! Cannot begin tracking..."
print(f"Detected {num_objects} initial objects!")

# Set up 'memory bank'
init_encimgs = track_model.encode_image(first_frame, max_side_length_track, use_square_sizing)
init_mem = track_model.encode_prompt_memory_from_mask(init_encimgs, init_masks)
prompt_mems = deque([init_mem])
frame_mems = deque([], maxlen=6)

# Handle compilation if needed (doing this here to avoid including setup in compilation)
if enable_compilation:
    print("Compiling enabled! First run may take a while...", flush=True)
    track_model.enable_compilation(first_frame, max_side_length_track, use_square_sizing)

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
        mask_preds_bnhw, iou_preds_bn, obj_ptrs_bnc, obj_score_b = track_model.step_video_masking(
            encoded_img, prompt_mems, frame_mems
        )
        if is_using_cuda:
            torch.cuda.synchronize()
        t2 = perf_counter()
        txt_time_ms = f"{round(1000 * (t2 - t1))} ms"

        # Store object results for future frames
        ok_obj_track = obj_score_b > 0
        total_tracked_objs = ok_obj_track.sum().int().item()
        if total_tracked_objs > 0:
            encoded_mem = track_model.encode_frame_memory(encoded_img, mask_preds_bnhw, obj_ptrs_bnc, obj_score_b)
            frame_mems.append(encoded_mem)
        else:
            print("Bad object scores! No objects are being tracked!")
        txt_obj_count = f"{total_tracked_objs}/{num_objects} objects"

        # For clarity, we index out the 'best' masks while properly handling the multiplex dimension & missing objs
        num_multiplex = mask_preds_bnhw.shape[0]
        best_masks_mhw = mask_preds_bnhw[ok_obj_track]  # Remove 'missing' masks
        if best_masks_mhw.shape[0] == 0:
            best_masks_mhw = torch.full((1, 1, 4, 4), -10.0, device=best_masks_mhw.device, dtype=best_masks_mhw.dtype)

        # Create combined mask for display
        dispres_mask_m1hw = torch.nn.functional.interpolate(
            best_masks_mhw,
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
