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
from collections import defaultdict
import cv2
import numpy as np
import torch
from muggled_sam.make_sam import make_sam_from_state_dict
from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults
from muggled_sam.demo_helpers.bounding_boxes import get_2box_iou

# Define pathing & device usage
initial_frame_index = 0
video_path = "/path/to/video.mp4"
detection_model_path = "/path/to/sam3.pth"
tracking_model_path = None  # Can use a SAMv2 model! Leave as None to re-use the detection model for tracking
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# All coordinates are normalized between 0 and 1. Top left of image is (0,0), bottom-right is (1,1)
pos_box_xy1xy2_norm_list = []  # Format is: [[(x1, y1), (x2, y2)]]
neg_box_xy1xy2_norm_list = []
pos_point_xy_norm_list = []  # Format is [(x1, y1)]
neg_point_xy_norm_list = []
text_prompt = "person"

# Controls for detection/tracking
detect_every_n_frames = 10  # Set to None to only run once on startup
detection_score_threshold = 0.5
existing_box_iou_threshold = 0.25
remove_after_n_missed_frames = 5

# Controls for visualization
enable_detection_visualization = True
visualization_delay_ms = 500  # Set to 0 to force a full pause (requires any-keypress to unpause)
display_scale = 0.35

# Bundle re-used data together for ease of use
display_scale_dict = {"dsize": None, "fx": display_scale, "fy": display_scale}
track_imgenc_config_dict = {"max_side_length": None, "use_square_sizing": True}
detection_imgenc_config_dict = {"max_side_length": None, "use_square_sizing": True}
detection_prompts_dict = {
    "text": text_prompt,
    "box_xy1xy2_norm_list": pos_box_xy1xy2_norm_list,
    "point_xy_norm_list": pos_point_xy_norm_list,
    "negative_boxes_list": neg_box_xy1xy2_norm_list,
    "negative_points_list": neg_point_xy_norm_list,
}

# Read first frame to verify video is ok
vcap = cv2.VideoCapture(video_path)
vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # See: https://github.com/opencv/opencv/issues/26795
vcap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_index)
ok_frame, first_frame = vcap.read()
assert ok_frame, f"Could not read frames from video: {video_path}"

# Set up model
print("Loading model...")
model_config_dict, track_model = make_sam_from_state_dict(detection_model_path)
assert track_model.name == "samv3", "Error! Only SAMv3 models support object detection..."
track_model.to(device=device, dtype=dtype)
detmodel = track_model.make_detector_model()
print("  Done!")

# Allow loading of alternate tracking model
if tracking_model_path is not None:
    print("Loading separate tracking model...")
    _, track_model = make_sam_from_state_dict(tracking_model_path)
    assert track_model.name in ("samv2", "samv3"), "Only SAMv2/v3 are supported for video tracking"
    track_model.to(device=device, dtype=dtype)
    print("  Done!")

# Set up storage for tracking memory and keeping track of lost objects
memory_per_obj_dict = defaultdict(SAMVideoObjectResults.create)
missed_frames_per_obj_dict = defaultdict(int)

# Process video frames
detect_every_n_frames = 2**31 if detect_every_n_frames is None else detect_every_n_frames
stack_func = np.hstack if first_frame.shape[0] > first_frame.shape[1] else np.vstack
close_keycodes = {27, ord("q")}  # Esc or q to close
try:
    is_webcam = isinstance(video_path, int)
    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 100_000
    for idx_frame in range(initial_frame_index, total_frames):

        # Read frames
        ok_frame, frame = vcap.read()
        if not ok_frame:
            break
        display_frame = cv2.resize(frame, **display_scale_dict)

        # Start model inference timing
        t1 = perf_counter()

        # Encode image data for tracking (this is the heaviest part of video inference)
        encoded_imgs_list, _, _ = track_model.encode_image(frame, **track_imgenc_config_dict)

        # Advance video tracking for all known objects
        objs_to_remove_list = []
        masks_on_frame_list = []
        for idx_obj, obj_memory in memory_per_obj_dict.items():
            obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = track_model.step_video_masking(
                encoded_imgs_list, **obj_memory.to_dict()
            )

            # Skip storage for bad results
            obj_score = obj_score.item()
            if obj_score < 0:
                missed_frames_per_obj_dict[idx_obj] += 1
                if missed_frames_per_obj_dict[idx_obj] > remove_after_n_missed_frames:
                    objs_to_remove_list.append(idx_obj)
                continue
            missed_frames_per_obj_dict[idx_obj] = 0

            # Store memory encodings for continued tracking and 'best' mask for display
            obj_memory.store_frame_result(idx_frame, mem_enc, obj_ptr)
            masks_on_frame_list.append(mask_preds[0, best_mask_idx, :, :])

        # Run detections to pick up new objects
        no_tracked_objects = len(memory_per_obj_dict) == 0
        need_detection = ((idx_frame - initial_frame_index) % detect_every_n_frames) == 0 or no_tracked_objects
        if need_detection:
            print(f"  Performing detection update! (frame {idx_frame})")
            det_encimgs, _, _ = detmodel.encode_detection_image(frame, **detection_imgenc_config_dict)
            det_exemplars = detmodel.encode_exemplars(det_encimgs, **detection_prompts_dict)
            det_masks, det_boxes, _, _ = detmodel.generate_detections(
                det_encimgs, det_exemplars, detection_filter_threshold=detection_score_threshold
            )

            # If we get new detections, compare to existing objects to see if anything new has appeared
            num_detections = det_masks.shape[1]
            print(f"    -> Detected {num_detections} objects")
            if num_detections > 0:

                # Get bounding boxes of existing objects
                known_boxes_list = []
                for mask_tensor in masks_on_frame_list:
                    mask_uint8 = (mask_tensor[0] > 0).byte().cpu().numpy()
                    contours_list, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour = max(contours_list, key=cv2.contourArea) if len(contours_list) > 1 else contours_list[0]
                    box_x, box_y, box_w, box_h = cv2.boundingRect(contour)
                    box_xy1xy2_px = torch.tensor(((box_x, box_y), (box_x + box_w, box_y + box_h)))
                    box_xy1xy2_norm = box_xy1xy2_px / torch.tensor((mask_uint8.shape[1], mask_uint8.shape[0]))
                    known_boxes_list.append(box_xy1xy2_norm.to(det_boxes))

                # Any detection that doesn't overlap an existing object is assumed to be new
                is_new_obj_list = []
                for idx_det in range(num_detections):
                    new_box = det_boxes[0, idx_det]
                    is_known = any(get_2box_iou(new_box, b) > existing_box_iou_threshold for b in known_boxes_list)
                    is_new_obj_list.append(not is_known)

                # Draw visualization to show known objects vs. new detections
                if enable_detection_visualization:
                    scaled_h, scaled_w = display_frame.shape[0:2]
                    norm_to_px_scale = torch.tensor((scaled_w - 1, scaled_h - 1)).to(det_boxes)

                    # Drawn known boxes
                    for kidx, known_box_xy1xy2_norm in enumerate(known_boxes_list):
                        known_box_px = known_box_xy1xy2_norm * norm_to_px_scale
                        known_box_xy1, known_box_xy2 = known_box_px.int().cpu().numpy()
                        cv2.rectangle(display_frame, known_box_xy1, known_box_xy2, (255, 0, 255), 4)

                    # Draw detections
                    for det_idx, is_new_obj in enumerate(is_new_obj_list):
                        det_box_px = det_boxes[0, det_idx] * norm_to_px_scale
                        det_box_xy1, det_box_xy2 = det_box_px.int().cpu().numpy()
                        det_color = (0, 255, 0) if is_new_obj else (0, 255, 255)
                        cv2.rectangle(display_frame, det_box_xy1, det_box_xy2, det_color, 1)

                # Initialize new detections using the corresponding mask predictions
                next_new_idx = max(memory_per_obj_dict.keys()) + 1 if len(memory_per_obj_dict) > 0 else 0
                new_det_idxs_list = [det_idx for det_idx, is_new in enumerate(is_new_obj_list) if is_new]
                print(f"    -> Adding {len(new_det_idxs_list)} new objects")
                for idx_offset, det_idx in enumerate(new_det_idxs_list):
                    raw_det_mask = det_masks[0, det_idx]
                    init_mem = track_model.initialize_from_mask(encoded_imgs_list, raw_det_mask > 0)
                    new_idx = next_new_idx + idx_offset
                    memory_per_obj_dict[new_idx].store_prompt_result(idx_frame, init_mem)
                    masks_on_frame_list.append(raw_det_mask.unsqueeze(0))
                pass

        # Finish timing model inference
        t2 = perf_counter()
        print(f"Took {round(1000 * (t2 - t1))} ms for {len(memory_per_obj_dict)} objects")

        # Stop tracking objects that were marked for removal
        for idx_obj in objs_to_remove_list:
            memory_per_obj_dict.pop(idx_obj)
            missed_frames_per_obj_dict.pop(idx_obj)
            print("  -> Removed object:", idx_obj)

        # Combine all tracking masks for display
        combined_mask_result = np.zeros(display_frame.shape[0:2], dtype=bool)
        for mask_tensor in masks_on_frame_list:
            scaled_mask = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0),
                size=display_frame.shape[0:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_binary = (scaled_mask > 0.0).cpu().numpy().squeeze()
            combined_mask_result = np.bitwise_or(combined_mask_result, mask_binary)

        # Combine original image & mask result side-by-side for display
        combined_mask_result_uint8 = combined_mask_result.astype(np.uint8) * 255
        disp_mask = cv2.cvtColor(combined_mask_result_uint8, cv2.COLOR_GRAY2BGR)
        sidebyside_frame = stack_func((display_frame, disp_mask))

        # Show results
        cv2.imshow("Video Segmentation Result - q to quit", sidebyside_frame)
        display_delay_ms = 1 if not (need_detection and enable_detection_visualization) else visualization_delay_ms
        keypress = cv2.waitKey(display_delay_ms) & 0xFF
        if keypress in close_keycodes:
            break

except KeyboardInterrupt:
    pass

finally:
    vcap.release()
    cv2.destroyAllWindows()
