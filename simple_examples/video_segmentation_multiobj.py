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
from collections import defaultdict
import cv2
import numpy as np
import torch
from muggled_sam.make_sam import make_sam_from_state_dict
from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults


# Define pathing & device usage
video_path = "/path/to/video.mp4"
model_path = "/path/to/samv2_model.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Define image processing config (shared for all video frames)
imgenc_config_dict = {"max_side_length": 1024, "use_square_sizing": True}

# For demo purposes, we'll define all prompts ahead of time and store them per frame index & object
# -> First level key (e.g. 0, 90, 140) represents the frame index where the prompts should be applied
# -> Second level key (e.g. 'obj1', 'obj2') represents which 'object' the prompt belongs to for tracking purposes
prompts_per_frame_index = {
    0: {
        "obj1": {
            "box_tlbr_norm_list": [],
            "fg_xy_norm_list": [(0.79, 0.70)],
            "bg_xy_norm_list": [],
        }
    },
    90: {
        "obj2": {
            "box_tlbr_norm_list": [[(0.06, 0.62), (0.22, 0.80)]],
            "fg_xy_norm_list": [],
            "bg_xy_norm_list": [],
        }
    },
    140: {
        "obj3": {
            "box_tlbr_norm_list": [[(0.01, 0.61), (0.21, 0.85)]],
            "fg_xy_norm_list": [],
            "bg_xy_norm_list": [],
        }
    },
}
enable_prompt_visualization = True
# *** These prompts are set up for a video of horses available from pexels.com ***
# https://www.pexels.com/video/horses-running-on-grassland-4215784/
# By: Adrian Hoparda

# Set up memory storage for tracked objects
# -> Assumes each object is represented by a unique dictionary key (e.g. 'obj1')
# -> This holds both the 'prompt' & 'recent' memory data needed for tracking!
memory_per_obj_dict = defaultdict(SAMVideoObjectResults.create)

# Read first frame to check that we can read from the video, then reset playback
vcap = cv2.VideoCapture(video_path)
vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # See: https://github.com/opencv/opencv/issues/26795
ok_frame, first_frame = vcap.read()
assert ok_frame, f"Could not read frames from video: {video_path}"
vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Set up model
print("Loading model...")
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
assert sammodel.name in ("samv2", "samv3"), "Only SAMv2/v3 are supported for video segmentation"
sammodel.to(device=device, dtype=dtype)

# Process video frames
stack_func = np.hstack if first_frame.shape[0] > first_frame.shape[1] else np.vstack
close_keycodes = {27, ord("q")}  # Esc or q to close
try:
    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(total_frames):

        # Read frames
        ok_frame, frame = vcap.read()
        if not ok_frame:
            print("", "Done! No more frames...", sep="\n")
            break

        # Encode frame data (shared for all objects)
        encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)

        # Generate & store prompt memory encodings for each object as needed
        prompts_dict = prompts_per_frame_index.get(frame_idx, None)
        if prompts_dict is not None:

            # Loop over all sets of prompts for the current frame
            for obj_key_name, obj_prompts in prompts_dict.items():
                print(f"Generating prompt for object: {obj_key_name} (frame {frame_idx})")
                init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(encoded_imgs_list, **obj_prompts)
                memory_per_obj_dict[obj_key_name].store_prompt_result(frame_idx, init_mem, init_ptr)

                # Draw prompts for debugging
                if enable_prompt_visualization:
                    prompt_vis_frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
                    norm_to_px_factor = np.float32((prompt_vis_frame.shape[1] - 1, prompt_vis_frame.shape[0] - 1))
                    for xy_norm in obj_prompts.get("fg_xy_norm_list", []):
                        xy_px = np.int32(xy_norm * norm_to_px_factor)
                        cv2.circle(prompt_vis_frame, xy_px, 3, (0, 255, 0), -1)
                    for xy_norm in obj_prompts.get("bg_xy_norm_list", []):
                        xy_px = np.int32(xy_norm * norm_to_px_factor)
                        cv2.circle(prompt_vis_frame, xy_px, 3, (0, 0, 255), -1)
                    for xy1_norm, xy2_norm in obj_prompts.get("box_tlbr_norm_list", []):
                        xy1_px = np.int32(xy1_norm * norm_to_px_factor)
                        xy2_px = np.int32(xy2_norm * norm_to_px_factor)
                        cv2.rectangle(prompt_vis_frame, xy1_px, xy2_px, (0, 255, 255), 2)

                    # Show prompt in it's own window and close after viewing
                    wintitle = f"Prompt ({obj_key_name}) - Press key to continue"
                    cv2.imshow(wintitle, prompt_vis_frame)
                    cv2.waitKey(0)
                    cv2.destroyWindow(wintitle)

        # Update tracking using newest frame
        combined_mask_result = np.zeros(frame.shape[0:2], dtype=bool)
        for obj_key_name, obj_memory in memory_per_obj_dict.items():
            obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
                encoded_imgs_list, **obj_memory.to_dict()
            )

            # Skip storage for bad results (often due to occlusion)
            obj_score = obj_score.item()
            if obj_score < 0:
                print(f"Bad object score for {obj_key_name}! Skipping memory storage...")
                continue

            # Store 'recent' memory encodings from current frame (helps track objects with changing appearance)
            # -> This can be commented out and tracking may still work, if object doesn't change much
            obj_memory.store_frame_result(frame_idx, mem_enc, obj_ptr)

            # Add object mask prediction to 'combine' mask for display
            # -> This is just for visualization, not needed for tracking
            obj_mask = torch.nn.functional.interpolate(
                mask_preds[:, best_mask_idx, :, :],
                size=combined_mask_result.shape,
                mode="bilinear",
                align_corners=False,
            )
            obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()
            combined_mask_result = np.bitwise_or(combined_mask_result, obj_mask_binary)

        # Combine original image & mask result side-by-side for display
        combined_mask_result_uint8 = combined_mask_result.astype(np.uint8) * 255
        disp_mask = cv2.cvtColor(combined_mask_result_uint8, cv2.COLOR_GRAY2BGR)
        sidebyside_frame = stack_func((frame, disp_mask))
        sidebyside_frame = cv2.resize(sidebyside_frame, dsize=None, fx=0.5, fy=0.5)

        # Show result
        cv2.imshow("Video Segmentation Result - q to quit", sidebyside_frame)
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
