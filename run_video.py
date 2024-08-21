#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse
import os.path as osp
from time import perf_counter
from enum import Enum

import torch
import cv2

from lib.make_sam import make_sam_from_state_dict
from lib.v2_sam.sam_v2_model import SAMV2Model

from lib.demo_helpers.shared_ui_layout import PromptUIControl, PromptUI
from lib.demo_helpers.ui.window import DisplayWindow, KEY
from lib.demo_helpers.ui.video import LoopingVideoReader, LoopingVideoPlaybackSlider, ValueChangeTracker
from lib.demo_helpers.ui.layout import HStack, VStack
from lib.demo_helpers.ui.buttons import ToggleButton, ImmediateButton, force_same_button_width
from lib.demo_helpers.ui.static import StaticMessageBar
from lib.demo_helpers.ui.text import ValueBlock

from lib.demo_helpers.misc import PeriodicVRAMReport, make_device_config, get_default_device_string
from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.contours import get_contours_from_mask
from lib.demo_helpers.video_data_storage import SAM2VideoObjectResults


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_prompts_path = None
default_display_size = 900
default_base_size = 1024
default_max_memory_history = 6
default_max_pointer_history = 15
default_max_prompt_history = 8

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to run Segment-Anything (SAM) on a single image")
parser.add_argument("-i", "--image_path", default=default_image_path, help="Path to input image")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Path to SAM model weights")
parser.add_argument(
    "-s",
    "--display_size",
    default=default_display_size,
    type=int,
    help=f"Controls size of displayed results (default: {default_display_size})",
)
parser.add_argument(
    "-d",
    "--device",
    default=default_device,
    type=str,
    help=f"Device to use when running model, such as 'cpu' (default: {default_device})",
)
parser.add_argument(
    "-f32",
    "--use_float32",
    default=False,
    action="store_true",
    help="Use 32-bit floating point model weights. Note: this doubles VRAM usage",
)
parser.add_argument(
    "-ar",
    "--use_aspect_ratio",
    default=False,
    action="store_true",
    help="Process the image at it's original aspect ratio",
)
parser.add_argument(
    "-b",
    "--base_size_px",
    default=default_base_size,
    type=int,
    help="Override base model size (default {default_base_size})",
)
parser.add_argument(
    "--max_memories",
    default=default_max_memory_history,
    type=int,
    help="Maximum number of previous-frame memory encodings to store (default {default_max_memory_history})",
)
parser.add_argument(
    "--max_pointers",
    default=default_max_pointer_history,
    type=int,
    help="Maximum number of previous-frame object pointers to store (default {default_max_pointer_history})",
)
parser.add_argument(
    "--max_prompts",
    default=default_max_prompt_history,
    type=int,
    help="Maximum number of prompts to store for video segmentation (default {default_max_prompt_history})",
)
parser.add_argument(
    "--keep_bad_objscores",
    default=False,
    action="store_true",
    help="If set, masks associated with low object-scores will NOT be discarded",
)
parser.add_argument(
    "--keep_history_on_new_prompts",
    default=False,
    action="store_true",
    help="If set, existing history data will not be cleared when adding new prompts",
)
parser.add_argument(
    "--hide_info",
    default=False,
    action="store_true",
    help="Hide text info elements from UI",
)
parser.add_argument(
    "-cam",
    "--use_webcam",
    default=False,
    action="store_true",
    help="Use a webcam as the video input, instead of a file",
)

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
use_square_sizing = not args.use_aspect_ratio
imgenc_base_size = args.base_size_px
max_memory_history = args.max_memories
max_pointer_history = args.max_pointers
max_prompt_history = args.max_prompts
discard_on_bad_objscore = not args.keep_bad_objscores
clear_history_on_new_prompts = not args.keep_history_on_new_prompts
show_info = not args.hide_info
use_webcam = args.use_webcam

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Create history to re-use selected inputs
history = HistoryKeeper()
_, history_vidpath = history.read("video_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
video_path = ask_for_path_if_missing(arg_image_path, "video", history_vidpath) if not use_webcam else 0
model_path = ask_for_model_path_if_missing(__file__, arg_model_path, history_modelpath)

# Store history for use on reload (but don't save video path when using webcam)
if use_webcam:
    history.store(model_path=model_path)
else:
    history.store(video_path=video_path, model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Set up shared image encoder settings (needs to be consistent across image/video frame encodings)
imgenc_config_dict = {"max_side_length": imgenc_base_size, "use_square_sizing": use_square_sizing}

# Get the model name, for reporting
model_name = osp.basename(model_path)

print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
assert isinstance(sammodel, SAMV2Model), "Only SAMv2 models are supported for video predictions!"
sammodel.to(**device_config_dict)

# Set up access to video
vreader = LoopingVideoReader(video_path).release()
sample_frame = vreader.get_sample_frame()

# Initial model run to make sure everything succeeds
print("", "Encoding image data...", sep="\n", flush=True)
t1 = perf_counter()
encoded_img, token_hw, preencode_img_hw = sammodel.encode_image(sample_frame, **imgenc_config_dict)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
time_taken_ms = round(1000 * (t2 - t1))
print(f"  -> Took {time_taken_ms} ms", flush=True)

# Run model without prompts as sanity check. Also gives initial result values
box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list = [], [], []
encoded_prompts = sammodel.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts, blank_promptless_output=False)
prediction_hw = mask_preds.shape[2:]
mask_uint8 = (mask_preds[:, 0, :, :] > 0.0 * 255).byte().cpu().numpy().squeeze()

# Provide some feedback about how the model is running
model_device = device_config_dict["device"]
model_dtype = str(device_config_dict["dtype"]).split(".")[-1]
image_hw_str = f"{preencode_img_hw[0]} x {preencode_img_hw[1]}"
token_hw_str = f"{token_hw[0]} x {token_hw[1]}"
print(
    "",
    f"Config ({model_name}):",
    f"  Device: {model_device} ({model_dtype})",
    f"  Resolution HW: {image_hw_str}",
    f"  Tokens HW: {token_hw_str}",
    sep="\n",
    flush=True,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up UI

# Playback control UI for adjusting video position
vreader.pause()
playback_slider = LoopingVideoPlaybackSlider(vreader, stay_paused_on_change=True)

# Set up shared UI elements & control logic
ui_elems = PromptUI(sample_frame, mask_preds)
uictrl = PromptUIControl(ui_elems)

# Set up text-based reporting UI
vram_text = ValueBlock("VRAM: ", "-", "MB", max_characters=5)
objscore_text = ValueBlock("Score: ", None, max_characters=3)
num_prompts_text = ValueBlock("Prompts: ", "0", max_characters=2)
num_history_text = ValueBlock("History: ", "0", max_characters=2)


# Set up button controls
show_preview_btn = ToggleButton("Preview")
track_btn = ToggleButton("Track", on_color=(30, 140, 30))
store_prompt_btn = ImmediateButton("Store Prompt", text_scale=0.35, color=(145, 160, 40))
clear_prompts_btn = ImmediateButton("Clear Prompts", text_scale=0.35, color=(80, 110, 230))
enable_history_btn = ToggleButton("Enable History", default_state=True, text_scale=0.35, on_color=(90, 85, 115))
clear_history_btn = ImmediateButton("Clear History", text_scale=0.35, color=(130, 60, 90))
force_same_button_width(store_prompt_btn, clear_prompts_btn, enable_history_btn, clear_history_btn)

# Set up info header
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(model_name, f"{token_hw_str} tokens", device_dtype_str)
footer_msgbar = StaticMessageBar(
    "[space] Play/Pause", "[tab] Track", "[p] Preview", text_scale=0.35, space_equally=True, bar_height=30
)

# Set up full display layout
show_info = True
disp_layout = VStack(
    header_msgbar if show_info else None,
    ui_elems.layout,
    playback_slider if not use_webcam else None,
    HStack(vram_text, objscore_text),
    HStack(num_prompts_text, track_btn, num_history_text),
    HStack(store_prompt_btn, clear_prompts_btn, enable_history_btn, clear_history_btn),
    footer_msgbar if show_info else None,
).set_debug_name("DisplayLayout")

# Render out an image with a target size, to figure out which side we should limit when rendering
display_image = disp_layout.render(h=display_size_px, w=display_size_px)
render_side = "h" if display_image.shape[1] > display_image.shape[0] else "w"
render_limit_dict = {render_side: display_size_px}
min_display_size_px = disp_layout._rdr.limits.min_h if render_side == "h" else disp_layout._rdr.limits.min_w


# ---------------------------------------------------------------------------------------------------------------------
# %% Video loop

# Setup display window
window = DisplayWindow("Display - q to quit", display_fps=1000 / vreader.get_frame_delay_ms())
window.attach_mouse_callbacks(disp_layout)
window.attach_keypress_callback(" ", vreader.pause)

# Change tools/masks on arrow keys
uictrl.attach_arrowkey_callbacks(window)
window.attach_keypress_callback("p", show_preview_btn.toggle)
window.attach_keypress_callback(KEY.TAB, track_btn.toggle)

# For clarity, some additional keypress codes
KEY_ZOOM_IN = ord("=")
KEY_ZOOM_OUT = ord("-")

# Set up various value tracking helpers
imgenc_idx_keeper = ValueChangeTracker(-1)
track_idx_keeper = ValueChangeTracker(-1)
pause_keeper = ValueChangeTracker(vreader.get_pause_state())
vram_report = PeriodicVRAMReport(update_period_ms=2000)

# Allocate storage for SAM2 video masking
objbuffer = SAM2VideoObjectResults.create(
    memory_history_length=max_memory_history,
    pointer_history_length=max_pointer_history,
    prompt_history_length=max_prompt_history,
)

# Helper used to define/keep track of all states of the UI
STATES = Enum(
    "state",
    [
        "ADJUST_PLAYBACK",
        "SWITCH_TRACK_ON",
        "SWITCH_TRACK_OFF",
        "SWITCH_PAUSE_ON",
        "SWITCH_PAUSE_OFF",
        "TRACKING",
        "PAUSED",
        "PLAYING_WITHOUT_TRACKING",
    ],
)

# Initialize values that need to be set before UI starts
curr_state = STATES.PAUSED if vreader.get_pause_state() else STATES.PLAYING_WITHOUT_TRACKING
mselect_idx = 1
mask_contours_norm = []
try:

    for is_paused, frame_idx, frame in vreader:

        # Read controls
        is_changed_pause_state = pause_keeper.is_changed(is_paused)
        is_changed_track_idx = track_idx_keeper.is_changed(frame_idx)
        _, is_trackhistory_enabled = enable_history_btn.read()
        is_trackstate_changed, is_tracking = track_btn.read()
        _, show_mask_preview = show_preview_btn.read()

        # Wipe out buffered data
        if clear_prompts_btn.read():
            objbuffer.prompts_buffer.clear()
            track_idx_keeper.clear()
        if clear_history_btn.read():
            objbuffer.prevframe_buffer.clear()
            track_idx_keeper.clear()

        # Update text feedback
        vram_usage_mb = vram_report.get_vram_usage()
        vram_text.set_value(vram_usage_mb)
        num_prompt_mems, num_prev_mems = objbuffer.get_num_memories()
        num_prompts_text.set_value(num_prompt_mems)
        num_history_text.set_value(num_prev_mems)

        # Ugly: Figure out current state
        if playback_slider.is_active():
            curr_state = STATES.ADJUST_PLAYBACK
        elif is_trackstate_changed and is_tracking:
            curr_state = STATES.SWITCH_TRACK_ON
        elif is_trackstate_changed and not is_tracking:
            curr_state = STATES.SWITCH_TRACK_OFF
        elif is_changed_pause_state and is_paused:
            curr_state = STATES.SWITCH_PAUSE_ON
        elif is_changed_pause_state and not is_paused:
            curr_state = STATES.SWITCH_PAUSE_OFF
        elif is_tracking:
            curr_state = STATES.TRACKING
        elif is_paused:
            curr_state = STATES.PAUSED
        else:
            curr_state = STATES.PLAYING_WITHOUT_TRACKING

        # Encode any 'new' frames as needed
        # -> unless the playback slider is changing, it would cripple the machine
        need_image_encode = imgenc_idx_keeper.is_changed(frame_idx)
        if need_image_encode and curr_state != STATES.ADJUST_PLAYBACK:
            encoded_img, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)
            imgenc_idx_keeper.record(frame_idx)

        # Handle pausing changes
        if is_changed_pause_state:

            # Consume user prompt inputs (if we don't do this, inputs queued up during playback can appear!)
            uictrl.read_prompts()
            ui_elems.clear_prompts()

            # Enable/disable prompt UI when playing/pausing
            ui_elems.enable_tools(is_paused)
            ui_elems.enable_masks(is_paused)
            pause_keeper.record(is_paused)

        # Wipe out masking/contours when jumping around playback (otherwise stays over top of changing video!)
        if curr_state == STATES.ADJUST_PLAYBACK:
            mask_contours_norm = []
            mask_preds = mask_preds * 0.0
            mask_uint8 = mask_uint8 * 0
            ui_elems.clear_prompts()
            track_btn.toggle(False)

        # Handle tracking changes
        if curr_state == STATES.SWITCH_TRACK_ON:

            # Unpause video
            vreader.pause(set_is_paused=not is_tracking)

            # If a prompt exists when tracking begins, assume we should use it
            if sammodel.check_have_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list):
                _, init_mem, init_ptr = sammodel.initialize_video_masking(
                    encoded_img,
                    box_tlbr_norm_list,
                    fg_xy_norm_list,
                    bg_xy_norm_list,
                    mask_index_select=mselect_idx,
                )
                objbuffer.store_prompt_result(frame_idx, init_mem, init_ptr)
                if clear_history_on_new_prompts:
                    objbuffer.prevframe_buffer.clear()

            # For QoL, if user was on FG point, switch back to hover
            _, _, selected_tool = ui_elems.tools_constraint.read()
            need_hover_switch = selected_tool == ui_elems.tools.fgpt
            if need_hover_switch:
                ui_elems.tools_constraint.change_to(ui_elems.tools.hover)
            pass

        elif curr_state == STATES.SWITCH_TRACK_OFF:

            # Wipe out segmentation data and any UI interactions that may have queued
            mask_contours_norm = []
            uictrl.read_prompts()
            ui_elems.clear_prompts()
            show_preview_btn.toggle(False)

            # Pause when disabling tracking (user can play/pause separately if desired)
            vreader.pause(True)

        elif curr_state == STATES.SWITCH_PAUSE_ON:
            track_btn.toggle(False)

        elif curr_state == STATES.SWITCH_PAUSE_OFF:
            mask_contours_norm = []
            mask_preds = mask_preds * 0.0
            mask_uint8 = mask_uint8 * 0
            if objbuffer.get_num_memories()[0] > 0:
                track_btn.toggle(True)

        # Handle main steady state (paused or tracking)
        if curr_state == STATES.PAUSED:

            # Store encoded prompts as needed
            if store_prompt_btn.read():
                _, init_mem, init_ptr = sammodel.initialize_video_masking(
                    encoded_img,
                    box_tlbr_norm_list,
                    fg_xy_norm_list,
                    bg_xy_norm_list,
                    mask_index_select=mselect_idx,
                )
                objbuffer.store_prompt_result(frame_idx, init_mem, init_ptr)
                ui_elems.clear_prompts()

            # Look for user interactions
            is_mask_changed, mselect_idx, selected_mask_btn = ui_elems.masks_constraint.read()
            need_prompt_encode, box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list = uictrl.read_prompts()
            have_user_prompts = sammodel.check_have_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
            have_track_prompts = objbuffer.get_num_memories()[0] > 0
            if need_prompt_encode and (have_user_prompts or not have_track_prompts):
                encoded_prompts = sammodel.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
                mask_preds, iou_preds = sammodel.generate_masks(
                    encoded_img,
                    encoded_prompts,
                    mask_hint=None,
                    blank_promptless_output=True,
                )
                track_idx_keeper.clear()

            # If there are no user prompts but there are tracking prompts, run the tracker to get a segmentation
            if have_track_prompts and not have_user_prompts and is_changed_track_idx:
                obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
                    encoded_img, **objbuffer.to_dict()
                )
                mselect_idx = int(best_mask_idx.squeeze().cpu())
                track_idx_keeper.record(frame_idx)

            pass

        elif curr_state == STATES.TRACKING:

            # Only run tracking if we're on a new index
            if is_changed_track_idx:
                track_idx_keeper.record(frame_idx)

                # Run model on each frame & store results
                obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
                    encoded_img, **objbuffer.to_dict()
                )
                mselect_idx = int(best_mask_idx.squeeze().cpu())

                # Only store history for high-scoring predictions
                obj_score = float(obj_score.squeeze().float().cpu().numpy())
                if obj_score < 0 and discard_on_bad_objscore:
                    mask_preds = mask_preds * 0.0
                elif is_trackhistory_enabled:
                    objbuffer.store_result(frame_idx, mem_enc, obj_ptr)

                objscore_text.set_value(round(obj_score, 1))

        # Update the mask indicators
        ui_elems.masks_constraint.change_to(mselect_idx)
        uictrl.update_mask_previews(mask_preds, mselect_idx)

        # Process contour data
        mask_uint8 = uictrl.create_hires_mask_uint8(mask_preds, mselect_idx, preencode_img_hw)
        _, mask_contours_norm = get_contours_from_mask(mask_uint8, normalize=True)

        # Update the main display image in the UI
        uictrl.update_main_display_image(frame, mask_uint8, mask_contours_norm, show_mask_preview)

        # Switch 'refresh rate' depending on state
        frame_delay_ms = None
        if is_paused:
            frame_delay_ms = 20
        elif is_tracking:
            frame_delay_ms = 1

        # Display final image
        display_image = disp_layout.render(**render_limit_dict)
        req_break, keypress = window.show(display_image, frame_delay_ms)
        if req_break:
            break

        # Updates playback indicator & allows for adjusting playback
        playback_slider.update(frame_idx)

        # Scale display size up when pressing +/- keys
        if keypress == KEY_ZOOM_IN:
            display_size_px = min(display_size_px + 50, 10000)
            render_limit_dict = {render_side: display_size_px}
        if keypress == KEY_ZOOM_OUT:
            display_size_px = max(display_size_px - 50, min_display_size_px)
            render_limit_dict = {render_side: display_size_px}

except KeyboardInterrupt:
    print("", "Closed with Ctrl+C", sep="\n")

except Exception as err:
    raise err

finally:
    cv2.destroyAllWindows()
    vreader.release()
