#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse
import os.path as osp
from time import perf_counter
from enum import Enum
from dataclasses import dataclass

import torch
import cv2
import numpy as np

from lib.make_sam import make_sam_from_state_dict
from lib.v2_sam.sam_v2_model import SAMV2Model

from lib.demo_helpers.ui.window import DisplayWindow, KEY
from lib.demo_helpers.ui.video import ReversibleLoopingVideoReader, LoopingVideoPlaybackSlider, ValueChangeTracker
from lib.demo_helpers.ui.layout import HStack, VStack
from lib.demo_helpers.ui.buttons import ToggleButton, ImmediateButton, RadioConstraint
from lib.demo_helpers.ui.static import StaticMessageBar
from lib.demo_helpers.ui.text import ValueBlock, TextBlock
from lib.demo_helpers.ui.base import force_same_min_width
from lib.demo_helpers.ui.overlays import DrawPolygonsOverlay

from lib.demo_helpers.shared_ui_layout import PromptUIControl, PromptUI
from lib.demo_helpers.crop_ui import run_crop_ui

from lib.demo_helpers.misc import PeriodicVRAMReport, make_device_config, get_default_device_string
from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.contours import get_contours_from_mask
from lib.demo_helpers.video_data_storage import SAM2VideoObjectResults
from lib.demo_helpers.saving import save_video_frames, get_save_name


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
default_num_object_buffers = 4

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
    help=f"Override base model size (default: {default_base_size})",
)
parser.add_argument(
    "-n",
    "--num_buffers",
    default=default_num_object_buffers,
    type=int,
    help=f"Number of object buffers in the saving/recording UI (default {default_num_object_buffers})",
)
parser.add_argument(
    "--max_memories",
    default=default_max_memory_history,
    type=int,
    help=f"Maximum number of previous-frame memory encodings to store (default: {default_max_memory_history})",
)
parser.add_argument(
    "--max_pointers",
    default=default_max_pointer_history,
    type=int,
    help=f"Maximum number of previous-frame object pointers to store (default: {default_max_pointer_history})",
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
parser.add_argument(
    "-nosave",
    "--disable_save",
    default=False,
    action="store_true",
    help="If set, this simplifies the UI by hiding the element associated with saving",
)
parser.add_argument(
    "--crop",
    default=False,
    action="store_true",
    help="If set, a cropping UI will appear on start-up to allow for the image to be cropped prior to processing",
)

# For convenience
args = parser.parse_args()
enable_saving = not args.disable_save
arg_image_path = args.image_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
use_square_sizing = not args.use_aspect_ratio
imgenc_base_size = args.base_size_px
num_obj_buffers = args.num_buffers if enable_saving else 1
max_memory_history = args.max_memories
max_pointer_history = args.max_pointers
discard_on_bad_objscore = not args.keep_bad_objscores
clear_history_on_new_prompts = not args.keep_history_on_new_prompts
show_info = not args.hide_info
use_webcam = args.use_webcam
enable_crop_ui = args.crop

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
vreader = ReversibleLoopingVideoReader(video_path).release()
sample_frame = vreader.get_sample_frame()
if enable_crop_ui:
    print("", "Cropping enabled: Adjust box to select image area for further processing", sep="\n", flush=True)
    _, history_crop_tlbr = history.read("crop_tlbr_norm")
    yx_crop_slice, crop_tlbr_norm = run_crop_ui(sample_frame, display_size_px, history_crop_tlbr)
    sample_frame = sample_frame[yx_crop_slice]
    history.store(crop_tlbr_norm=crop_tlbr_norm)

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
init_mask_preds, _ = sammodel.generate_masks(encoded_img, encoded_prompts, blank_promptless_output=False)
prediction_hw = init_mask_preds.shape[2:]
# mask_uint8 = ((mask_preds[:, 0, :, :] > 0.0) * 255).byte().cpu().numpy().squeeze()

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
# %% Helper Data types


@dataclass
class MaskResults:
    """Storage for (per-object) displayable masking results"""

    preds: torch.Tensor
    idx: int = 0
    objscore: float = 0.0

    @classmethod
    def create(cls, mask_predictions, mask_index=1, object_score=0.0):
        """Helper used to create an empty instance of mask results"""
        empty_predictions = torch.full_like(mask_predictions, -7)
        return cls(empty_predictions, mask_index, object_score)

    def clear(self):
        self.preds = torch.zeros_like(self.preds)
        self.objscore = 0.0
        return self

    def update(self, mask_predictions, mask_index, object_score=None):
        if mask_predictions is not None:
            self.preds = mask_predictions
        if mask_index is not None:
            self.idx = mask_index
        if object_score is not None:
            self.objscore = object_score
        return self


@dataclass
class SaveBufferData:
    """Storage for (per-object) encoded png save data"""

    png_per_frame_dict: dict[int, np.ndarray]
    bytes_per_frame_dict: dict[int, int]
    total_bytes: int = 0

    @classmethod
    def create(cls):
        """Helper used to create an empty instance of save buffer data"""
        return cls({}, {}, 0)

    def clear(self):
        self.png_per_frame_dict = {}
        self.bytes_per_frame_dict = {}
        self.total_bytes = 0
        return self


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up UI

# Playback control UI for adjusting video position
playback_slider = LoopingVideoPlaybackSlider(vreader, stay_paused_on_change=True)

# Set up shared UI elements & control logic
ui_elems = PromptUI(sample_frame, init_mask_preds, 2)
uictrl = PromptUIControl(ui_elems)

# Add extra polygon drawer for unselected objects
unselected_olay = DrawPolygonsOverlay((50, 5, 130), (25, 0, 60))
ui_elems.overlay_img.add_overlays(unselected_olay)

# Set up text-based reporting UI
vram_text = ValueBlock("VRAM: ", "-", "MB", max_characters=5)
objscore_text = ValueBlock("Score: ", None, max_characters=3)
num_prompts_text = ValueBlock("Prompts: ", "0", max_characters=2)
num_history_text = ValueBlock("History: ", "0", max_characters=2)
force_same_min_width(vram_text, objscore_text)

# Set up button controls
show_preview_btn = ToggleButton("Preview", default_state=False)
invert_mask_btn = ToggleButton("Invert", default_state=False)
track_btn = ToggleButton("Track", on_color=(30, 140, 30))
reversal_btn = ToggleButton("Reverse", default_state=False, text_scale=0.35)
store_prompt_btn = ImmediateButton("Store Prompt", text_scale=0.35, color=(145, 160, 40))
clear_prompts_btn = ImmediateButton("Clear Prompts", text_scale=0.35, color=(80, 110, 230))
enable_history_btn = ToggleButton("Enable History", default_state=True, text_scale=0.35, on_color=(90, 85, 115))
clear_history_btn = ImmediateButton("Clear History", text_scale=0.35, color=(130, 60, 90))
force_same_min_width(store_prompt_btn, clear_prompts_btn, enable_history_btn, clear_history_btn)


# Create save UI
enable_record_btn = ToggleButton("Enable Recording", default_state=False, on_color=(0, 15, 255), button_height=60)
buffer_btns_list = []
buffer_text_list = []
buffer_elems = []
for objidx in range(num_obj_buffers):
    buffer_btn = ToggleButton(f"Buffer {1+objidx}", button_height=20, text_scale=0.5, on_color=(145, 120, 65))
    buffer_txt = TextBlock(0.0, block_height=25, text_scale=0.35, max_characters=3)
    buffer_elems.extend([HStack(buffer_btn, buffer_txt)])
    buffer_btns_list.append(buffer_btn)
    buffer_text_list.append(buffer_txt)
force_same_min_width(*buffer_btns_list)
buffer_btn_constraint = RadioConstraint(*buffer_btns_list)
buffer_title_text = TextBlock("Buffered Mask Data (MB)", block_height=20, text_scale=0.35)
buffer_save_btn = ImmediateButton("Save Buffer", button_height=30, text_scale=0.5, color=(110, 145, 65))
buffer_clear_btn = ImmediateButton("Clear Buffer", button_height=30, text_scale=0.5, color=(80, 60, 190))
save_sidebar = VStack(enable_record_btn, buffer_title_text, *buffer_elems, buffer_save_btn, buffer_clear_btn)

# Set up info bars
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(model_name, f"{token_hw_str} tokens", device_dtype_str, space_equally=True)
footer_msgbar = StaticMessageBar(
    "[tab] Store Prompt",
    "[v/b] Buffers" if enable_saving else "[i] Invert",
    "[space] Play/Pause",
    "[p] Preview",
    text_scale=0.35,
    space_equally=True,
    bar_height=30,
)

# Set up full display layout
disp_layout = VStack(
    header_msgbar if show_info else None,
    HStack(ui_elems.layout, save_sidebar) if enable_saving else ui_elems.layout,
    playback_slider if not use_webcam else None,
    HStack(vram_text, objscore_text),
    HStack(num_prompts_text, track_btn, num_history_text),
    HStack(store_prompt_btn, clear_prompts_btn, reversal_btn, enable_history_btn, clear_history_btn),
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
window = DisplayWindow("Display - q to quit", display_fps=60)
window.attach_mouse_callbacks(disp_layout)

# Change tools/masks on arrow keys
uictrl.attach_arrowkey_callbacks(window)
window.attach_keypress_callback(" ", vreader.toggle_pause)
window.attach_keypress_callback("p", show_preview_btn.toggle)
window.attach_keypress_callback("i", invert_mask_btn.toggle)
window.attach_keypress_callback("b", buffer_btn_constraint.next)
window.attach_keypress_callback("v", buffer_btn_constraint.previous)
window.attach_keypress_callback(KEY.TAB, store_prompt_btn.click)
window.attach_keypress_callback("r", reversal_btn.toggle)

# For clarity, some additional keypress codes
KEY_ZOOM_IN = ord("=")
KEY_ZOOM_OUT = ord("-")

# Set up various value tracking helpers
imgenc_idx_keeper = ValueChangeTracker(-1)
track_idx_keeper = ValueChangeTracker(-1)
pause_keeper = ValueChangeTracker(vreader.get_pause_state())
vram_report = PeriodicVRAMReport(update_period_ms=2000)

# Helper used to define/keep track of all states of the UI
STATES = Enum(
    "state", ["ADJUST_PLAYBACK", "SWITCH_PAUSE_ON", "SWITCH_PAUSE_OFF", "TRACKING", "PAUSED", "NO_TRANSISTION"]
)

# Set up per-object storage for masking/saving results
objiter = list(range(num_obj_buffers))
maskresults_list = [MaskResults.create(init_mask_preds) for _ in objiter]
savebuffers_list = [SaveBufferData.create() for _ in objiter]
memory_list = [
    SAM2VideoObjectResults.create(max_memory_history, max_pointer_history, prompt_history_length=32) for _ in objiter
]

vreader.pause()
curr_state = STATES.PAUSED
tran_state = STATES.NO_TRANSISTION
try:

    for is_paused, frame_idx, frame in vreader:

        # Crop incoming frames if needed
        if enable_crop_ui:
            full_frame = frame.copy()
            frame = frame[yx_crop_slice]

        # Change playback direction, if needed
        is_reversed_changed, reverse_video = reversal_btn.read()
        if is_reversed_changed:
            vreader.toggle_reverse_state(reverse_video)

        # Read controls
        is_changed_pause_state = pause_keeper.is_changed(is_paused)
        is_changed_track_idx = track_idx_keeper.is_changed(frame_idx)
        _, is_trackhistory_enabled = enable_history_btn.read()
        _, show_mask_preview = show_preview_btn.read()
        _, is_inverted_mask = invert_mask_btn.read()

        is_changed_buffer, buffer_select_idx, _ = buffer_btn_constraint.read()
        if is_changed_buffer:
            ui_elems.clear_prompts()

        # Allow the track button to play/pause the video
        is_trackstate_changed, is_track_on = track_btn.read()
        if is_trackstate_changed:
            vreader.pause(not is_track_on)
            pass

        # Wipe out buffered data
        if clear_prompts_btn.read():
            memory_list[buffer_select_idx].prompts_buffer.clear()
            maskresults_list[buffer_select_idx].clear()
            track_idx_keeper.clear()
        if clear_history_btn.read():
            memory_list[buffer_select_idx].prevframe_buffer.clear()
            track_idx_keeper.clear()

        # Update text feedback
        vram_usage_mb = vram_report.get_vram_usage()
        vram_text.set_value(vram_usage_mb)
        num_prompt_mems, num_prev_mems = memory_list[buffer_select_idx].get_num_memories()
        num_prompts_text.set_value(num_prompt_mems)
        num_history_text.set_value(num_prev_mems)

        # Ugly: Figure out current states
        is_playback_adjusting = playback_slider.is_adjusting()
        if is_playback_adjusting:
            curr_state = STATES.ADJUST_PLAYBACK
        elif is_paused:
            curr_state = STATES.PAUSED
        else:
            curr_state = STATES.TRACKING

        # Handle transition states (mostly need to account for playback slider!)
        if is_playback_adjusting:
            tran_state = STATES.ADJUST_PLAYBACK
        elif is_changed_pause_state and is_paused:
            tran_state = STATES.SWITCH_PAUSE_ON
        elif is_changed_pause_state and not is_paused:
            tran_state = STATES.SWITCH_PAUSE_OFF
        else:
            tran_state = STATES.NO_TRANSISTION

        # Encode any 'new' frames as needed (but not on playback slider changes, would cripple the machine)
        need_image_encode = imgenc_idx_keeper.is_changed(frame_idx)
        if need_image_encode and not is_playback_adjusting:
            encoded_img, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)
            imgenc_idx_keeper.record(frame_idx)

        # Wipe out masking/contours when jumping around playback (otherwise stays over top of changing video!)
        if is_playback_adjusting:
            for maskresult in maskresults_list:
                maskresult.clear()

            ui_elems.clear_prompts()
            vreader.pause()
            track_btn.toggle(False, flag_if_changed=False)

        # Universal updates whenever the pause state changes
        if is_changed_pause_state:

            # Consume user prompt inputs (if we don't do this, inputs queued up during playback can appear!)
            uictrl.read_prompts()
            ui_elems.clear_prompts()

            # Enable/disable prompt UI when playing/pausing
            ui_elems.enable_tools(is_paused)
            ui_elems.enable_masks(is_paused)
            pause_keeper.record(is_paused)

        # Handle transistion states
        if tran_state == STATES.SWITCH_PAUSE_ON:

            # Make sure track button is disabled to indicate pause state
            track_btn.toggle(False, flag_if_changed=False)

            # For QoL, if user was on FG point, switch back to hover (more intuitive to work with)
            _, _, selected_tool = ui_elems.tools_constraint.read()
            need_hover_switch = selected_tool == ui_elems.tools.fgpt
            if need_hover_switch:
                ui_elems.tools_constraint.change_to(ui_elems.tools.hover)

            # Wipe out segmentation data and any UI interactions that may have queued
            uictrl.read_prompts()
            ui_elems.clear_prompts()
            show_preview_btn.toggle(False)

        elif tran_state == STATES.SWITCH_PAUSE_OFF:

            # Make sure track button is enabled to indicate active playback/tracking
            track_btn.toggle(True, flag_if_changed=False)

            # If a prompt exists when tracking begins, assume we should use it
            if sammodel.check_have_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list):
                _, init_mem, init_ptr = sammodel.initialize_video_masking(
                    encoded_img,
                    box_tlbr_norm_list,
                    fg_xy_norm_list,
                    bg_xy_norm_list,
                    mask_index_select=maskresults_list[buffer_select_idx].idx,
                )
                memory_list[buffer_select_idx].store_prompt_result(frame_idx, init_mem, init_ptr)
                if clear_history_on_new_prompts:
                    memory_list[buffer_select_idx].prevframe_buffer.clear()

            # If there is no tracking data, clear any on-screen masking (i.e. from user interactions)
            no_prompt_data = all(mem.check_has_prompts() == 0 for mem in memory_list)
            if no_prompt_data:
                for maskresult in maskresults_list:
                    maskresult.clear()

            pass

        # Handle main steady states (paused or tracking)
        if curr_state == STATES.PAUSED:

            # Initialize storage for predictions(which may not occur
            paused_mask_preds = None
            paused_obj_score = None

            # Look for user interactions
            _, paused_mask_idx, _ = ui_elems.masks_constraint.read()
            need_prompt_encode, box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list = uictrl.read_prompts()
            have_user_prompts = sammodel.check_have_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
            have_track_prompts = any(mem.check_has_prompts() for mem in memory_list)
            if need_prompt_encode and (have_user_prompts or not have_track_prompts):
                encoded_prompts = sammodel.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
                paused_mask_preds, _ = sammodel.generate_masks(
                    encoded_img,
                    encoded_prompts,
                    mask_hint=None,
                    blank_promptless_output=True,
                )
                track_idx_keeper.clear()

            # If there are no user prompts but there are tracking prompts, run the tracker to get a segmentation
            if have_track_prompts and not have_user_prompts and is_changed_track_idx:
                selected_memory_dict = memory_list[buffer_select_idx].to_dict()
                paused_obj_score, _, paused_mask_preds, _, _ = sammodel.step_video_masking(
                    encoded_img, **selected_memory_dict
                )
                paused_obj_score = float(paused_obj_score.squeeze().float().cpu().numpy())
                track_idx_keeper.record(frame_idx)

            # Store encoded prompts as needed
            if store_prompt_btn.read():
                _, init_mem, init_ptr = sammodel.initialize_video_masking(
                    encoded_img,
                    box_tlbr_norm_list,
                    fg_xy_norm_list,
                    bg_xy_norm_list,
                    mask_index_select=paused_mask_idx,
                )
                memory_list[buffer_select_idx].store_prompt_result(frame_idx, init_mem, init_ptr)
                ui_elems.clear_prompts()

            # Store user-interaction results for selected object while paused
            maskresults_list[buffer_select_idx].update(paused_mask_preds, paused_mask_idx, paused_obj_score)

        elif curr_state == STATES.TRACKING:

            # Only run tracking if we're on a new index
            if is_changed_track_idx:
                track_idx_keeper.record(frame_idx)

                for objidx in objiter:

                    # Don't run objects with no prompts
                    if not memory_list[objidx].check_has_prompts():
                        continue

                    # Only run model if we have stored prompts
                    obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
                        encoded_img, **memory_list[objidx].to_dict()
                    )
                    obj_score = float(obj_score.squeeze().float().cpu().numpy())
                    tracked_mask_idx = int(best_mask_idx.squeeze().cpu())

                    # Only store history for high-scoring predictions
                    if obj_score < 0 and discard_on_bad_objscore:
                        mask_preds = mask_preds * 0.0
                    elif is_trackhistory_enabled:
                        memory_list[objidx].store_result(frame_idx, mem_enc, obj_ptr)

                    # UGLY! Store results for each tracked object
                    maskresults_list[objidx].update(mask_preds, tracked_mask_idx, obj_score)

        # Update the mask indicators
        selected_mask_preds = maskresults_list[buffer_select_idx].preds
        selected_mask_idx = maskresults_list[buffer_select_idx].idx
        selected_obj_score = maskresults_list[buffer_select_idx].objscore
        ui_elems.masks_constraint.change_to(selected_mask_idx)
        uictrl.update_mask_previews(selected_mask_preds, selected_mask_idx, invert_mask=is_inverted_mask)

        # Update the (selected) object score
        objscore_text.set_value(round(selected_obj_score, 1))

        # Process contour data
        selected_mask_contours, selected_mask_uint8 = None, None
        unselected_contours = []
        for objidx, maskresult in enumerate(maskresults_list):
            mask_preds, mask_idx = maskresult.preds, maskresult.idx
            mask_uint8 = uictrl.create_hires_mask_uint8(mask_preds, mask_idx, preencode_img_hw)
            _, mask_contours_norm = get_contours_from_mask(mask_uint8, normalize=True)
            mask_contours_norm = tuple(mask_contours_norm)
            is_selected_idx = objidx == buffer_select_idx
            if is_selected_idx:
                selected_mask_contours = tuple(mask_contours_norm)
                selected_mask_uint8 = mask_uint8
            else:
                unselected_contours.extend(mask_contours_norm)

        # Update the main display image in the UI
        disp_mask_uint8 = cv2.bitwise_not(selected_mask_uint8) if is_inverted_mask else selected_mask_uint8
        uictrl.update_main_display_image(frame, disp_mask_uint8, selected_mask_contours, show_mask_preview)

        # Show unselected outlines separately to help distinguish them
        unselected_olay.set_polygons(unselected_contours if not show_mask_preview else None)

        # Display final image
        display_image = disp_layout.render(**render_limit_dict)
        req_break, keypress = window.show(display_image, None if is_paused else 1)
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

        # Handle recording of segmentation data
        _, is_record_enabled = enable_record_btn.read()
        if is_record_enabled and curr_state == STATES.TRACKING:

            # Sizing to use for saved mask results
            save_hw = frame.shape[0:2]

            for objidx in objiter:

                # Don't save anything for un-tracked objects
                if not memory_list[objidx].check_has_prompts():
                    continue

                # Generate a full sized mask matching the frame
                mask_preds, mask_idx = maskresults_list[objidx].preds, maskresults_list[objidx].idx
                save_mask_1ch_uint8 = uictrl.create_hires_mask_uint8(mask_preds, mask_idx, save_hw)
                if is_inverted_mask:
                    save_mask_1ch_uint8 = cv2.bitwise_not(save_mask_1ch_uint8)

                # Select whether we use the existing frame/mask or expand to the full (uncropped) sizing
                save_frame = frame
                if enable_crop_ui:
                    mask_bg = 255 if is_inverted_mask else 0
                    full_mask_1ch = np.full(full_frame.shape[0:2], mask_bg, dtype=np.uint8)
                    full_mask_1ch[yx_crop_slice] = save_mask_1ch_uint8
                    save_mask_1ch_uint8 = full_mask_1ch
                    save_frame = full_frame

                # Add mask to alpha channel (and clear masked RGB data, reduces file size!)
                save_frame = cv2.bitwise_and(save_frame, cv2.cvtColor(save_mask_1ch_uint8, cv2.COLOR_GRAY2BGR))
                save_frame = cv2.cvtColor(save_frame, cv2.COLOR_BGR2BGRA)
                save_frame[:, :, -1] = save_mask_1ch_uint8

                # Encode frame data in memory (want to save in bulk, to avoid killing filesystem)
                ok_encode, png_encoding = cv2.imencode(".png", save_frame)
                if ok_encode:
                    png_bytes = len(png_encoding)
                    existing_bytes = savebuffers_list[objidx].bytes_per_frame_dict.get(frame_idx, 0)
                    savebuffers_list[objidx].bytes_per_frame_dict[frame_idx] = png_bytes
                    savebuffers_list[objidx].total_bytes += png_bytes - existing_bytes
                    savebuffers_list[objidx].png_per_frame_dict[frame_idx] = png_encoding

                # Update buffer text
                save_buffer_mb = round(savebuffers_list[objidx].total_bytes / 1_000_000, 1)
                buffer_text_list[objidx].set_text(save_buffer_mb)

        # Save data to disk and clear storage
        if buffer_save_btn.read():

            # Only save if we actually have frame data!
            png_per_frame_dict = savebuffers_list[buffer_select_idx].png_per_frame_dict
            num_frames = len(png_per_frame_dict.keys())
            if num_frames > 0:
                save_folder, save_idx = get_save_name(video_path, "video")
                save_file_path = save_video_frames(save_folder, save_idx, buffer_select_idx, png_per_frame_dict)
                print("", f"Saving frame data ({num_frames} frames)...", f"@ {save_file_path}", sep="\n")
                buffer_clear_btn.click()

        # Wipe out save data if needed
        if buffer_clear_btn.read():
            savebuffers_list[buffer_select_idx].clear()
            buffer_text_list[buffer_select_idx].set_text(0.0)

except KeyboardInterrupt:
    print("", "Closed with Ctrl+C", sep="\n")

except Exception as err:
    raise err

finally:
    # Clean up resources
    cv2.destroyAllWindows()
    vreader.release()

    # Save any buffered frame data
    for objidx, savebuffer in enumerate(savebuffers_list):
        png_per_frame_dict = savebuffer.png_per_frame_dict
        num_frames = len(png_per_frame_dict.keys())
        if num_frames > 0:
            save_folder, save_idx = get_save_name(video_path, "video")
            save_file_path = save_video_frames(save_folder, save_idx, objidx, png_per_frame_dict)
            print("", f"Saving frame data ({num_frames} frames)...", f"@ {save_file_path}", sep="\n")

    pass
