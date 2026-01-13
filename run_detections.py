#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse
import os.path as osp
from time import perf_counter

import torch
import cv2
import numpy as np

from muggled_sam.make_sam import make_sam_from_state_dict

from muggled_sam.demo_helpers.ui.window import DisplayWindow, KEY
from muggled_sam.demo_helpers.ui.images import ExpandingImage
from muggled_sam.demo_helpers.ui.layout import HStack, VStack, OverlayStack
from muggled_sam.demo_helpers.ui.buttons import ToggleButton, ImmediateButton, RadioConstraint
from muggled_sam.demo_helpers.ui.sliders import HSlider
from muggled_sam.demo_helpers.ui.text import ValueBlock
from muggled_sam.demo_helpers.ui.static import StaticMessageBar, HSeparator
from muggled_sam.demo_helpers.ui.overlays import DrawBoxOverlay, HoverOverlay, PointSelectOverlay, BoxSelectOverlay
from muggled_sam.demo_helpers.ui.base import force_flex_min_width, force_same_min_width
from muggled_sam.demo_helpers.ui.helpers.text import TextDrawer
from muggled_sam.demo_helpers.ui.helpers.images import get_image_hw_for_max_side_length

from muggled_sam.demo_helpers.crop_ui import run_crop_ui
from muggled_sam.demo_helpers.video_frame_select_ui import run_video_frame_select_ui
from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from muggled_sam.demo_helpers.misc import (
    get_default_device_string,
    make_device_config,
    get_total_cuda_vram_usage_mb,
)
from muggled_sam.demo_helpers.text_input import read_user_text_input


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_vocab_path = None
default_display_size = 640
default_base_size = None

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to run Segment-Anything (SAM) on a single image")
parser.add_argument("-i", "--image_path", default=default_image_path, type=str, help="Path to input image")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Path to SAM3 model weights")
parser.add_argument(
    "-r", "--ref_image_path", nargs="?", default=None, const="", type=str, help="Path to a (different) reference image"
)
parser.add_argument("--bpe_vocab_path", default=default_vocab_path, type=str, help="Path to a BPE vocab")
parser.add_argument(
    "--text_prompt",
    default="visual",
    type=str,
    help="Default text prompt (used on startup and on 'clearing'). Use empty string: '' to disable",
)
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
    help="Set image processing size (will use model default if not set)",
)
parser.add_argument(
    "--hide_info",
    default=False,
    action="store_true",
    help="Hide text info elements from UI",
)
parser.add_argument(
    "--realtime",
    action="store_true",
    help="If set, real-time prompt inputs will be enabled by default",
)
parser.add_argument(
    "--crop",
    default=False,
    action="store_true",
    help="If set, a cropping UI will appear on start-up to allow for the image to be cropped prior to processing",
)

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
arg_ref_image_path = args.ref_image_path
arg_bpe_path = args.bpe_vocab_path
default_text_prompt = args.text_prompt if len(args.text_prompt) > 0 else None
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
use_square_sizing = not args.use_aspect_ratio
imgenc_base_size = args.base_size_px
show_info = not args.hide_info
default_realtime = args.realtime
enable_crop_ui = args.crop

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Create history to re-use selected inputs
history = HistoryKeeper()
_, history_imgpath = history.read("image_path")
_, history_modelpath = history.read("model_path")
_, history_refimgpath = history.read("reference_image_path")

# Get pathing to resources, if not provided already
have_different_ref_image = arg_ref_image_path is not None
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath)
if have_different_ref_image:
    arg_ref_image_path = ask_for_path_if_missing(arg_ref_image_path, "reference image", history_refimgpath)
model_path = ask_for_model_path_if_missing(__file__, arg_model_path, history_modelpath)

# Store history for use on reload
if have_different_ref_image:
    history.store(image_path=image_path, model_path=model_path, reference_image_path=arg_ref_image_path)
else:
    history.store(image_path=image_path, model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Set up shared image encoder settings (needs to be consistent across image/video frame encodings)
imgenc_config_dict = {"max_side_length": imgenc_base_size, "use_square_sizing": use_square_sizing}

# Get the model name, for reporting
model_name = osp.basename(model_path)

print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
assert sammodel.name == "samv3", "Error! Only SAMv3 models support detection..."
sammodel.to(**device_config_dict)
detmodel = sammodel.make_detector_model(arg_bpe_path)

# Load image and get shaping info for providing display
loaded_image_bgr = cv2.imread(image_path)
if loaded_image_bgr is None:
    ok_video, loaded_image_bgr = run_video_frame_select_ui(image_path)
    if not ok_video:
        print("", "Unable to load image!", f"  @ {image_path}", sep="\n", flush=True)
        raise FileNotFoundError(osp.basename(image_path))

# Crop input image if needed
input_image_bgr = loaded_image_bgr
yx_crop_slice = None
if enable_crop_ui:
    print("", "Cropping enabled: Adjust box to select image area for further processing", sep="\n", flush=True)
    _, history_crop_tlbr = history.read("crop_tlbr_norm")
    yx_crop_slice, crop_tlbr_norm = run_crop_ui(loaded_image_bgr, display_size_px, history_crop_tlbr)
    input_image_bgr = loaded_image_bgr[yx_crop_slice]
    history.store(crop_tlbr_norm=crop_tlbr_norm)

# Load reference image, if needed
input_ref_image_bgr = cv2.imread(arg_ref_image_path) if have_different_ref_image else input_image_bgr.copy()
assert input_ref_image_bgr is not None, f"Error loading reference imag: ({arg_ref_image_path})"


# ---------------------------------------------------------------------------------------------------------------------
# %% Run image encoder

# Run image encoding (only need to do this once)
print("", "Encoding image data...", sep="\n", flush=True)
t1 = perf_counter()
encoded_imgs, token_hw, preencode_img_hw = detmodel.encode_detection_image(input_image_bgr, **imgenc_config_dict)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
time_taken_ms = round(1000 * (t2 - t1))
print(f"  -> Took {time_taken_ms} ms", flush=True)

# Optionally encode a separate reference image
ref_encoded_imgs = encoded_imgs
if have_different_ref_image:
    print("", "Using separate reference image", "Encoding reference image data...", sep="\n", flush=True)
    ref_encoded_imgs, _, _ = detmodel.encode_detection_image(input_ref_image_bgr, **imgenc_config_dict)
    print("  Done", flush=True)

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

# Provide memory usage feedback, if using cuda GPU
if model_device == "cuda":
    total_vram_mb = get_total_cuda_vram_usage_mb()
    print("  VRAM:", total_vram_mb, "MB")


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up the UI

# Colors for convenience
color_pos_olay, color_neg_olay, color_topn_olay = (0, 255, 0), (0, 0, 255), (0, 120, 255)
color_pos_btn, color_neg_btn = (50, 80, 45), (45, 50, 90)
color_txt_ind = (90, 10, 215)
color_msg_bar = (65, 55, 50)

# Set up main image displays
ref_img_elem = ExpandingImage(input_image_bgr)
out_img_elem = ExpandingImage(input_image_bgr)

# Set up display overlays
hover_olay = HoverOverlay(read_right_clicks=True)
pos_box_tool_olay = BoxSelectOverlay(color_pos_olay, 2)
neg_box_tool_olay = BoxSelectOverlay(color_neg_olay, 2)
pos_point_tool_olay = PointSelectOverlay(color_pos_olay)
neg_point_tool_olay = PointSelectOverlay(color_neg_olay)
bounding_box_olay = DrawBoxOverlay(color_pos_olay)
ref_olay = OverlayStack(
    ref_img_elem, hover_olay, pos_box_tool_olay, pos_point_tool_olay, neg_box_tool_olay, neg_point_tool_olay
)
res_olay = OverlayStack(out_img_elem, bounding_box_olay)

# Set up tool bar components
tool_pospoint_toggle = ToggleButton("Point +", text_scale=0.5, on_color=color_pos_btn)
tool_posbox_toggle = ToggleButton("Box +", text_scale=0.5, default_state=True, on_color=color_pos_btn)
tool_negpoint_toggle = ToggleButton("Point -", text_scale=0.5, on_color=color_neg_btn)
tool_negbox_toggle = ToggleButton("Box -", text_scale=0.5, default_state=True, on_color=color_neg_btn)
tool_text_toggle = ToggleButton("Text", text_scale=0.5, on_color=color_txt_ind)
tool_clear_btn = ImmediateButton("Clear", text_scale=0.5)
radio_constraint = RadioConstraint(
    tool_pospoint_toggle, tool_negpoint_toggle, tool_posbox_toggle, tool_negbox_toggle, tool_text_toggle
)

# Set up text outputs
numpts_txtblock = ValueBlock("Points: ", "-")
numbox_txtblock = ValueBlock("Boxes: ", "-")
tokens_txtblock = ValueBlock("Tokens: ", "-")
txtpmt_txtblock = ValueBlock("Text: ", default_text_prompt)
numdet_txtblock = ValueBlock("Detections: ", 0)
force_same_min_width(numpts_txtblock, numbox_txtblock, txtpmt_txtblock, numdet_txtblock)
score_range_txtblock = ValueBlock("Scores: ", "-")

# Set up controls under the images
allow_realtime_toggle = ToggleButton("Real-time", default_state=default_realtime, text_scale=0.5, on_color=(90, 40, 70))
use_boxes_toggle = ToggleButton("Use Boxes", default_state=True, text_scale=0.5)
use_points_toggle = ToggleButton("Use Points", default_state=True, text_scale=0.5)
use_text_toggle = ToggleButton("Use Text", default_state=True, text_scale=0.5)
include_coords_toggle = ToggleButton("Include Coords", default_state=True, text_scale=0.35, on_color=(90, 90, 60))
thresh_slider = HSlider("Detection Threshold", 0.5, 0, 1, 0.01, marker_steps=10)
topn_slider = HSlider("Top-N (missing only)", 5, 0, 25, 1, marker_steps=5)
mask_opacity_slider = HSlider("Mask Opacity", 0.5, 0, 1, 0.05, marker_steps=5)
force_same_min_width(topn_slider, score_range_txtblock, mask_opacity_slider)

# Set up message bars
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(model_name, f"{token_hw_str} tokens", device_dtype_str, space_equally=True)
reg_msgs = ("Shift click to add more points/boxes", "Right click to delete")
txt_msgs = ("Use terminal to input a text prompt", "*TEXT MODE*", "Enter nothing to leave text mode")
footer_msgbar = StaticMessageBar(*reg_msgs, bar_height=30, text_scale=0.35, space_equally=True)

# Set up sizing const raints
tool_btns = (
    tool_pospoint_toggle,
    tool_negpoint_toggle,
    tool_posbox_toggle,
    tool_negbox_toggle,
    tool_text_toggle,
    tool_clear_btn,
)
force_flex_min_width(*tool_btns, flex=(1, 1, 1, 1, 1, 0.5))
force_same_min_width(allow_realtime_toggle, use_boxes_toggle, use_points_toggle, use_text_toggle, include_coords_toggle)

# Set up full display layout
disp_layout = VStack(
    header_msgbar if show_info else None,
    HStack(*tool_btns),
    HStack(numpts_txtblock, numbox_txtblock, txtpmt_txtblock, tokens_txtblock, numdet_txtblock),
    HStack(ref_olay, HSeparator(8), res_olay),
    HStack(allow_realtime_toggle, use_points_toggle, use_boxes_toggle, use_text_toggle, include_coords_toggle),
    thresh_slider,
    HStack(topn_slider, score_range_txtblock, mask_opacity_slider),
    footer_msgbar if show_info else None,
)

# Render out an image with a target size, to figure out which side we should limit when rendering
display_image = disp_layout.render(h=display_size_px, w=display_size_px)
render_side = "h" if display_image.shape[1] > display_image.shape[0] else "w"
render_limit_dict = {render_side: display_size_px}
min_display_size_px = disp_layout._rdr.limits.min_h if render_side == "h" else disp_layout._rdr.limits.min_w

# Set up hidden state (using UI element for convenience) to manage transistion in/out of text mode
hidden_text_state = ToggleButton("hidden")


# ---------------------------------------------------------------------------------------------------------------------
# %% Display loop

# Set up window
cv2.destroyAllWindows()
window = DisplayWindow("Display - q to quit", display_fps=60).attach_mouse_callbacks(disp_layout)
window.move(200, 50)

# Keypress for secondary controls
window.attach_keypress_callback("r", allow_realtime_toggle.toggle)
window.attach_keypress_callback("p", use_points_toggle.toggle)
window.attach_keypress_callback("b", use_boxes_toggle.toggle)
window.attach_keypress_callback("t", use_text_toggle.toggle)
window.attach_keypress_callback("i", include_coords_toggle.toggle)
window.attach_keypress_callback("c", tool_clear_btn.click)
window.attach_keypress_callback(KEY.LEFT_ARROW, radio_constraint.previous)
window.attach_keypress_callback(KEY.RIGHT_ARROW, radio_constraint.next)
window.attach_keypress_callback(KEY.SPACEBAR, lambda: radio_constraint.change_to(tool_text_toggle))
print(
    "",
    "Keypress controls:",
    "r: Toggle real-time inputs",
    "p, b, t: Toggle use points/boxes/text",
    "i: Toggle inclusion of coord. encoding in exemplars",
    "c: Clear all prompts (resets text to default input)",
    "spacebar: Enter text input mode",
    "Left/right arrows: Change selected tool",
    sep="\n",
)

# For clarity, some additional keypress codes
KEY_ZOOM_IN = ord("=")
KEY_ZOOM_OUT = ord("-")

# Set up data used in display loop
ref_hw = get_image_hw_for_max_side_length(input_ref_image_bgr, max_side_length=800)
out_hw = get_image_hw_for_max_side_length(input_image_bgr, max_side_length=800)
ref_src_img = cv2.resize(input_ref_image_bgr, dsize=(ref_hw[::-1]))
out_src_img = cv2.resize(input_image_bgr, dsize=(out_hw[::-1]))
out_wh = (out_hw[1], out_hw[0])
norm_to_px_scale = torch.tensor(out_wh, dtype=torch.float32) - 1.0

# Set up special image used when activating text mode (to help indicate switch to terminal)
ref_img_elem.set_image(ref_src_img)
out_img_elem.set_image((ref_src_img * 0.5).astype(np.uint8))
txt_drawer = TextDrawer(scale=0.75, thickness=2, color=color_txt_ind, bg_color=(0, 0, 0))
txt_mode_img = (ref_src_img * 0.25).astype(np.uint8)
_drawtxt_config = {"scale_step_size": 0.25, "margin_xy_px": (20, 20)}
txt_mode_img = txt_drawer.draw_to_box_norm(txt_mode_img, "Disabled", (0.1, 0.4), (0.9, 0.5), **_drawtxt_config)
txt_mode_img = txt_drawer.draw_to_box_norm(txt_mode_img, "(use terminal)", (0.1, 0.5), (0.9, 0.6), **_drawtxt_config)

# Set up initial prompts
radio_constraint.change_to(0)
text_prompt = default_text_prompt
point_xy_norm_list = []
box_xy1xy2_norm_list = []
negative_points_list = []
negative_boxes_list = []

# Some feedback
print(
    "",
    "Draw points or boxes on the left reference image,",
    "detections will be shown on the right image.",
    "- Shift-click to add multiple points/boxes",
    "- Right-click to remove points/boxes",
    "- Use -/+ keys to change display sizing",
    "- Press q or esc to close the window",
    "",
    sep="\n",
    flush=True,
)

try:

    # Trigger detection reset + update on startup
    tool_clear_btn.click()

    while True:

        # Reset detection state (so we don't run the model on every frame)
        need_detection_update = False

        # Read controls
        is_tool_select_changed, _, tool_ref = radio_constraint.read()
        _, allow_realtime = allow_realtime_toggle.read()
        is_use_points_changed, use_points = use_points_toggle.read()
        is_use_box_changed, use_boxes = use_boxes_toggle.read()
        is_use_text_changed, use_text = use_text_toggle.read()
        is_incl_coords_changed, include_coords = include_coords_toggle.read()
        is_detthresh_changed, det_thresh = thresh_slider.read()
        is_topn_changed, top_n = topn_slider.read()
        is_mask_opacity_changed, mask_opacity = mask_opacity_slider.read()
        is_mouse_moved, is_mouse_clicked, mouse_evt_xy = hover_olay.read()
        _, enable_terminal_text_input = hidden_text_state.read()

        # Wipe out prompts on clear
        if tool_clear_btn.read():
            text_prompt = default_text_prompt
            box_xy1xy2_norm_list = []
            point_xy_norm_list = []
            negative_boxes_list = []
            negative_points_list = []
            for olay in (pos_box_tool_olay, neg_box_tool_olay, pos_point_tool_olay, neg_point_tool_olay):
                olay.clear()
            need_detection_update = True

        # Handle enabling/disabling of inputs when switching tools
        if is_tool_select_changed:
            is_pos_point, is_neg_point = [tool_ref is elem for elem in (tool_pospoint_toggle, tool_negpoint_toggle)]
            is_pos_box, is_neg_box = [tool_ref is elem for elem in (tool_posbox_toggle, tool_negbox_toggle)]
            is_point_tool = is_pos_point or is_neg_point
            is_box_tool = is_pos_box or is_neg_box
            is_text_tool = tool_ref == tool_text_toggle
            pos_point_tool_olay.enable(is_pos_point)
            pos_box_tool_olay.enable(is_pos_box)
            neg_point_tool_olay.enable(is_neg_point)
            neg_box_tool_olay.enable(is_neg_box)

            # Make sure that the active tool is enabled
            if is_point_tool:
                use_points_toggle.toggle(True)
            if is_box_tool:
                use_boxes_toggle.toggle(True)
            if is_text_tool:
                use_text_toggle.toggle(True)

            # Trigger text mode if tool changes to text
            # (do this to get a display update in before we lock out UI due to terminal usage)
            ref_img_elem.set_image(ref_src_img)
            new_bar_msgs, new_bar_color = reg_msgs, color_msg_bar
            if is_text_tool:
                hidden_text_state.toggle(True)
                new_bar_msgs, new_bar_color = txt_msgs, color_txt_ind
                ref_img_elem.set_image(txt_mode_img)
                print(
                    "",
                    "*** Entering text mode ***",
                    "To exit, enter a blank input",
                    "Detection threshold can be changed by entering a float value",
                    f"Current text prompt: {text_prompt}",
                    sep="\n",
                    flush=True,
                )
            window.toggle_keypress_callbacks(not is_text_tool)
            footer_msgbar.update_message(*new_bar_msgs, bar_bg_color=new_bar_color)

        # Handle text input from terminal
        if enable_terminal_text_input:
            is_user_exit, is_user_float, user_txt_input = read_user_text_input()
            if is_user_exit:
                # Note: use of toggle state means we absorb 1 keypress before returning
                # to regular mode. This helps avoid registering buffered 'esc' inputs!
                hidden_text_state.toggle(False)
                radio_constraint.next()
                print(f"Exiting with prompt: {text_prompt}", "", sep="\n", flush=True)

            # Try to read floats as threshold settings
            elif is_user_float:
                user_txt_float = float(user_txt_input)
                user_txt_float = user_txt_float if user_txt_float < 1.01 else user_txt_float / 100
                thresh_slider.set(max(0, min(user_txt_float, 1.0)), False)
                is_detthresh_changed, det_thresh = thresh_slider.read()
                print(f"Update detection threshold to: {det_thresh}")

            else:
                # Record input as text prompt if we get here
                text_prompt = user_txt_input
                need_detection_update = True

            pass

        # Handle point inputs
        if is_point_tool:

            # Handle point updates
            is_new_pos_point, pos_point_xy_list = pos_point_tool_olay.read()
            is_new_neg_point, neg_point_xy_list = neg_point_tool_olay.read()
            is_point_changed = is_new_pos_point or is_new_neg_point
            if is_new_pos_point and is_mouse_clicked:
                point_xy_norm_list = pos_point_xy_list
                need_detection_update = True
            elif is_new_neg_point and is_mouse_clicked:
                negative_points_list = neg_point_xy_list
                need_detection_update = True
            elif allow_realtime and is_mouse_moved:
                # Allow for real-time point inputs (e.g. hover) if there are no existing inputs
                if is_pos_point and len(pos_point_xy_list) == 0:
                    point_xy_norm_list = [mouse_evt_xy.xy_norm] if mouse_evt_xy.is_in_region else []
                    need_detection_update = True
                elif is_neg_point and len(neg_point_xy_list) == 0:
                    negative_points_list = [mouse_evt_xy.xy_norm] if mouse_evt_xy.is_in_region else []
                    need_detection_update = True
                pass
            pass

        # Handle box updates
        if is_box_tool:
            is_new_pos_box, pos_box_xy1xy2_norm = pos_box_tool_olay.read()
            is_new_neg_box, neg_box_xy1xy2_norm = neg_box_tool_olay.read()
            if is_new_pos_box and (allow_realtime or not pos_box_tool_olay.check_is_in_progress()):
                box_xy1xy2_norm_list = pos_box_xy1xy2_norm
                need_detection_update = True
            if is_new_neg_box and (allow_realtime or not neg_box_tool_olay.check_is_in_progress()):
                negative_boxes_list = neg_box_xy1xy2_norm
                need_detection_update = True
            pass

        # Trigger detection update if any 'use X' toggle changes and there is data to use/not use
        if any((is_use_box_changed, is_use_points_changed, is_use_text_changed)):
            if is_use_text_changed and text_prompt is not None:
                need_detection_update = True
            if is_use_box_changed and len(box_xy1xy2_norm_list) > 0:
                need_detection_update = True
            if is_use_points_changed and len(point_xy_norm_list) > 0:
                need_detection_update = True

        # Trigger detection update if we have coord-based prompts and coord inclusion state changes
        if is_incl_coords_changed:
            coords_list = (box_xy1xy2_norm_list, point_xy_norm_list, negative_boxes_list, negative_points_list)
            if any(len(data) > 0 for data in coords_list):
                need_detection_update = True
            pass

        # Run model to update detections
        if need_detection_update:
            exemplars = detmodel.encode_exemplars(
                ref_encoded_imgs,
                text_prompt if use_text else None,
                box_xy1xy2_norm_list if use_boxes else None,
                point_xy_norm_list if use_points else None,
                negative_boxes_list if use_boxes else None,
                negative_points_list if use_points else None,
                include_coordinate_encodings=include_coords,
            )
            mask_preds, box_preds, det_scores, _ = detmodel.generate_detections(encoded_imgs, exemplars)

            # Update reporting
            num_pos_box = len(box_xy1xy2_norm_list) if use_boxes else 0
            num_neg_box = len(negative_boxes_list) if use_boxes else 0
            num_pos_pts = len(point_xy_norm_list) if use_points else 0
            num_neg_pts = len(negative_points_list) if use_points else 0
            tokens_txtblock.set_value(exemplars.shape[1])
            numpts_txtblock.set_value(f"+{num_pos_pts}, -{num_neg_pts}")
            numbox_txtblock.set_value(f"+{num_pos_box}, -{num_neg_box}")
            txtpmt_txtblock.set_value(text_prompt if use_text else None)

        # Update displayed mask/boxes
        if need_detection_update or is_detthresh_changed or is_topn_changed or is_mask_opacity_changed:

            # Use scores to filter results
            is_good_score = det_scores > det_thresh
            good_scores = det_scores[is_good_score]

            # Get top scores (used to display results when we don't get any detections)
            _, top_score_idxs = det_scores.float().cpu().sort(descending=True)
            top_n_idxs = top_score_idxs[0, 0:top_n].tolist()

            # Get 'good' results for display
            num_filtered = int(is_good_score.sum().item())
            disp_boxes = box_preds[is_good_score] if num_filtered > 0 else box_preds[0, top_n_idxs]
            disp_masks = mask_preds[is_good_score] if num_filtered > 0 else mask_preds[0, top_n_idxs]
            numdet_txtblock.set_value(num_filtered)

            # Update indicator to show score range
            if num_filtered > 0:
                score_range_txtblock.set_value(f"[{100*good_scores.min():.0f}, {100*good_scores.max():.0f}]")
            elif top_n > 0:
                top_n_scores = det_scores[0, top_n_idxs]
                score_range_txtblock.set_value(f"[{100*top_n_scores.min():.0f}, {100*top_n_scores.max():.0f}]")
            else:
                score_range_txtblock.set_value(f"[{100*det_scores.min():.0f}, {100*det_scores.max():.0f}]")

            #  Show bounding box predictions
            bounding_box_olay.style(color=color_pos_olay if num_filtered > 0 else color_topn_olay)
            bounding_box_olay.set_boxes(disp_boxes.float().cpu().numpy())

            # Form display mask
            weight_d, weight_m = mask_opacity, 1.0 - mask_opacity
            combined_mask, _ = (disp_masks > 0).max(0) if num_filtered > 0 else (mask_preds[0, 0] > 1e6, None)
            combined_mask_uint8 = (combined_mask.byte() * 255).cpu().numpy()
            scaled_mask = cv2.resize(combined_mask_uint8, out_wh, interpolation=cv2.INTER_NEAREST)
            out_img = cv2.addWeighted(out_src_img, weight_d, cv2.copyTo(out_src_img, scaled_mask), weight_m, 0.0)
            out_img_elem.set_image(out_img)

        # Render final output
        display_image = disp_layout.render(**render_limit_dict)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        # Scale display size up when pressing +/- keys
        if keypress == KEY_ZOOM_IN:
            display_size_px = min(display_size_px + 50, 10000)
            render_limit_dict = {render_side: display_size_px}
        if keypress == KEY_ZOOM_OUT:
            display_size_px = max(display_size_px - 50, min_display_size_px)
            render_limit_dict = {render_side: display_size_px}

        pass

except KeyboardInterrupt:
    pass

finally:
    cv2.destroyAllWindows()
