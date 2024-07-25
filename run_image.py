#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse
import os.path as osp
from time import perf_counter

import torch
import torch.nn as nn
import cv2
import numpy as np

from lib.make_sam import make_sam_from_original_state_dict

from lib.demo_helpers.ui.window import DisplayWindow, KEY
from lib.demo_helpers.ui.layout import HStack, VStack, OverlayStack
from lib.demo_helpers.ui.images import ExpandingImage
from lib.demo_helpers.ui.buttons import ToggleButton, ToggleImage, ImmediateButton, RadioConstraint
from lib.demo_helpers.ui.sliders import HSlider
from lib.demo_helpers.ui.static import StaticMessageBar
from lib.demo_helpers.ui.overlays import PointSelectOverlay, BoxSelectOverlay, HoverOverlay, DrawPolygonsOverlay
from lib.demo_helpers.ui.helpers.images import CheckerPattern

from lib.demo_helpers.contours import (
    get_contours_from_mask,
    get_largest_contour,
    get_contours_containing_xy,
    simplify_contour_px,
    normalize_contours,
    pixelize_contours,
)

from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.saving import save_segmentation_results
from lib.demo_helpers.misc import (
    get_default_device_string,
    make_device_config,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_display_size = 900
default_base_size = 1024
default_window_size = 16
default_show_iou_preds = False

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
    "-b",
    "--base_size_px",
    default=default_base_size,
    type=int,
    help="Override base model size (default {default_base_size})",
)
parser.add_argument(
    "-w",
    "--window_size",
    default=default_window_size,
    type=int,
    help="Change the window size of attention blocks within SAM's image encoder (default:{default_window_size})",
)
parser.add_argument(
    "-q",
    "--show_q_estimate",
    default=default_show_iou_preds,
    action="store_false" if default_show_iou_preds else "store_true",
    help="Show mask quality estimates" if default_show_iou_preds else "Hide mask quality estimates",
)

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
imgenc_base_size = args.base_size_px
imgenc_window_size = args.window_size
show_iou_preds = args.show_q_estimate

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Create history to re-use selected inputs
history = HistoryKeeper()
_, history_imgpath = history.read("image_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath)
model_path = ask_for_model_path_if_missing(__file__, arg_model_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path, model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Get the model name, for reporting
model_name = osp.basename(model_path)

print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
model_config_dict, sammodel = make_sam_from_original_state_dict(model_path)
sammodel.to(**device_config_dict)

# Load image and get shaping info for providing display
full_image_bgr = cv2.imread(image_path)
if full_image_bgr is None:
    print("", "Unable to load image!", f"  @ {image_path}", sep="\n", flush=True)
    raise FileNotFoundError(osp.basename(image_path))
image_ar = full_image_bgr.shape[1] / full_image_bgr.shape[0]
is_tall_img, is_very_tall_img, is_very_wide_img = image_ar < 0.7, image_ar < 0.4, image_ar > 2


# ---------------------------------------------------------------------------------------------------------------------
# %% Run image encoder

# Run Model
print("", "Encoding image data...", sep="\n", flush=True)
t1 = perf_counter()
encoded_img, preencode_img_hw = sammodel.encode_image(full_image_bgr, imgenc_base_size, imgenc_window_size)
torch.cuda.synchronize()
t2 = perf_counter()
time_taken_ms = round(1000 * (t2 - t1))
print(f"  -> Took {time_taken_ms} ms", flush=True)

# Run model without prompts as sanity check. Also gives initial result values
box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list = [], [], []
encoded_prompts = sammodel.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts)

# Provide some feedback about how the model is running
model_device = device_config_dict["device"]
model_dtype = str(device_config_dict["dtype"]).split(".")[-1]
image_hw_str = f"{preencode_img_hw[0]} x {preencode_img_hw[1]}"
token_hw_str = f"{encoded_img.shape[2]} x {encoded_img.shape[3]}"
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
    using_vram_mb = torch.cuda.memory_allocated() // 1_000_000
    peak_vram_mb = torch.cuda.max_memory_allocated() // 1_000_000
    print("  VRAM:", peak_vram_mb, "MB")


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up the UI

# Set up prompt UI interactions
hover_olay = HoverOverlay()
box_olay = BoxSelectOverlay(thickness=2)
fgpt_olay = PointSelectOverlay((0, 255, 0), point_radius=3)
bgpt_olay = PointSelectOverlay((0, 0, 0), bg_color=(0, 255, 0), point_radius=3).style(bg_thickness=2)
pgon_olay = DrawPolygonsOverlay((100, 10, 255), bg_color=(0, 0, 0))

# Set up tool selection UI
clear_all_prompts_btn = ImmediateButton("Clear", color=(0, 0, 150))
hover_btn, box_btn, fgpt_btn, bgpt_btn = ToggleButton.many("Hover", "Box", "FG Point", "BG Point")
toolselect_cb = HStack(hover_btn, box_btn, fgpt_btn, bgpt_btn, clear_all_prompts_btn)
selected_tool_constraint = RadioConstraint(hover_btn, box_btn, fgpt_btn, bgpt_btn)

# Tie tool overlays to toggle buttons
hover_btn.add_on_change_listeners(hover_olay.enable)
box_btn.add_on_change_listeners(box_olay.enable)
fgpt_btn.add_on_change_listeners(fgpt_olay.enable)
bgpt_btn.add_on_change_listeners(bgpt_olay.enable)

# Set up mask preview selection UI
blank_mask_btn_imgs = [np.zeros(mask_preds.shape[-2:], dtype=np.uint8)] * 4
mask_btns_list = ToggleImage.many(*blank_mask_btn_imgs, highlight_color=(0, 120, 255))
selected_mask_constraint = RadioConstraint(*mask_btns_list, initial_selected_index=1)
for mbtn in mask_btns_list:
    mbtn.set_text(text=None, scale=0.35, xy_norm=(1, 1), offset_xy_px=(-6, -6), bg_color=(0, 0, 0))

# Set up slider controls
thresh_slider = HSlider("Mask Threshold", 0, -8.0, 8.0, 0.1, marker_steps=10)
padding_slider = HSlider("Pad contours", 0, -50, 50, 1, marker_steps=5)
simplify_slider = HSlider("Simplify contours", 0, 0, 10, 0.25, marker_steps=4)

# Set up message bars to communicate data info & controls
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(model_name, f"{token_hw_str} tokens", device_dtype_str)
footer_msgbar = StaticMessageBar(
    "[arrows] Tools/Masks", "[i] Invert", "[tab] Contouring", "[spacebar] Preview", "[s] Save"
)

# Set up display images
colorimg_cb = ExpandingImage(full_image_bgr).set_debug_name("ColorImg")
imgoverlay_cb = OverlayStack(colorimg_cb, pgon_olay, hover_olay, box_olay, fgpt_olay, bgpt_olay)
imgoverlay_cb._rdr.pad.color = (35, 25, 30)

# Set up secondary button controls
show_preview_btn, invert_mask_btn, large_mask_only_btn, pick_best_btn = ToggleButton.many(
    "Preview", "Invert", "Largest Only", "Pick best", default_state=False
)
img_with_ctrls = VStack(imgoverlay_cb, HStack(show_preview_btn, invert_mask_btn, large_mask_only_btn, pick_best_btn))
large_mask_only_btn.toggle(True)

# Set up main display row, with image & mask previews, re-oriented based on tall vs. wide images
maskselect_cb = VStack(*mask_btns_list)
main_row = HStack(img_with_ctrls, maskselect_cb)
if is_very_wide_img:
    maskselect_cb = VStack(HStack(*mask_btns_list[:2]), HStack(*mask_btns_list[2:]))
    main_row = VStack(maskselect_cb, img_with_ctrls)
if is_tall_img:
    maskselect_cb = VStack(HStack(*mask_btns_list[:2]), HStack(*mask_btns_list[2:]))
    main_row = HStack(img_with_ctrls, maskselect_cb)
if is_very_tall_img:
    maskselect_cb = HStack(*mask_btns_list)
    main_row = HStack(img_with_ctrls, maskselect_cb)
maskselect_cb._rdr.pad.color = (60, 60, 60)
maskselect_cb.set_debug_name("MaskStack")
main_row.set_debug_name("DisplayRow")

# Set up full display layout
disp_layout = VStack(
    header_msgbar,
    toolselect_cb,
    main_row,
    thresh_slider,
    padding_slider,
    simplify_slider,
    footer_msgbar,
).set_debug_name("DisplayLayout")

# Render out an image with a target size, to figure out which side we should limit when rendering
display_image = disp_layout.render(h=display_size_px, w=display_size_px)
render_side = "h" if display_image.shape[1] > display_image.shape[0] else "w"
render_limit_dict = {render_side: display_size_px}


# ---------------------------------------------------------------------------------------------------------------------
# %% Window setup

# Set up display
cv2.destroyAllWindows()
window = DisplayWindow("Display", display_fps=60).attach_mouse_callbacks(disp_layout)
window.move(200, 50)

# Change tools/masks on arrow keys
window.attach_keypress_callback(KEY.LEFT_ARROW, selected_tool_constraint.previous)
window.attach_keypress_callback(KEY.RIGHT_ARROW, selected_tool_constraint.next)
window.attach_keypress_callback(KEY.UP_ARROW, selected_mask_constraint.previous)
window.attach_keypress_callback(KEY.DOWN_ARROW, selected_mask_constraint.next)

# Keypress for secondary controls
window.attach_keypress_callback(KEY.SPACEBAR, show_preview_btn.toggle)
window.attach_keypress_callback(KEY.TAB, large_mask_only_btn.toggle)
window.attach_keypress_callback("i", invert_mask_btn.toggle)

# For clarity, some additional keypress codes
KEY_ZOOM_IN = ord("=")
KEY_ZOOM_OUT = ord("-")
KEY_SAVE = ord("s")


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Main display loop ***

# Set up checker renderer for displaying masked images
checker_pattern = CheckerPattern()

# Initialize state values
box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list = [], [], []
base_disp_img = full_image_bgr.copy()
dispimg_h, dispimg_w = base_disp_img.shape[0:2]
prev_disp_h, prev_disp_w = base_disp_img.shape[0:2]

try:
    while True:

        # Re-generate base display image at requried scale
        # -> Not strictly needed, but avoids constant re-sizing of base image (helpful for large images)
        dispimg_h, dispimg_w = colorimg_cb.get_render_hw()
        if dispimg_h != prev_disp_h or dispimg_w != prev_disp_w:
            base_disp_img = cv2.resize(full_image_bgr, dsize=(dispimg_w, dispimg_h))
            prev_disp_h, prev_disp_w = base_disp_img.shape[0:2]

        # Read prompting controls
        is_tool_changed, tool_idx, selected_tool = selected_tool_constraint.read()
        is_mask_changed, mask_idx, selected_mask_btn = selected_mask_constraint.read()
        clear_prompts = clear_all_prompts_btn.read()
        if clear_prompts:
            box_olay.clear()
            fgpt_olay.clear()
            bgpt_olay.clear()

        # Read secondary controls
        is_preview_changed, show_mask_preview = show_preview_btn.read()
        is_invert_changed, use_inverted_mask = invert_mask_btn.read()
        is_large_changed, use_only_large_contour = large_mask_only_btn.read()
        is_best_changed, use_best_mask = pick_best_btn.read()
        is_secondary_ctrl_changed = any([is_preview_changed, is_invert_changed, is_large_changed, is_best_changed])

        # Read sliders
        is_mask_thresh_changed, mask_thresh = thresh_slider.read()
        is_simplify_changed, simplify_value = simplify_slider.read()
        is_padding_changed, contour_pad = padding_slider.read()
        is_slider_changed = any((is_mask_thresh_changed, is_simplify_changed, is_padding_changed))

        # Read prompt inputs
        box_prompt_changed, box_tlbr_norm_list = box_olay.read()
        fg_prompt_changed, fg_xy_norm_list = fgpt_olay.read()
        bg_prompt_changed, bg_xy_norm_list = bgpt_olay.read()

        # Only add hover point when the tool is active
        temp_hover_list = []
        hover_prompt_changed = False
        if selected_tool is hover_btn:
            hover_prompt_changed, hover_is_clicked, hover_xy_event = hover_olay.read()
            temp_hover_list = [hover_xy_event.xy_norm] if hover_xy_event.is_in_region else []

            # If the user clicks while hovering, treat it as a foreground point (and switch to FG tool)
            if hover_is_clicked:
                selected_tool_constraint.change_to(fgpt_btn)
                fgpt_olay.add_points(hover_xy_event.xy_norm)

        # Gather all fg points together for convenience
        all_fg_norm_list = [*fg_xy_norm_list, *temp_hover_list]

        # Toggle back to hover in case where fg point is removed and no other points remain
        if fg_prompt_changed:
            no_prompts = sum(len(pts) for pts in (all_fg_norm_list, box_tlbr_norm_list, bg_xy_norm_list)) == 0
            if no_prompts:
                selected_tool_constraint.change_to(hover_btn)

        # Only run the model when an input affecting the output has changed!
        prompt_value_changed = hover_prompt_changed or box_prompt_changed or fg_prompt_changed or bg_prompt_changed
        is_prompt_changed = is_tool_changed or prompt_value_changed or clear_prompts
        if is_prompt_changed:

            # Re-run SAM model to generate new segmentation masks
            encoded_prompts = sammodel.encode_prompts(box_tlbr_norm_list, all_fg_norm_list, bg_xy_norm_list)
            mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts, mask_hint=None)
            if use_best_mask:
                best_mask_idx = sammodel.get_best_mask_index(iou_preds)
                selected_mask_constraint.change_to(best_mask_idx)

        # Update mask selection images
        if is_prompt_changed or is_mask_thresh_changed or is_invert_changed:
            mask_preds_uint8 = ((mask_preds.squeeze(0) > mask_thresh) * 255).byte().cpu().numpy()
            for pred_idx, (mpred_uint8, mbtn) in enumerate(zip(mask_preds_uint8, mask_btns_list)):
                mbtn.set_image(mpred_uint8 if not use_inverted_mask else np.bitwise_not(mpred_uint8))
                if show_iou_preds:
                    quality_estimate = round(float(iou_preds[0, pred_idx].float().cpu()) * 100)
                    mbtn.set_text(str(quality_estimate))

        # Update selected base mask
        if is_prompt_changed or is_mask_changed or is_mask_thresh_changed:
            raw_mask_select = mask_preds[:, mask_idx, :, :].unsqueeze(1)
            raw_mask_upscale = nn.functional.interpolate(raw_mask_select.float(), size=preencode_img_hw)
            mask_uint8 = ((raw_mask_upscale[0, 0] > mask_thresh) * 255).byte().cpu().numpy()

        # Process contour data
        ok_contours, mask_contours_norm = get_contours_from_mask(mask_uint8, normalize=True)
        if not ok_contours:
            pgon_olay.clear()
            colorimg = base_disp_img.copy() if not show_mask_preview else checker_pattern.draw_like(base_disp_img)

        else:
            # For convenience
            mask_shape = mask_uint8.shape
            mask_h, mask_w = mask_shape[0:2]

            # Handle 'largest only' contour setting
            if use_only_large_contour:

                # Special case, if only 1 FG point is set, make sure we try to pick the largest
                # contour from the ones that contain the given FG point (otherwise result can look unintuitive)
                only_one_fg_pt = len(all_fg_norm_list) == 1 and len(box_tlbr_norm_list) == 0
                if only_one_fg_pt:
                    pt_xy = all_fg_norm_list[0]
                    have_filtered_result, filtered_contours_px = get_contours_containing_xy(mask_contours_norm, pt_xy)
                    if have_filtered_result:
                        mask_contours_norm = filtered_contours_px

                # Keep only the largest contour
                _, largest_contour_norm = get_largest_contour(mask_contours_norm, reference_shape=mask_shape)
                mask_contours_norm = [largest_contour_norm]

            # Convert contour to pixel units for further processing
            mask_contours_px = pixelize_contours(mask_contours_norm, mask_shape)

            # Simplify contour shape if needed
            if simplify_value > 0:
                mask_contours_px = [simplify_contour_px(contour, simplify_value) for contour in mask_contours_px]
                mask_contours_norm = normalize_contours(mask_contours_px, mask_shape)

            # Build base mask
            final_mask = np.zeros_like(mask_uint8)
            final_mask = cv2.fillPoly(final_mask, mask_contours_px, 255, cv2.LINE_AA)
            if use_inverted_mask:
                final_mask = np.bitwise_not(final_mask)

            # Add padding to mask, if needed
            need_padding = contour_pad != 0
            if need_padding:
                morph_type = cv2.MORPH_DILATE if contour_pad > 0 else cv2.MORPH_ERODE
                kernel_size = (abs(contour_pad), abs(contour_pad))
                morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
                final_mask = cv2.morphologyEx(final_mask, morph_type, morph_kernel)
                ok_pad_contour, padded_contour_norm = get_contours_from_mask(final_mask, normalize=True)
                if ok_pad_contour:
                    mask_contours_norm = padded_contour_norm

            # If showing the preview, mask out only the masked sections and layer on top of checker pattern
            colorimg = base_disp_img.copy()
            if show_mask_preview:
                colorimg = checker_pattern.superimpose(colorimg, final_mask)
                pgon_olay.clear()
            else:
                pgon_olay.set_polygons(mask_contours_norm)
            pass

        # Set the main image used for display
        colorimg_cb.set_image(colorimg)

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
            display_size_px = max(display_size_px - 50, 250)
            render_limit_dict = {render_side: display_size_px}

        # Save data
        if keypress == KEY_SAVE:
            save_folder, save_idx = save_segmentation_results(image_path, mask_contours_norm, mask_uint8)
            print(f"SAVED ({save_idx}):", save_folder)

        pass

except KeyboardInterrupt:
    print("", "Closed with Ctrl+C", sep="\n")

except Exception as err:
    raise err

finally:
    cv2.destroyAllWindows()
