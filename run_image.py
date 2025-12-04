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
from muggled_sam.demo_helpers.ui.layout import HStack, VStack
from muggled_sam.demo_helpers.ui.buttons import ToggleButton, ImmediateButton
from muggled_sam.demo_helpers.ui.sliders import HSlider
from muggled_sam.demo_helpers.ui.static import StaticMessageBar
from muggled_sam.demo_helpers.ui.base import force_flex_min_width, force_same_min_width

from muggled_sam.demo_helpers.shared_ui_layout import PromptUIControl, PromptUI, ReusableBaseImage
from muggled_sam.demo_helpers.crop_ui import run_crop_ui
from muggled_sam.demo_helpers.video_frame_select_ui import run_video_frame_select_ui

from muggled_sam.demo_helpers.mask_postprocessing import MaskPostProcessor

from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing, load_init_prompts
from muggled_sam.demo_helpers.saving import save_image_segmentation, get_save_name, make_prompt_save_data
from muggled_sam.demo_helpers.misc import (
    get_default_device_string,
    make_device_config,
    get_total_cuda_vram_usage_mb,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_prompts_path = None
default_mask_hint_path = None
default_display_size = 900
default_base_size = None
default_simplify = 0.0

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to run Segment-Anything (SAM) on a single image")
parser.add_argument("-i", "--image_path", default=default_image_path, help="Path to input image")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Path to SAM model weights")
parser.add_argument(
    "-p",
    "--prompts_path",
    default=default_prompts_path,
    type=str,
    help="Path to a json file containing initial prompts to use on start-up (see saved json results for formatting)",
)
parser.add_argument(
    "--mask_path",
    default=default_mask_hint_path,
    type=str,
    help="Path to a mask image, which will be used as a prompt for segmentation",
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
    "-l",
    "--simplify",
    default=default_simplify,
    type=float,
    help="Set starting 'simplify' setting (value between 0 and 1)",
)
parser.add_argument(
    "-q",
    "--hide_iou",
    action="store_true",
    help="Hide mask quality estimates",
)
parser.add_argument(
    "--hide_info",
    default=False,
    action="store_true",
    help="Hide text info elements from UI",
)
parser.add_argument(
    "--enable_promptless_masks",
    default=False,
    action="store_true",
    help="If set, the model will generate mask predictions even when no prompts are given",
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
init_prompts_path = args.prompts_path
mask_hint_path = args.mask_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
use_square_sizing = not args.use_aspect_ratio
imgenc_base_size = args.base_size_px
init_simplify = args.simplify
show_iou_preds = not args.hide_iou
show_info = not args.hide_info
disable_promptless_masks = not args.enable_promptless_masks
enable_crop_ui = args.crop

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
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
sammodel.to(**device_config_dict)

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

# Try loading the given mask hint
mask_hint_img = None
if mask_hint_path is not None:
    assert osp.exists(mask_hint_path), f"Invalid mask hint path: {mask_hint_path}"
    mask_hint_img = cv2.imread(mask_hint_path)
    assert mask_hint_img is not None, f"Error loading mask hint image: {mask_hint_path}"
use_mask_hint = mask_hint_img is not None


# ---------------------------------------------------------------------------------------------------------------------
# %% Run image encoder

# Run Model
print("", "Encoding image data...", sep="\n", flush=True)
t1 = perf_counter()
encoded_img, token_hw, preencode_img_hw = sammodel.encode_image(input_image_bgr, imgenc_base_size, use_square_sizing)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
time_taken_ms = round(1000 * (t2 - t1))
print(f"  -> Took {time_taken_ms} ms", flush=True)

# Run model without prompts as sanity check. Also gives initial result values
box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list = [], [], []
encoded_prompts = sammodel.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
mask_preds, iou_preds = sammodel.generate_masks(
    encoded_img, encoded_prompts, blank_promptless_output=disable_promptless_masks
)

# Set up mask hint to match image encoding, if needed
mask_hint = None
if use_mask_hint:
    pred_h, pred_w = mask_preds.shape[2:]
    mask_hint_img_1ch = cv2.cvtColor(mask_hint_img, cv2.COLOR_BGR2GRAY)
    mask_hint_img_1ch = cv2.resize(mask_hint_img_1ch, dsize=(pred_w, pred_h))
    mask_hint = torch.from_numpy(mask_hint_img_1ch).squeeze().unsqueeze(0)
    mask_hint = ((mask_hint / max(mask_hint.max(), 1.0)) - 0.5) * 20.0
    mask_hint = mask_hint.to(mask_preds)
    mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts, mask_hint)

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

# Set up shared UI elements & control logic
ui_elems = PromptUI(input_image_bgr, mask_preds)
uictrl = PromptUIControl(ui_elems)

# Set up message bars to communicate data info & controls
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(model_name, f"{token_hw_str} tokens", device_dtype_str, space_equally=True)
footer_msgbar = StaticMessageBar(
    "[p] Preview",
    "[i] Invert",
    "[tab] Contouring",
    "[m] Mask hint" if use_mask_hint else "[arrows] Tools/Masks",
    text_scale=0.35,
)

# Set up secondary button controls
mask_hint_btn, show_preview_btn, invert_mask_btn, ext_mask_only_btn, pick_best_btn = ToggleButton.many(
    "Mask Hint", "Preview", "Invert", "External Only", "Pick best", default_state=False, text_scale=0.5
)
mask_hint_btn.toggle(use_mask_hint)
save_btn = ImmediateButton("Save", (60, 170, 20))
secondary_ctrls = HStack(
    mask_hint_btn if use_mask_hint else None,
    show_preview_btn,
    invert_mask_btn,
    ext_mask_only_btn,
    pick_best_btn,
    save_btn,
)

# Set up slider controls
thresh_slider = HSlider("Mask Threshold", 0, -8.0, 8.0, 0.1, marker_steps=10)
bridge_slider = HSlider("Bridge Gaps", 0, -50, 50, 1, marker_steps=5)
small_hole_slider = HSlider("Remove holes", 0, 0, 100, 1, marker_steps=20)
small_island_slider = HSlider("Remove islands", 0, 0, 100, 1, marker_steps=20)
padding_slider = HSlider("Pad contours", 0, -50, 50, 1, marker_steps=5)
simplify_slider = HSlider("Simplify contours", init_simplify, 0, 1, 0.01, marker_steps=25)
simplify_to_perimeter_btn = ToggleButton("By-Perimeter", text_scale=0.5, default_state=True)

# Set up sizing constraints
force_same_min_width(small_hole_slider, small_island_slider)
force_flex_min_width(simplify_slider, simplify_to_perimeter_btn, flex=(3, 1))

# Set up full display layout
disp_layout = VStack(
    header_msgbar if show_info else None,
    ui_elems.layout,
    secondary_ctrls,
    thresh_slider,
    bridge_slider,
    HStack(small_hole_slider, small_island_slider),
    padding_slider,
    HStack(simplify_slider, simplify_to_perimeter_btn),
    footer_msgbar if show_info else None,
).set_debug_name("DisplayLayout")

# Render out an image with a target size, to figure out which side we should limit when rendering
display_image = disp_layout.render(h=display_size_px, w=display_size_px)
render_side = "h" if display_image.shape[1] > display_image.shape[0] else "w"
render_limit_dict = {render_side: display_size_px}
min_display_size_px = disp_layout._rdr.limits.min_h if render_side == "h" else disp_layout._rdr.limits.min_w

# Load initial prompts, if provided
have_init_prompts, init_prompts_dict = load_init_prompts(init_prompts_path)
if have_init_prompts:
    uictrl.load_initial_prompts(init_prompts_dict)


# ---------------------------------------------------------------------------------------------------------------------
# %% Window setup

# Set up display
cv2.destroyAllWindows()
window = DisplayWindow("Display - q to quit", display_fps=60).attach_mouse_callbacks(disp_layout)
window.move(200, 50)

# Change tools/masks on arrow keys
uictrl.attach_arrowkey_callbacks(window)

# Keypress for secondary controls
window.attach_keypress_callback("p", show_preview_btn.toggle)
window.attach_keypress_callback(KEY.TAB, ext_mask_only_btn.toggle)
window.attach_keypress_callback("i", invert_mask_btn.toggle)
window.attach_keypress_callback("s", save_btn.click)
window.attach_keypress_callback("c", ui_elems.tools.clear.click)

# Add toggle for mask hinting if needed
if use_mask_hint:
    window.attach_keypress_callback("m", mask_hint_btn.toggle)

# For clarity, some additional keypress codes
KEY_ZOOM_IN = ord("=")
KEY_ZOOM_OUT = ord("-")

# Set up helper objects for managing display/mask data
base_img_maker = ReusableBaseImage(input_image_bgr)
mask_postprocessor = MaskPostProcessor()

# Some feedback
print(
    "",
    "Use prompts to segment the image!",
    "- Shift-click to add multiple points",
    "- Right-click to remove points",
    "- Press -/+ keys to change display sizing",
    "- Press q or esc to close the window",
    "",
    sep="\n",
    flush=True,
)

# *** Main display loop ***
try:
    while True:

        # Read prompt input data & selected mask
        is_prompt_changed, (box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list) = uictrl.read_prompts()
        is_mask_changed, mselect_idx, selected_mask_btn = ui_elems.masks_constraint.read()

        # Read secondary controls
        is_mhint_changed, enable_mask_hint = mask_hint_btn.read()
        _, show_mask_preview = show_preview_btn.read()
        is_invert_changed, use_inverted_mask = invert_mask_btn.read()
        _, use_external_contours = ext_mask_only_btn.read()
        _, use_best_mask = pick_best_btn.read()

        # Read sliders
        is_mthresh_changed, mthresh = thresh_slider.read()
        _, mholes = small_hole_slider.read()
        _, mislands = small_island_slider.read()
        _, mbridge = bridge_slider.read()
        _, mpadding = padding_slider.read()
        _, msimplify = simplify_slider.read()
        _, mperimeter = simplify_to_perimeter_btn.read()

        # Update post-processor based on control values
        scaled_msimplify = (msimplify**2) * 0.01
        mask_postprocessor.update(mholes, mislands, mbridge, mpadding, scaled_msimplify, mperimeter)

        # Only run the model when an input affecting the output has changed!
        need_prompt_encode = is_prompt_changed or is_mhint_changed
        if need_prompt_encode:
            encoded_prompts = sammodel.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
            mask_preds, iou_preds = sammodel.generate_masks(
                encoded_img,
                encoded_prompts,
                mask_hint if enable_mask_hint else None,
                blank_promptless_output=disable_promptless_masks,
            )
            if use_best_mask:
                best_mask_idx = sammodel.get_best_mask_index(iou_preds)
                ui_elems.masks_constraint.change_to(best_mask_idx)

        # Update mask previews & selected mask for outlines
        need_mask_update = any((need_prompt_encode, is_mthresh_changed, is_invert_changed, is_mask_changed))
        if need_mask_update:
            selected_mask_uint8 = uictrl.create_hires_mask_uint8(mask_preds, mselect_idx, preencode_img_hw, mthresh)
            uictrl.update_mask_previews(mask_preds, mthresh, use_inverted_mask)
            if show_iou_preds:
                uictrl.draw_iou_predictions(iou_preds)

        # Process contour data, if present
        final_mask_uint8, contour_data = mask_postprocessor(selected_mask_uint8, use_external_contours)
        if use_inverted_mask:
            final_mask_uint8 = np.bitwise_not(final_mask_uint8)

        # Re-generate display image at required display size
        # -> Not strictly needed, but can avoid constant re-sizing of base image (helpful for large images)
        contours_to_draw = contour_data.contour_norms_list
        display_hw = ui_elems.image.get_render_hw()
        disp_img = base_img_maker.regenerate(display_hw)
        uictrl.update_main_display_image(disp_img, final_mask_uint8, contours_to_draw, show_mask_preview)

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

        # Save data
        if save_btn.read():

            # Get additional data for saving
            disp_image = ui_elems.display_block.rerender()
            all_prompts_dict = make_prompt_save_data(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)

            # Make raw result matching input image sizing
            loaded_hw = loaded_image_bgr.shape[0:2]
            raw_mask_result_uint8 = uictrl.create_hires_mask_uint8(mask_preds, mselect_idx, loaded_hw, mthresh)

            # Generate & save segmentation images!
            save_folder, save_idx = get_save_name(image_path, "run_image")
            save_image_segmentation(
                save_folder,
                save_idx,
                loaded_image_bgr,
                disp_image,
                raw_mask_result_uint8,
                contour_data,
                all_prompts_dict,
                use_inverted_mask,
                yx_crop_slice,
            )
            print(f"SAVED ({save_idx}):", save_folder)

        pass

except KeyboardInterrupt:
    print("", "Closed with Ctrl+C", sep="\n")

except Exception as err:
    raise err

finally:
    cv2.destroyAllWindows()
