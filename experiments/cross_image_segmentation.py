#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

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

import os.path as osp
import argparse
from time import perf_counter
from collections import deque

import torch
import cv2

from muggled_sam.make_sam import make_sam_from_state_dict

from muggled_sam.demo_helpers.ui.video import ValueChangeTracker
from muggled_sam.demo_helpers.ui.window import DisplayWindow, KEY
from muggled_sam.demo_helpers.ui.layout import HStack, VStack, OverlayStack
from muggled_sam.demo_helpers.ui.images import ExpandingImage
from muggled_sam.demo_helpers.ui.text import ValueBlock
from muggled_sam.demo_helpers.ui.static import StaticMessageBar, HSeparator, VSeparator
from muggled_sam.demo_helpers.ui.buttons import ImmediateButton
from muggled_sam.demo_helpers.ui.sliders import HSlider
from muggled_sam.demo_helpers.shared_ui_layout import (
    build_mask_preview_buttons,
    build_tool_buttons,
    build_tool_overlays,
    read_prompts,
    make_hires_mask_uint8,
    update_mask_preview_buttons,
)

from muggled_sam.demo_helpers.video_frame_select_ui import run_video_frame_select_ui
from muggled_sam.demo_helpers.contours import MaskContourData, get_contours_from_mask
from muggled_sam.demo_helpers.mask_postprocessing import calculate_mask_stability_score
from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing, load_init_prompts
from muggled_sam.demo_helpers.misc import get_default_device_string, make_device_config, get_total_cuda_vram_usage_mb
from muggled_sam.demo_helpers.saving import save_image_segmentation, get_save_name, make_prompt_save_data


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path_1 = None
default_image_path_2 = None
default_model_path = None
default_prompts_path = None
default_display_size = 900
default_base_size = None

# Define script arguments
parser = argparse.ArgumentParser(description="Segment one image by prompting another, using SAMv2 'video' capability")
parser.add_argument("-ia", "--image_path_a", default=default_image_path_1, help="Path to 1st input image")
parser.add_argument("-ib", "--image_path_b", default=default_image_path_2, help="Path to 2nd input image")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Path to SAM model weights")
parser.add_argument(
    "-p",
    "--prompts_path",
    default=default_prompts_path,
    type=str,
    help="Path to a json file containing initial prompts to use on start-up (see saved json results for formatting)",
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
    "--hstack",
    default=False,
    action="store_true",
    help="Force images to stack horizontally",
)
parser.add_argument(
    "--vstack",
    default=False,
    action="store_true",
    help="Force images to stack vertically",
)

# For convenience
args = parser.parse_args()
arg_image_path_a = args.image_path_a
arg_image_path_b = args.image_path_b
arg_model_path = args.model_path
init_prompts_path = args.prompts_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
imgenc_base_size = args.base_size_px
show_info = not args.hide_info
force_hstack = args.hstack
force_vstack = args.vstack

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Create history to re-use selected inputs
root_path = osp.dirname(osp.dirname(__file__))
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_crossimgpath = history.read("cross_image_path")
_, history_vidpath = history.read("video_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path_a = ask_for_path_if_missing(arg_image_path_a, "image", history_imgpath)
image_path_b = ask_for_path_if_missing(arg_image_path_b, "cross image", history_crossimgpath)
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path_a, cross_image_path=image_path_b, model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Try loading model weights
model_name = osp.basename(model_path)
print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
assert sammodel.name in ("samv2", "samv3"), "Only SAMv2/v3 models are supported for cross-segmentation!"
sammodel.to(**device_config_dict)

# Load image and get shaping info for providing display
full_image_a = cv2.imread(image_path_a)
full_image_b = cv2.imread(image_path_b)

# Handle case where videos are provided instead of images
if full_image_a is None:
    ok_video, full_image_a = run_video_frame_select_ui(image_path_a, window_title="Select frame for image A")
    if not ok_video:
        print("", "Unable to load first image!", f"  @ {image_path_a}", sep="\n", flush=True)
        raise FileNotFoundError(osp.basename(image_path_a))
if full_image_b is None:
    ok_video, full_image_b = run_video_frame_select_ui(image_path_b, window_title="Select frame for image B")
    if not ok_video:
        print("", "Unable to load first image!", f"  @ {image_path_b}", sep="\n", flush=True)
        raise FileNotFoundError(osp.basename(image_path_b))

# Determine stacking direction
use_hstack_images = None
if force_hstack:
    use_hstack_images = True
elif force_vstack:
    use_hstack_images = False
else:
    img_a_h, img_a_w = full_image_a.shape[0:2]
    img_b_h, img_b_w = full_image_b.shape[0:2]
    have_narrow_img = (img_a_h > img_a_w) or (img_b_h > img_b_w)
    use_hstack_images = have_narrow_img


# ---------------------------------------------------------------------------------------------------------------------
# %% Run image encoder

# Set up shared image encoder settings
# -> Hard-coded square sizing so that images with different aspect ratios can be cross-prompted
imgenc_config_dict = {"max_side_length": imgenc_base_size, "use_square_sizing": True}

# Run Model
print("", "Encoding images...", sep="\n", flush=True)
t1 = perf_counter()
encoded_img_a, token_hw, preencode_img_hw = sammodel.encode_image(full_image_a, **imgenc_config_dict)
encoded_img_b, _, _ = sammodel.encode_image(full_image_b, **imgenc_config_dict)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
time_taken_ms = round(1000 * (t2 - t1))
print(f"  -> Took {time_taken_ms} ms", flush=True)

# Run model without prompts as sanity check. Also gives initial result values
encoded_prompts = sammodel.encode_prompts([], [], [])
mask_a_preds, iou_a_preds = sammodel.generate_masks(encoded_img_a, encoded_prompts)
mask_b_preds, iou_b_preds = sammodel.generate_masks(encoded_img_b, encoded_prompts)

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
    print("  VRAM usage:", total_vram_mb, "MB")


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up the UI

# Build images with overlays
img_a_elem, img_b_elem = ExpandingImage(full_image_a), ExpandingImage(full_image_b)
overlays_a, overlays_b = build_tool_overlays(), build_tool_overlays()
overlay_img_a = OverlayStack(img_a_elem, *overlays_a.totuple())
overlay_img_b = OverlayStack(img_b_elem, *overlays_b.totuple())
overlays_a.clear_all(flag_is_changed=True)

# Create mask images
mask_a_btns, masks_a_constraint = build_mask_preview_buttons(mask_a_preds)
mask_b_btns, masks_b_constraint = build_mask_preview_buttons(mask_a_preds)

# Set up (shared) tool button UI
tools_group, tools_constraint = build_tool_buttons(text_scale=0.35)
tools_bar = HStack(*tools_group.totuple())
tools_group.hover.add_on_change_listeners(overlays_a.hover.enable, overlays_b.hover.enable)
tools_group.box.add_on_change_listeners(overlays_a.box.enable, overlays_b.box.enable)
tools_group.fgpt.add_on_change_listeners(overlays_a.fgpt.enable, overlays_b.fgpt.enable)
tools_group.bgpt.add_on_change_listeners(overlays_a.bgpt.enable, overlays_b.bgpt.enable)

# Set up slider used to encode multiple frame 'memories'
video_iter_slider = HSlider("Repeat frame encoding", 0, 0, 12, step_size=1, marker_steps=1)
threshold_slider = HSlider("Mask threshold", 0.0, -10.0, 10.0, step_size=0.01, marker_steps=100)

# Set up text-based reporting UI
has_cuda = torch.cuda.is_available()
stability_score_txt_a = ValueBlock("Stability A: ")
objscore_text = ValueBlock("Object Score: ", None, max_characters=3)
stability_score_txt_b = ValueBlock("Stability B: ")
save_btn = ImmediateButton("Save", (60, 170, 20), text_scale=0.35)
text_scores_bar = HStack(stability_score_txt_a, objscore_text, stability_score_txt_b, save_btn)

# Decide on images/masks layout
preview_a_btns = VStack(*mask_a_btns)
preview_b_btns = VStack(*mask_b_btns)
img_layout = HStack(overlay_img_a, preview_a_btns, HSeparator(8), overlay_img_b, preview_b_btns)
if not use_hstack_images:
    vsep_a, vsep_b = VSeparator.many(2, 8)
    img_layout = HStack(VStack(overlay_img_a, vsep_a, overlay_img_b), VStack(preview_a_btns, vsep_b, preview_b_btns))

# Set up message bars to communicate data info & controls
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(
    model_name, f"{token_hw_str} tokens", device_dtype_str, text_scale=0.35, space_equally=True
)
footer_msg_bar = StaticMessageBar("[w/s] Change mask A", "[e/d] Change mask B", text_scale=0.35, space_equally=True)
disp_layout = VStack(
    header_msgbar if show_info else None,
    tools_bar,
    img_layout,
    text_scores_bar,
    video_iter_slider,
    threshold_slider,
    footer_msg_bar if show_info else None,
)
render_side = "h"
render_limit_dict = {render_side: display_size_px}
min_display_size_px = 250


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Main loop ***

# Load initial prompts, if provided
have_init_prompts, init_prompts_dict = load_init_prompts(init_prompts_path)
if have_init_prompts:
    overlays_a.box.add_boxes(*init_prompts_dict.get("boxes", []))
    overlays_a.fgpt.add_points(*init_prompts_dict.get("fg_points", []))
    overlays_a.bgpt.add_points(*init_prompts_dict.get("bg_points", []))

# Set up display
cv2.destroyAllWindows()
window = DisplayWindow("Display - q to quit", display_fps=60)
window.attach_mouse_callbacks(disp_layout)
window.move(200, 50)

# Change tools/masks on arrow keys
window.attach_keypress_callback(KEY.LEFT_ARROW, tools_constraint.previous)
window.attach_keypress_callback(KEY.RIGHT_ARROW, tools_constraint.next)
window.attach_keypress_callback("c", tools_group.clear.click)
window.attach_keypress_callback("y", masks_a_constraint.previous)
window.attach_keypress_callback("h", masks_a_constraint.next)
window.attach_keypress_callback("u", masks_b_constraint.previous)
window.attach_keypress_callback("j", masks_b_constraint.next)
window.attach_keypress_callback("s", save_btn.click)

# For clarity, some additional keypress codes
KEY_ZOOM_IN = ord("=")
KEY_ZOOM_OUT = ord("-")

# Initialize values needed on first run
side_select = "a"
last_side_keeper = ValueChangeTracker(side_select)
hover_a_keeper = ValueChangeTracker(False)
hover_b_keeper = ValueChangeTracker(False)
encoded_img, cross_encoded_img = encoded_img_a, encoded_img_b
ref_preds, cross_preds = mask_a_preds, mask_b_preds
ref_ptrs = torch.zeros((1, mask_a_preds.shape[1], encoded_img_a[0].shape[1])).to(mask_a_preds)
stability_offset = 2

# Some feedback
print(
    "",
    "Use prompts on one image to segment the other!",
    "- Shift-click to add multiple points",
    "- Right-click to remove points",
    "- Press 'c' key to clear prompts",
    "- Press -/+ keys to change display sizing",
    "- Use y/h and u/j to adjust mask selections",
    "- Press 's' key to save cross-segmented result",
    "- Press q or esc to close the window",
    "",
    sep="\n",
    flush=True,
)

try:
    while True:

        # Read controls
        is_numiter_changed, num_video_iter = video_iter_slider.read()
        is_thresh_changed, mask_threshold = threshold_slider.read()

        # Read mask selections
        is_mask_a_changed, mask_a_idx, _ = masks_a_constraint.read()
        is_mask_b_changed, mask_b_idx, _ = masks_b_constraint.read()
        is_mask_select_changed = is_mask_a_changed or is_mask_b_changed

        # Clear all prompts (for both images) when clear button is pressed
        need_prompt_clear = tools_group.clear.read()
        if need_prompt_clear:
            overlays_a.clear_all(flag_is_changed=False)
            overlays_b.clear_all(flag_is_changed=False)

        # Read prompts
        prompt_a_changed, prompts_a = read_prompts(overlays_a, tools_group, tools_constraint, "a")
        prompt_b_changed, prompts_b = read_prompts(overlays_b, tools_group, tools_constraint, "b")
        if prompt_a_changed and img_a_elem.is_hovered():
            side_select = "a"
        elif prompt_b_changed and img_b_elem.is_hovered():
            side_select = "b"

        # If we switch which image is prompted, clear prompts from the other image
        side_select_changed = last_side_keeper.is_changed(side_select)
        if side_select_changed:
            if side_select == "a":
                overlays_b.clear_all(flag_is_changed=False)
            elif side_select == "b":
                overlays_a.clear_all(flag_is_changed=False)
            last_side_keeper.record(side_select)

            # Update which images are used for reference/cross-prompting
            encoded_img = encoded_img_a if side_select == "a" else encoded_img_b
            cross_encoded_img = encoded_img_b if side_select == "a" else encoded_img_a

        # Only run the model when an input affecting the output has changed
        need_prompt_encode = prompt_a_changed or prompt_b_changed or need_prompt_clear
        if need_prompt_encode:
            selected_prompts = prompts_a if side_select == "a" else prompts_b
            encoded_prompts = sammodel.encode_prompts(*selected_prompts)
            with torch.inference_mode():
                grid_posenc = sammodel.coordinate_encoder.get_grid_position_encoding(token_hw)
                ref_preds, _, ref_ptrs, obj_score = sammodel.mask_decoder(encoded_img, encoded_prompts, grid_posenc)

        # Encode memory data from selected mask
        need_memory_encode = need_prompt_encode or is_mask_select_changed or is_numiter_changed
        if need_memory_encode:
            mem_mask_idx = mask_a_idx if side_select == "a" else mask_b_idx
            mask_for_mem = ref_preds[:, [mem_mask_idx], :, :]
            with torch.inference_mode():
                ref_mem = sammodel.memory_encoder(encoded_img[0], mask_for_mem, obj_score, is_prompt_encoding=True)
            ref_ptr = ref_ptrs[:, [mem_mask_idx]]

            # Run 'video' segmentation to prompt other image
            prompt_mem, prompt_ptr = tuple([ref_mem]), tuple([ref_ptr])
            prev_mems, prev_ptrs = deque([]), deque([])
            for _ in range(1 + num_video_iter):
                obj_score, best_mask_idx, cross_preds, new_mem, new_ptr = sammodel.step_video_masking(
                    cross_encoded_img, prompt_mem, prompt_ptr, prev_mems, prev_ptrs
                )
                prev_mems.appendleft(new_mem)
                prev_ptrs.appendleft(new_ptr)

            # Update score, which indicates 'goodness of match' between prompted & cross image
            objscore_text.set_value(round(obj_score.item(), 1))

        # Update segmentation display outlines
        need_outlines_update = need_memory_encode or is_thresh_changed
        if need_outlines_update:

            # Figure out which prediction goes with which image
            if side_select == "a":
                mask_a_preds, mask_b_preds = ref_preds, cross_preds
            elif side_select == "b":
                mask_a_preds, mask_b_preds = cross_preds, ref_preds
            else:
                raise ValueError(f"Unrecognized side selection: {side_select}")

            # Update displayed predictions
            update_mask_preview_buttons(mask_a_preds, mask_a_btns, mask_threshold)
            update_mask_preview_buttons(mask_b_preds, mask_b_btns, mask_threshold)
            mask_a, mask_b = mask_a_preds[0, mask_a_idx], mask_b_preds[0, mask_b_idx]

            # Get contours from masks & draw them
            hires_a = make_hires_mask_uint8(mask_a, preencode_img_hw, mask_threshold)
            hires_b = make_hires_mask_uint8(mask_b, preencode_img_hw, mask_threshold)
            _, cont_a = get_contours_from_mask(hires_a, normalize=True)
            _, cont_b = get_contours_from_mask(hires_b, normalize=True)
            overlays_a.polygon.set_polygons(cont_a)
            overlays_b.polygon.set_polygons(cont_b)

            # Calculate stability scores
            stability_score_a = calculate_mask_stability_score(mask_a, stability_offset, mask_threshold).item()
            stability_score_b = calculate_mask_stability_score(mask_b, stability_offset, mask_threshold).item()
            stability_score_txt_a.set_value(round(stability_score_a, 1))
            stability_score_txt_b.set_value(round(stability_score_b, 1))

        # Display final image
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

            # Get data for saving -> want to save image that isn't prompted
            # (though we'll save the prompts from the other image, just because...)
            is_side_a_prompted = sammodel.check_have_prompts(*prompts_a)
            is_side_b_prompted = sammodel.check_have_prompts(*prompts_b)
            if not (is_side_a_prompted or is_side_b_prompted):
                print("", "No prompts! Will skip saving...", sep="\n", flush=True)
                continue

            # Get data for saving
            selected_prompts = prompts_a if is_side_b_prompted else prompts_b
            prompt_image_save_path = image_path_a if is_side_a_prompted else image_path_b
            image_save_path = image_path_b if is_side_a_prompted else image_path_a
            mask_select_idx = mask_b_idx if is_side_a_prompted else mask_a_idx
            img_save_bgr = full_image_b if is_side_a_prompted else full_image_a
            mask_save_preds = cross_preds[0, mask_select_idx]

            # Get additional (shared) data for saving
            disp_image = disp_layout.rerender()
            all_prompts_dict = make_prompt_save_data(*selected_prompts)

            # Make result matching input image sizing
            img_save_hw = img_save_bgr.shape[0:2]
            mask_save_uint8 = make_hires_mask_uint8(mask_save_preds, img_save_hw, mask_threshold)
            contour_save_data = MaskContourData(mask_save_uint8)

            # Generate & save segmentation images!
            parent_folder_name = "cross_segmentation"
            save_folder, save_idx = get_save_name(image_save_path, parent_folder_name, base_save_folder=root_path)
            save_image_segmentation(
                save_folder,
                save_idx,
                img_save_bgr,
                disp_image,
                mask_save_uint8,
                contour_save_data,
                all_prompts_dict,
                yx_crop_slices=None,
            )

            # Save simple text file containing path to the image used for prompting
            ref_save_name = osp.join(save_folder, f"{save_idx}_prompt_image_path.txt")
            with open(ref_save_name, "w") as outfile:
                outfile.write(prompt_image_save_path)

            # Feedback on completion
            print(f"SAVED ({save_idx}):", save_folder)

except KeyboardInterrupt:
    print("", "Closed with Ctrl+C", sep="\n")

finally:
    cv2.destroyAllWindows()
