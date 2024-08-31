#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import lib  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to lib folder!")

import argparse
import os.path as osp
from time import perf_counter

import torch
import cv2

from lib.make_sam import make_sam_from_state_dict
from lib.v2_sam.sam_v2_model import SAMV2Model

from lib.demo_helpers.ui.window import DisplayWindow, KEY
from lib.demo_helpers.ui.base import force_same_min_width
from lib.demo_helpers.ui.layout import HStack, VStack
from lib.demo_helpers.ui.buttons import ToggleButton
from lib.demo_helpers.ui.sliders import HSlider
from lib.demo_helpers.ui.static import StaticMessageBar
from lib.demo_helpers.ui.text import ValueBlock
from lib.demo_helpers.ui.colormaps import HColormapsBar, make_spectral_colormap
from lib.demo_helpers.shared_ui_layout import PromptUIControl, PromptUI, ReusableBaseImage

from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing, load_init_prompts
from lib.demo_helpers.misc import get_default_device_string, make_device_config, normalize_to_npuint8


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_prompts_path = None
default_display_size = 900
default_base_size = 1024
default_max_window_size = 32

# Define script arguments
parser = argparse.ArgumentParser(description="Visualizes SAM mask data while allowing for altered window sizing")
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
    "-w",
    "--max_window_size",
    default=default_max_window_size,
    type=int,
    help="Change max allowable window sizing. Higher sizes require more VRAM (default {default_base_size})",
)
parser.add_argument(
    "-q",
    "--quality_estimate",
    default=False,
    action="store_true",
    help="Show mask quality estimates",
)
parser.add_argument(
    "--hide_info",
    default=False,
    action="store_true",
    help="Hide text info elements from UI",
)

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
init_prompts_path = args.prompts_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
use_square_sizing = not args.use_aspect_ratio
imgenc_base_size = args.base_size_px
max_window_size = args.max_window_size
show_iou_preds = args.quality_estimate
show_info = not args.hide_info

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Create history to re-use selected inputs
root_path = osp.dirname(osp.dirname(__file__))
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath)
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path, model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Get the model name, for reporting
model_name = osp.basename(model_path)

print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
sammodel.to(**device_config_dict)
is_v2_model = isinstance(sammodel, SAMV2Model)

# Load image and get shaping info for providing display
full_image_bgr = cv2.imread(image_path)
if full_image_bgr is None:
    vreader = cv2.VideoCapture(image_path)
    ok_read, full_image_bgr = vreader.read()
    vreader.release()
    if not ok_read:
        print("", "Unable to load image!", f"  @ {image_path}", sep="\n", flush=True)
        raise FileNotFoundError(osp.basename(image_path))


# ---------------------------------------------------------------------------------------------------------------------
# %% Initial model run

# Set up shared image encoder settings (needs to be consistent across image/video frame encodings)
imgenc_config_dict = {"max_side_length": imgenc_base_size, "use_square_sizing": use_square_sizing}

# Run Model
print("", "Encoding image data...", sep="\n", flush=True)
t1 = perf_counter()
encoded_img, token_hw, preencode_img_hw = sammodel.encode_image(full_image_bgr, **imgenc_config_dict)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
init_time_taken_ms = round(1000 * (t2 - t1), 1)
print(f"  -> Took {init_time_taken_ms} ms", flush=True)

# Run model without prompts as sanity check. Also gives initial result values
box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list = [], [], []
encoded_prompts = sammodel.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts, blank_promptless_output=False)
mask_uint8 = normalize_to_npuint8(mask_preds[0, 0, :, :])

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

# Figure out initial window sizing
init_winsize_per_stage = None
if is_v2_model:
    init_winsize_per_stage = model_config_dict.get("imgencoder_window_size_per_stage", (8, 4, 14, 7))
else:
    init_winsize_per_stage = [model_config_dict["base_window_size"]] * 4

# Figure out window sizing limits. Needed to stop crashing due to pytorch 'scaled_dot_product_attention'
# -> Crash seems to occur when providing tensors with size >2^16 (i.e. 65536 or higher)
# -> This can occur when using small window sizes & large input sizes
min_winsize_per_stage = (1, 1, 1, 1)
if is_v2_model:
    num_lowres_tokens = token_hw[0] * token_hw[1]
    tokens_per_stage = [16 * num_lowres_tokens, 4 * num_lowres_tokens, num_lowres_tokens, num_lowres_tokens // 2]
    min_winsize_per_stage = [num_tokens // (2**15) for num_tokens in tokens_per_stage]


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up the UI

# Set up shared UI elements & control logic
ui_elems = PromptUI(full_image_bgr, mask_preds)
uictrl = PromptUIControl(ui_elems)

# Set up window size sliders
win_sliders = []
for idx in range(4):
    slider = HSlider(
        f"Stage {1 + idx}",
        initial_value=init_winsize_per_stage[idx],
        min_value=max(1, min_winsize_per_stage[idx]),
        max_value=max_window_size,
        step_size=1,
        marker_steps=8,
    )
    win_sliders.append(slider)
slider_block = VStack(*win_sliders)

# Set up control buttons
show_preds_btn = ToggleButton("Show Prediction", on_color=(160, 180, 65), default_state=False, text_scale=0.5)
show_binary_btn = ToggleButton("Show Binary", on_color=(75, 45, 90), default_state=False, text_scale=0.5)
infer_time_block = ValueBlock("Inference: ", init_time_taken_ms, " ms", max_characters=5)
button_bar = HStack(infer_time_block, show_preds_btn, show_binary_btn)
force_same_min_width(infer_time_block, show_preds_btn, show_binary_btn)

# Set up message bars
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(model_name, f"{token_hw_str} tokens", device_dtype_str, space_equally=True)
footer_msgbar = StaticMessageBar(
    "[Right-click] Reset sliders", "[tab] Prediction view", "[b] Binary view", text_scale=0.35, space_equally=True
)

# Create bar of colormaps for adjusting display style
cmap_bar = HColormapsBar(make_spectral_colormap(), cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_INFERNO, None)

# Set up full display layout
disp_layout = VStack(
    header_msgbar if show_info else None,
    cmap_bar,
    ui_elems.layout,
    button_bar,
    slider_block,
    footer_msgbar if show_info else None,
)

# Load initial prompts, if provided
have_init_prompts, init_prompts_dict = load_init_prompts(init_prompts_path)
if have_init_prompts:
    uictrl.load_initial_prompts(init_prompts_dict)


# ---------------------------------------------------------------------------------------------------------------------
# %% Main display

# Set up display
cv2.destroyAllWindows()
window = DisplayWindow("Display - q to quit", display_fps=60).attach_mouse_callbacks(disp_layout)
window.move(200, 50)

# Attach key controls
uictrl.attach_arrowkey_callbacks(window)
window.attach_keypress_callback("c", ui_elems.tools.clear.click)
window.attach_keypress_callback(KEY.TAB, show_preds_btn.toggle)
window.attach_keypress_callback("b", show_binary_btn.toggle)

# For clarity, some additional keypress codes
KEY_ZOOM_OUT, KEY_ZOOM_IN = ord("-"), ord("=")

# Set up helper object for re-using an image scaled to match display sizing
base_img_maker = ReusableBaseImage(full_image_bgr)

# *** Main display loop ***
try:
    while True:

        # Read prompt input data & selected mask
        need_prompt_encode, box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list = uictrl.read_prompts()
        is_mask_changed, mselect_idx, selected_mask_btn = ui_elems.masks_constraint.read()

        # Read controls
        _, show_preds_as_display = show_preds_btn.read()
        _, show_preds_as_binary = show_binary_btn.read()
        is_winsizes_changed, win_sizes = zip(*[slider.read() for slider in win_sliders])

        # Re-encode image if window sizing changes
        winsize_changed = any(is_winsizes_changed)
        if winsize_changed:

            # Force display back to mask view if image encoding is changing
            show_preds_btn.toggle(True)

            # Update window sizing & re-run image segmentation to get new (raw) mask outputs for display
            sammodel.image_encoder.set_window_sizes(win_sizes)
            t1 = perf_counter()
            encoded_img, _, _ = sammodel.encode_image(full_image_bgr, **imgenc_config_dict)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = perf_counter()

            # Update inference time reporting
            infer_time_block.set_value(round(1000 * (t2 - t1), 1))

        # Update masking result if window or prompts are changed
        if winsize_changed or need_prompt_encode:
            encoded_prompts = sammodel.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
            mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts)

        # Update display of mask previews
        uictrl.update_mask_previews(mask_preds, mselect_idx, mask_threshold=0.0, invert_mask=False)
        if show_iou_preds:
            uictrl.draw_iou_predictions(iou_preds)

        # Re-generate display image at required display size
        # -> Not strictly needed, but can avoid constant re-sizing of base image (helpful for large images)
        display_hw = ui_elems.image.get_render_hw()
        disp_img = base_img_maker.regenerate(display_hw)
        if show_preds_as_display:

            # Scale selected mask to match display sizing
            mask_scaled = torch.nn.functional.interpolate(
                mask_preds[:, mselect_idx, :, :].unsqueeze(1),
                size=display_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(dim=(0, 1))

            # Convert mask to displayable image
            if show_preds_as_binary:
                disp_img = ((mask_scaled > 0.0).byte() * 255).cpu().numpy()
            else:
                mask_uint8 = normalize_to_npuint8(mask_scaled)
                disp_img = cmap_bar.apply_colormap(mask_uint8)

        # Render final output
        ui_elems.image.set_image(disp_img)
        display_image = disp_layout.render(h=display_size_px)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        # Scale display size up when pressing +/- keys
        if keypress == KEY_ZOOM_IN:
            display_size_px = min(display_size_px + 50, 10000)
        if keypress == KEY_ZOOM_OUT:
            display_size_px = max(display_size_px - 50, 250)

        pass

except KeyboardInterrupt:
    print("", "Closed with Ctrl+C", sep="\n")

except Exception as err:
    raise err

finally:
    cv2.destroyAllWindows()
