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

import argparse
import os.path as osp
import cv2
import numpy as np

from muggled_sam.make_sam import make_sam_from_state_dict

from muggled_sam.v1_sam.components.image_encoder_attention import (
    GlobalAttentionBlock as V1GlobalBlock,
    WindowedAttentionBlock as V1WindowBlock,
)
from muggled_sam.v2_sam.components.hiera_blocks import (
    GlobalBlock as V2GlobalBlock,
    WindowedBlock as V2WindowBlock,
    PooledWindowedBlock as V2PoolBlock,
)
from muggled_sam.v3_sam.components.image_encoder_attention import (
    GlobalAttentionBlock as V3GlobalBlock,
    WindowedAttentionBlock as V3WindowBlock,
)

from muggled_sam.demo_helpers.ui.window import DisplayWindow, KEY
from muggled_sam.demo_helpers.ui.layout import VStack, HStack, GridStack
from muggled_sam.demo_helpers.ui.images import ExpandingImage
from muggled_sam.demo_helpers.ui.buttons import ToggleButton, ToggleImage, RadioConstraint
from muggled_sam.demo_helpers.ui.sliders import HSlider
from muggled_sam.demo_helpers.ui.colormaps import HColormapsBar
from muggled_sam.demo_helpers.ui.static import StaticMessageBar, HSeparator
from muggled_sam.demo_helpers.ui.text import ValueBlock
from muggled_sam.demo_helpers.ui.helpers.text import TextDrawer

from muggled_sam.demo_helpers.video_frame_select_ui import run_video_frame_select_ui
from muggled_sam.demo_helpers.model_capture import ModelOutputCapture
from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from muggled_sam.demo_helpers.misc import get_default_device_string, make_device_config, normalize_to_npuint8


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_display_size = 900
default_base_size = None

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to visualized SAM image encoder block outputs")
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
    help="Set image processing size (will use model default if not set)",
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
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
use_square_sizing = not args.use_aspect_ratio
imgenc_base_size = args.base_size_px
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

# Set up shared image encoder settings (needs to be consistent across image/video frame encodings)
imgenc_config_dict = {"max_side_length": imgenc_base_size, "use_square_sizing": use_square_sizing}

# Get the model name, for reporting
model_name = osp.basename(model_path)

print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
sammodel.to(**device_config_dict)

# Load image and get shaping info for providing display
full_image_bgr = cv2.imread(image_path)
if full_image_bgr is None:
    ok_video, full_image_bgr = run_video_frame_select_ui(image_path)
    if not ok_video:
        print("", "Unable to load image!", f"  @ {image_path}", sep="\n", flush=True)
        raise FileNotFoundError(osp.basename(image_path))


# ---------------------------------------------------------------------------------------------------------------------
# %% Capture model outputs

# Figure out which transformer block outputs we're trying to capture
is_v1_model, is_v2_model, is_v3_model = False, False, False
target_modules = None
if sammodel.name == "samv3":
    is_v3_model = True
    target_modules = (V3GlobalBlock, V3WindowBlock)
elif sammodel.name == "samv2":
    is_v2_model = True
    target_modules = (V2GlobalBlock, V2WindowBlock, V2PoolBlock)
elif sammodel.name == "samv1":
    is_v1_model = True
    target_modules = (V1GlobalBlock, V1WindowBlock)
else:
    raise TypeError("Unknown model type (expecting SAMv1, v2 or v3)")

# Capture target module output
captures = ModelOutputCapture(sammodel, target_modules=target_modules)
encoded_img, token_hw, preencode_img_hw = sammodel.encode_image(full_image_bgr, **imgenc_config_dict)
assert len(captures) > 0, "Error! No block data was captured... likely targeting the wrong blocks?"

# For SAMv1, windowed blocks will include a captured global block internally. We need to remove these!
# -> The internal global blocks have many 'windows' in the batch dimension
# -> So we remove any captured data with a batch size that isn't just 1 (assumes we don't batch more than 1 image!)
if not is_v2_model:
    captures = [cap for cap in captures if cap.shape[0] == 1]

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

# Print additional 'blocks per stage' info, which is relevant for interpreting block norm results
blocks_per_stage = model_config_dict.get("imgencoder_blocks_per_stage", "unknown")
if not is_v2_model:
    num_v1_blocks = model_config_dict.get("num_encoder_blocks", 0)
    num_v1_stages = model_config_dict.get("num_encoder_stages", 1)
    blocks_per_stage = tuple([num_v1_blocks // num_v1_stages] * num_v1_stages)
print("  Blocks per stage:", blocks_per_stage)


# ---------------------------------------------------------------------------------------------------------------------
# %% Compute (base) norm images

# Get information about captured token sizing (for display & control bounds)
max_tokens_h, max_tokens_w = -1, -1
max_channels = -1
for result in captures:
    _, res_h, res_w, res_ch = result.shape
    max_tokens_h = max(max_tokens_h, res_h)
    max_tokens_w = max(max_tokens_w, res_w)
    max_channels = max(max_channels, res_ch)

# Generate set of 'base' norm images, which are used to generate colormapped copies
# -> Also pre-calculate min/max values for re-use on updating the display
norm_max_side = max(max_tokens_h, max_tokens_w)
norm_scale_wh = tuple(round(256 * side / norm_max_side) for side in (max_tokens_w, max_tokens_h))
norm_minmax_list = []
base_norm_uint8_list = []
for result in captures:
    norm_tensor = result.squeeze(0).norm(dim=-1)
    norm_minmax_list.append((norm_tensor.min(), norm_tensor.max()))

    norm_uint8 = normalize_to_npuint8(norm_tensor)
    norm_uint8 = cv2.resize(norm_uint8, dsize=norm_scale_wh, interpolation=cv2.INTER_NEAREST_EXACT)
    norm_uint8 = cv2.copyMakeBorder(norm_uint8, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    base_norm_uint8_list.append(norm_uint8)


def redraw_norm_images(
    colormapped_norm_uint8_list,
    norm_min_max_list,
    text_color=(180, 180, 180),
    bg_color=(40, 40, 40),
    show_stats=True,
):
    """Hacky helper function used to redraw norm images with info text & colormapping"""

    # Set up text drawer to help with sizing to drawing area
    txtdraw = TextDrawer(color=text_color, thickness=2)

    out_images = []
    for block_idx, (image, (norm_min, norm_max)) in enumerate(zip(colormapped_norm_uint8_list, norm_min_max_list)):

        # Create info bar for drawing stats
        img_h, img_w = image.shape[0:2]
        bar_height = int(img_h * 0.15)
        info_bar = np.full((bar_height, img_w, 3), bg_color, dtype=np.uint8)

        # Write stats and form final norm image
        block_txt = f"B{block_idx}  [{norm_min:.3g}, {norm_max:.3g}]"
        info_bar = txtdraw.draw_to_box_norm(info_bar, block_txt, margin_xy_px=(20, 6))
        out_images.append(np.vstack((image.copy(), info_bar)) if show_stats else image.copy())

    return out_images


# ---------------------------------------------------------------------------------------------------------------------
# %% View results

# Set up bar for selecting the colormapping
cmap_bar = HColormapsBar(cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_INFERNO, cv2.COLORMAP_TURBO, None)

# Set up main display image
result_img = ExpandingImage(np.zeros((max_tokens_h, max_tokens_w, 3), dtype=np.uint8))

# Set up per-layer block norm images (as buttons for switching layers)
color_norm_list = [cmap_bar.apply_colormap(img) for img in base_norm_uint8_list]
norm_disp_list = redraw_norm_images(color_norm_list, norm_minmax_list)
norm_btns = ToggleImage.many(*norm_disp_list, highlight_color=(255, 255, 255))
norm_constraint = RadioConstraint(*norm_btns)

# Form grid of norm images, with aspect ratio to invert/balance the token aspect ratio
norm_grid = GridStack(*norm_btns, target_aspect_ratio=token_hw[0] / token_hw[1])

# Set up text displays
block_txt = ValueBlock("Block: ", 0)
min_txt = ValueBlock("Min: ", "-")
max_txt = ValueBlock("Min: ", "-")
channel_txt = ValueBlock("Channel: ", 0)
show_norm_btn = ToggleButton("Show Norm", default_state=False, text_scale=0.35, on_color=(35, 75, 200))
text_row = VStack(HStack(block_txt, min_txt, max_txt), HStack(show_norm_btn, channel_txt))

# Create slider to allow user to change which channel is being viewed
# -> Note different layers may have different channel counts, so a 0-to-1 range is used and scaled as needed!
channel_slider = HSlider(
    "Channel select",
    initial_value=0.5,
    min_value=0,
    max_value=1,
    step_size=1 / (max_channels - 1),
    enable_value_display=False,
)

# Set up message bars to communicate data info & controls
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(model_name, f"{token_hw_str} tokens", device_dtype_str, space_equally=True)
footer_msgbar = StaticMessageBar(
    "[Tab] Show norm", "[Arrows] Adjust channels/layers", "[g] Transpose grid", text_scale=0.35, space_equally=True
)

# Form final display layout
heading_bg_color = (0, 0, 0)
disp_layout = VStack(
    header_msgbar if show_info else None,
    cmap_bar,
    HStack(
        VStack(StaticMessageBar("Channel Features", bar_bg_color=heading_bg_color), result_img, text_row),
        HSeparator(8),
        VStack(StaticMessageBar("Block Norms", bar_bg_color=heading_bg_color), norm_grid),
    ),
    channel_slider,
    footer_msgbar if show_info else None,
)
render_limit_dict = {"h": display_size_px}

# Set up display & keyboard controls
window = DisplayWindow("Results - q to quit")
window.attach_mouse_callbacks(disp_layout)
window.attach_keypress_callback(KEY.UP_ARROW, norm_constraint.previous)
window.attach_keypress_callback(KEY.DOWN_ARROW, norm_constraint.next)
window.attach_keypress_callback(KEY.TAB, show_norm_btn.toggle)
window.attach_keypress_callback("g", norm_grid.transpose)

# For clarity, some additional keypress codes for controlling display size
KEY_ZOOM_IN = ord("=")
KEY_ZOOM_OUT = ord("-")

# Set up values that may not be set until hitting conditionals otherwise
channel_idx = 0
result_select = captures[0]
num_channels = result_select.shape[3]
max_allowable_channel_idx = num_channels - 1
result_uint8 = normalize_to_npuint8(result_select[0, :, :, channel_idx])

try:
    while True:

        # Read controls
        is_cmap_changed, _, _ = cmap_bar.read()
        is_channel_changed, channel_select_0to1 = channel_slider.read()
        is_block_changed, block_idx, _ = norm_constraint.read()
        is_show_norm_changed, show_norm_img = show_norm_btn.read()

        # Re-draw base norm images whenever the colormap changes
        if is_cmap_changed:
            color_norm_list = [cmap_bar.apply_colormap(img) for img in base_norm_uint8_list]
            norm_disp_list = redraw_norm_images(color_norm_list, norm_minmax_list)
            for btn, norm_img in zip(norm_btns, norm_disp_list):
                btn.set_image(norm_img)

        # Update indicator to say which block layer we're viewing
        if is_block_changed:
            block_txt.set_value(block_idx)

        # Update channel-specific values as needed
        need_channel_update = is_channel_changed or is_block_changed or is_show_norm_changed
        if need_channel_update:

            # Get new channel/norm data for display
            result_select = captures[block_idx]
            num_channels = result_select.shape[3]
            max_allowable_channel_idx = num_channels - 1
            channel_idx = round(channel_select_0to1 * max_allowable_channel_idx)
            if show_norm_img:
                result_tensor = result_select.squeeze(0).norm(dim=-1)
            else:
                result_tensor = result_select[0, :, :, channel_idx]
            result_uint8 = normalize_to_npuint8(result_tensor)

            # Update channel text values
            min_txt.set_value(round(result_tensor.min().item(), 2))
            max_txt.set_value(round(result_tensor.max().item(), 2))
            channel_txt.set_value("(norm)" if show_norm_img else channel_idx)

        # Get updated display sizing so we can do pixel-scaling as clean as possible for display
        render_h, render_w = result_img.get_render_hw()
        scale_wh = (render_w, render_h)
        result_as_color = cv2.resize(result_uint8, dsize=scale_wh, interpolation=cv2.INTER_NEAREST_EXACT)
        result_as_color = cmap_bar.apply_colormap(result_as_color)
        result_img.set_image(result_as_color)

        # Display results!
        display_image = disp_layout.render(**render_limit_dict)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        # Handle channel change on left/right arrow
        # -> It's possible for different layers to have different channel counts (e.g. on SAMv2),
        #    so inc/decrementing the channel slider by 1 doesn't always work nicely. Here, we
        #    increment the slider by an amount that will increment/decrement the channel index,
        #    regardless of the total number of channels on the current block layer!
        if keypress == KEY.LEFT_ARROW:
            prev_channel_idx = (channel_idx - 1) % num_channels
            channel_as_0to1 = prev_channel_idx / max_allowable_channel_idx
            channel_slider.set(channel_as_0to1, use_as_default_value=False)
        elif keypress == KEY.RIGHT_ARROW:
            next_channel_idx = (channel_idx + 1) % num_channels
            channel_as_0to1 = next_channel_idx / max_allowable_channel_idx
            channel_slider.set(channel_as_0to1, use_as_default_value=False)

        # Scale display size up when pressing +/- keys
        if keypress == KEY_ZOOM_IN:
            display_size_px = min(display_size_px + 50, 10000)
            render_limit_dict = {"h": display_size_px}
        if keypress == KEY_ZOOM_OUT:
            display_size_px = max(display_size_px - 50, 250)
            render_limit_dict = {"h": display_size_px}

except KeyboardInterrupt:
    print("", "Closed with Ctrl+C", sep="\n")

except Exception as err:
    raise err

finally:
    cv2.destroyAllWindows()
