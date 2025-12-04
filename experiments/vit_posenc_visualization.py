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

import torch
import cv2

from muggled_sam.make_sam import make_sam_from_state_dict

from muggled_sam.demo_helpers.ui.window import DisplayWindow, KEY
from muggled_sam.demo_helpers.ui.base import force_same_min_width
from muggled_sam.demo_helpers.ui.layout import HStack, VStack
from muggled_sam.demo_helpers.ui.buttons import ToggleButton
from muggled_sam.demo_helpers.ui.sliders import HSlider
from muggled_sam.demo_helpers.ui.static import StaticMessageBar, HSeparator
from muggled_sam.demo_helpers.ui.images import ExpandingImage

from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import ask_for_model_path_if_missing
from muggled_sam.demo_helpers.misc import get_default_device_string, make_device_config, normalize_to_npuint8


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_model_path = None
default_display_size = 900

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to visualize position encodings of SAM image encoders")
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
    "--hide_info",
    default=False,
    action="store_true",
    help="Hide text info elements from UI",
)
parser.add_argument(
    "--hide_norm",
    default=False,
    action="store_true",
    help="Hide the L2 norm display image from the UI",
)


# For convenience
args = parser.parse_args()
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
show_info = not args.hide_info
show_l2_norm = not args.hide_norm

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# # Create history to re-use selected inputs
root_path = osp.dirname(osp.dirname(__file__))
history = HistoryKeeper(root_path)
_, history_modelpath = history.read("model_path")
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)
history.store(model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Get the model name, for reporting
model_name = osp.basename(model_path)

print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
sammodel.to(**device_config_dict)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up position encoder


class PosencExtractor:
    """
    Helper class used to contain/manage positional encoderss of SAMv1/v2 models
    Main use is to create an instance and use it to generate new position encodings,
    for example:
        px = PosencExtractor(sam_model)
        new_position_encodings = px.make_new_encodings(patch_height=64, patch_height=32)
    """

    def __init__(self, sam_model):

        # Sanity check
        is_v1_model = sam_model.name == "samv1"
        is_v2_model = sam_model.name == "samv2"
        is_v3_model = sam_model.name == "samv3"
        assert any((is_v1_model, is_v2_model, is_v3_model)), "Unrecognized model! Cannot access positional encodings..."

        # Try to 'reach in' to access function used to generate position encodings
        try:
            posencoder = sam_model.image_encoder.posenc
            self._posenc_func = posencoder._scale_to_patch_grid
        except AttributeError:
            raise NameError(
                """Unable to find positional encoding function!
                This likely means the functions were renamed and this script is out-of-date...
                """
            )

        # Special handling of V2 window tiling, which we may disable
        self._window_tile_copy = None
        self._use_window_tile = True
        if is_v2_model:
            base_window_tile = posencoder.base_window_tile
            self._window_tile_copy = torch.nn.Parameter(base_window_tile)
            self._no_window_tile_copy = torch.nn.Parameter(base_window_tile * 0.0)

        # Store reference to position encoder, in case we're modifying window tiling later
        self._posenc = posencoder
        self.is_v1_model = is_v1_model
        self.is_v2_model = is_v2_model
        self.is_v3_model = is_v3_model

    def make_new_encodings(self, patch_height, patch_width):
        """Re-computes positional encodings for the given height & width"""
        with torch.inference_mode():
            posencoding_bchw = self._posenc_func((patch_height, patch_width))
        return posencoding_bchw

    def toggle_window_tiling(self):
        """
        Window tiling is an additive encoding that is repeated
        across the entire positional encoding when using SAMv2
        (an 8x8 tile, by default). This function can be used to
        turn off the tiling, since it adds a 'noisy' pattern
        to the underlying encoding.

        It is called 'pos_embed_window' in the original code
        base and it's use can be seen here:
        https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/backbones/hieradet.py#L269-L271

        The window tile encoding is specific to SAMv2, so this function
        does nothing when using V1 or V3.
        """

        # Only toggle when using V2 model
        if self.is_v2_model:

            # Swap between using the tiling or not
            self._use_window_tile = not self._use_window_tile
            self._posenc.base_window_tile = (
                self._window_tile_copy if self._use_window_tile else self._no_window_tile_copy
            )

            # Force cache reset so old encodings aren't re-used
            self._posenc.cached_encoding_bchw = torch.zeros((1, 1, 1, 1))

        return self


# Set up extractor and create example encoding for feature size information
posextract = PosencExtractor(sammodel)
base_h, base_w = (72, 72) if posextract.is_v3_model else (64, 64)
example_posenc_bchw = posextract.make_new_encodings(base_h, base_w)
_, features_per_token, _, _ = example_posenc_bchw.shape
is_v2_model = posextract.is_v2_model

# Calculate initial position encoding
posenc_bchw_tensor = example_posenc_bchw.clone()
posenc_norm = posenc_bchw_tensor.norm(dim=1).squeeze(0)

# Calculate initial display results
feats_uint8 = normalize_to_npuint8(posenc_bchw_tensor[0, 0, :, :])
posenc_norm_uint8 = normalize_to_npuint8(posenc_norm)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up the UI

# Change tools/masks on arrow keys
image_elem = ExpandingImage(feats_uint8)
separator = HSeparator(16, color=(0, 0, 0))
norm_image_elem = ExpandingImage(posenc_norm_uint8)

# Set up interactive elements
init_idx, feat_marker_steps = features_per_token // 2, (1 + (features_per_token // 10) // 10) * 10
tile_toggle_btn = ToggleButton("Include Window Tiling", on_color=(50, 120, 140), default_state=True, text_scale=0.35)
feature_slider = HSlider("Feature Index", init_idx, 0, features_per_token - 1, 1, marker_steps=feat_marker_steps)
height_slider = HSlider("Height", base_h, 2, 2 * base_h, 1, marker_steps=8)
width_slider = HSlider("Width", base_w, 2, 2 * base_w, 1, marker_steps=8)

# If using v2 model, include tile toggle beside feature slider (toggle is not rendered for v1 models)
feature_bar = feature_slider
if is_v2_model:
    force_same_min_width(feature_slider, tile_toggle_btn)
    feature_bar = HStack(feature_slider, tile_toggle_btn)

# Make info bars
header_bar = StaticMessageBar("Position Encoding Per Feature", "L2 Norm", space_equally=True)
if not show_l2_norm:
    header_bar = StaticMessageBar("Position Encoding Per Feature")

footer_bar = StaticMessageBar(
    "[arrow keys] Change features",
    "[w,a,s,d] Change height & width",
    "[-/+] Change display size",
    text_scale=0.35,
    space_equally=True,
)

# Set up full display layout
disp_layout = VStack(
    header_bar if show_info else None,
    HStack(image_elem, separator, norm_image_elem) if show_l2_norm else image_elem,
    feature_bar,
    height_slider,
    width_slider,
    footer_bar if show_info else None,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Main display

# Set up display
cv2.destroyAllWindows()
window = DisplayWindow("Display - q to quit", display_fps=60).attach_mouse_callbacks(disp_layout)
window.move(200, 50)

# Attach key controls
window.attach_keypress_callback("t", tile_toggle_btn.toggle)
window.attach_keypress_callback("w", height_slider.increment)
window.attach_keypress_callback("s", height_slider.decrement)
window.attach_keypress_callback("a", width_slider.decrement)
window.attach_keypress_callback("d", width_slider.increment)
window.attach_keypress_callback(KEY.LEFT_ARROW, feature_slider.decrement)
window.attach_keypress_callback(KEY.DOWN_ARROW, feature_slider.decrement)
window.attach_keypress_callback(KEY.RIGHT_ARROW, feature_slider.increment)
window.attach_keypress_callback(KEY.UP_ARROW, feature_slider.increment)

# For clarity, some additional keypress codes
KEY_ZOOM_OUT, KEY_ZOOM_IN = ord("-"), ord("=")

# Calculate initial position encoding
posenc_bchw_tensor = example_posenc_bchw.clone()
posenc_norm = posenc_bchw_tensor.norm(dim=1).squeeze(0)

# Calculate initial display results
posenc_norm_uint8 = normalize_to_npuint8(posenc_norm)
feats_uint8 = normalize_to_npuint8(posenc_bchw_tensor[0, 0, :, :])

# *** Main display loop ***
try:
    while True:

        # Read controls
        tile_changed, use_window_tiling = tile_toggle_btn.read()
        f_changed, feat_idx = feature_slider.read()
        h_changed, patch_h = height_slider.read()
        w_changed, patch_w = width_slider.read()

        # Turn tiling on/off as needed (specific to SAMv2)
        if tile_changed:
            posextract.toggle_window_tiling()

        # Re-generate the position encoding as needed
        regen_posenc = h_changed or w_changed or tile_changed
        if regen_posenc:

            # Generate new position encodings
            posenc_bchw_tensor = posextract.make_new_encodings(patch_h, patch_w)

            # Calculate token norms
            posenc_norm = posenc_bchw_tensor.norm(dim=1).squeeze(0)
            posenc_norm_uint8 = normalize_to_npuint8(posenc_norm)

        # Switch feature channel we're displaying
        if f_changed or regen_posenc:
            feats_uint8 = normalize_to_npuint8(posenc_bchw_tensor[0, feat_idx, :, :])

        # Scale to fixed display size
        renderdisp_h, renderdisp_w = image_elem.get_render_hw()
        scale_mult = max(1, max(renderdisp_h, renderdisp_w) // max(feats_uint8.shape))
        resize_dict = {"dsize": None, "fx": scale_mult, "fy": scale_mult, "interpolation": cv2.INTER_NEAREST_EXACT}
        disp_feats = cv2.resize(feats_uint8, **resize_dict)
        disp_norm = cv2.resize(posenc_norm_uint8, **resize_dict)

        # Apply colormap
        disp_feats = cv2.applyColorMap(disp_feats, cv2.COLORMAP_VIRIDIS)
        disp_norm = cv2.applyColorMap(disp_norm, cv2.COLORMAP_VIRIDIS)

        # Pad display images to fill space, to avoid interpolation on display
        feats_h, feats_w = disp_feats.shape[0:2]
        v_pad = max(0, renderdisp_h - feats_h)
        h_pad = max(0, renderdisp_w - feats_w)
        t_pad, l_pad = v_pad // 2, h_pad // 2
        b_pad, r_pad = v_pad - t_pad, h_pad - l_pad
        disp_feats = cv2.copyMakeBorder(disp_feats, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT)
        disp_norm = cv2.copyMakeBorder(disp_norm, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT)

        # Render final output
        image_elem.set_image(disp_feats)
        norm_image_elem.set_image(disp_norm)
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
