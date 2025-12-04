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
from time import perf_counter

import torch
import cv2
import numpy as np

from muggled_sam.make_sam import make_sam_from_state_dict

from muggled_sam.demo_helpers.ui.window import DisplayWindow, KEY
from muggled_sam.demo_helpers.ui.images import ExpandingImage
from muggled_sam.demo_helpers.ui.layout import HStack, VStack
from muggled_sam.demo_helpers.ui.buttons import ToggleButton
from muggled_sam.demo_helpers.ui.sliders import HSlider, HMultiSlider
from muggled_sam.demo_helpers.ui.static import StaticMessageBar
from muggled_sam.demo_helpers.ui.colormaps import HColormapsBar, make_spectral_colormap
from muggled_sam.demo_helpers.shared_ui_layout import PromptUIControl, PromptUI
from muggled_sam.demo_helpers.ui.helpers.images import get_image_hw_for_max_height

from muggled_sam.demo_helpers.video_frame_select_ui import run_video_frame_select_ui
from muggled_sam.demo_helpers.contours import get_contours_from_mask

from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing, load_init_prompts
from muggled_sam.demo_helpers.misc import get_default_device_string, make_device_config, normalize_to_npuint8


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def encode_image_samv3(
    model,
    image_bgr: np.ndarray,
    max_side_length: int = 1008,
    use_square_sizing: bool = True,
) -> tuple[list, list]:

    with torch.inference_mode():

        # Generate normal image encodings (i.e. with feature projection)
        hflip_image_bgr = np.fliplr(image_bgr)
        reg_encimg_list, _, _ = model.encode_image(image_bgr, max_side_length, use_square_sizing)
        flip_encimg_list, _, _ = model.encode_image(hflip_image_bgr, max_side_length, use_square_sizing)

        # Combine regular encodings with horizontal flip in batch dimension for output
        proj_encimg_list = []
        for regular_enc, flipped_enc in zip(reg_encimg_list, flip_encimg_list):
            batch_combined_encs = torch.concat((regular_enc, torch.flipud(flipped_enc)), dim=0)
            proj_encimg_list.append(batch_combined_encs)

        # Compute 'raw' encodings
        # -> Works by manually running the image encoder, without the final projection step
        image_tensor_bchw = model.image_encoder.prepare_image(image_bgr, imgenc_base_size, use_square_sizing)
        flipped_image_bchw = model.image_encoder.prepare_image(hflip_image_bgr, imgenc_base_size, use_square_sizing)
        result_tokens = []
        for input_img in [image_tensor_bchw, flipped_image_bchw]:
            raw_encoded_bchw = model.image_encoder(input_img)
            result_tokens.append(raw_encoded_bchw)

        # Combine regular 'raw' encoding with horizontal flip (stored in batch dimension)
        reg_encimg, flip_encimg = result_tokens
        raw_encimg = torch.concat((reg_encimg, torch.fliplr(flip_encimg)), dim=0)

    return [raw_encimg], proj_encimg_list


def encode_image_samv2(model, image_bgr, max_side_length=1024, use_square_sizing=True) -> tuple[list, list]:
    """
    Helper function used to compute 'normal' image encodings along with 'raw' encodings,
    which don't include the final projection steps that usually occur within the
    image encoder of SAMv2. This function also computes horizontally flipped copies
    of the encodings (for both the 'raw' and 'normal/projection' versions) and stores
    the h-flipped encodings in index-1 of the batch dimension of the outputs

    Returns:
        raw_image_encoding_list, projection_image_encoding_list
        -> Raw list contains 4 elements, projection list contains 3
        -> Both lists are ordered so the lowest-resolution encodings are in index 0
        -> Index 1 of raw encodings matches resoution of index 0 of projection encodings!
        -> All encodings have shape: 2xFxHxW
        -> 2 for regular & h-flipped results
        -> F varies per list item (as well as by model size for raw results)
        -> H & W are height/width of token grid, which varies by list item
    """

    with torch.inference_mode():

        # Generate normal image encodings (i.e. with feature projection)
        hflip_image_bgr = np.fliplr(image_bgr)
        reg_encimg_list, _, _ = model.encode_image(image_bgr, max_side_length, use_square_sizing)
        flip_encimg_list, _, _ = model.encode_image(hflip_image_bgr, max_side_length, use_square_sizing)

        # Combine regular encodings with horizontal flip in batch dimension for output
        proj_encimg_list = []
        for regular_enc, flipped_enc in zip(reg_encimg_list, flip_encimg_list):
            batch_combined_encs = torch.concat((regular_enc, torch.flipud(flipped_enc)), dim=0)
            proj_encimg_list.append(batch_combined_encs)

        # Compute 'raw' encodings
        # -> Works by manually running the image encoder, without the final projection step
        image_tensor_bchw = model.image_encoder.prepare_image(image_bgr, imgenc_base_size, use_square_sizing)
        flipped_image_bchw = model.image_encoder.prepare_image(hflip_image_bgr, imgenc_base_size, use_square_sizing)
        result_tokens = []
        for input_img in [image_tensor_bchw, flipped_image_bchw]:
            patch_tokens_bchw = model.image_encoder.patch_embed(input_img)
            patch_tokens_bchw = model.image_encoder.posenc(patch_tokens_bchw)
            multires_tokens_bchw_list = model.image_encoder.hiera(patch_tokens_bchw)
            result_tokens.append(multires_tokens_bchw_list)

        # Combine regular 'raw' encodings with horizontal flips
        reg_rawenc_list, flip_rawenc_list = result_tokens
        raw_encimg_list = []
        for regular_enc, flipped_enc in zip(reg_rawenc_list, flip_rawenc_list):
            batch_combined_encs = torch.concat((regular_enc, torch.flipud(flipped_enc)), dim=0)
            raw_encimg_list.append(batch_combined_encs)

        # Reverse raw encoding order to match projection order (i.e. lowest-res features are in 0-th index)
        raw_encimg_list = list(reversed(raw_encimg_list))

    return raw_encimg_list, proj_encimg_list


def encode_image_samv1(model, image_bgr, max_side_length=1024, use_square_sizing=True) -> tuple[list, list]:
    """
    Helper function used to compute 'normal' image encodings along with 'raw' encodings,
    which don't include the final projection steps that usually occur within the
    image encoder of SAMv1. This function also computes horizontally flipped copies
    of the encodings (for both the 'raw' and 'normal/projection' versions) and stores
    the h-flipped encodings in index-1 of the batch dimension of the outputs

    Note also that the resulting encodings are returned inside lists (e.g. lists of 1 element),
    for easier compatibility with SAMv2, which also produces list results.

    Returns:
        [raw_image_encoding], [projection_image_encoding]
        -> Both have shape: 2xFxHxW
        -> 2 for regular & h-flipped results
        -> F is 256 for projection result, but varies (by model size) for raw results
        -> H & W are height/width of token grid (64x64 by default)
    """

    with torch.inference_mode():

        # Generate normal image encodings (i.e. with feature projection)
        hflip_image_bgr = np.fliplr(image_bgr)
        reg_encimg, _, _ = model.encode_image(image_bgr, max_side_length, use_square_sizing)
        flip_encimg, _, _ = model.encode_image(hflip_image_bgr, max_side_length, use_square_sizing)

        # Combine regular encodings with horizontal flip in batch dimension for output
        proj_encimg = torch.concat((reg_encimg, torch.fliplr(flip_encimg)), dim=0)

        # Compute 'raw' encodings
        # -> Works by manually running the image encoder, without the final projection step
        image_tensor_bchw = model.image_encoder.prepare_image(image_bgr, imgenc_base_size, use_square_sizing)
        flipped_image_bchw = model.image_encoder.prepare_image(hflip_image_bgr, imgenc_base_size, use_square_sizing)
        result_tokens = []
        for input_img in [image_tensor_bchw, flipped_image_bchw]:
            patch_tokens_bchw = model.image_encoder.patch_embed(input_img)
            tokens_bhwc = model.image_encoder.posenc(patch_tokens_bchw).permute(0, 2, 3, 1)
            raw_encoded_bchw = model.image_encoder.stages(tokens_bhwc).permute(0, 3, 1, 2)
            result_tokens.append(raw_encoded_bchw)

        # Combine regular 'raw' encodings with horizontal flips
        reg_encimg, flip_encimg = result_tokens
        raw_encimg = torch.concat((reg_encimg, torch.fliplr(flip_encimg)), dim=0)

    # Return encodings as a 'list of one element' to mimic SAMv2 list structure
    # -> This way we can use same indexing code for both SAMv1 & v2
    return [raw_encimg], [proj_encimg]


def compute_similarity(reference_encoded_image, mask_prediction, comparison_encoded_image) -> torch.Tensor:
    """
    Function used to calculate a 'similarity image' by comparing masked image tokens from
    a given reference image (and mask) to a given comparison image. This computation is
    based on a post on the SAMv1 issues board:
    https://github.com/facebookresearch/segment-anything/issues/283#issuecomment-1531989328
    -> Note that this implementation does not perform any cropping

    This calculation works by averaging all of the reference image tokens that fall within
    the given mask and then computing the cosine similarity between the averaged token
    and all tokens in the comparison image. Each 'pixel' in the output will have a value
    between -1.0 and +1.0, indicating how similar that pixel is to the averaged image token.

    Note that the averaging step of the provided reference tokens will average *across*
    the batch dimension (i.e. averaging doesn't happen independently on each batch item).
    This is intended to allow for providing a horizontally-flipped copy of the image tokens
    as a second batch item, so both the normal & h-flipped tokens are averaged together!

    Returns:
        similarity_tensor (shape: HxW, H height, W width matching given comparison tokens)
    """

    with torch.inference_mode():
        scaled_encimg_bchw = torch.nn.functional.interpolate(
            reference_encoded_image,
            size=mask_prediction.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Get 'average' token associated with masked object
        # -> Assume the batch dimension holds a 'normal' and 'h-flipped' copy of encodings
        # -> We want to average all masked tokens, across both batch entries (if present)
        binary_mask = mask_prediction > 0.0
        masked_image_tokens_bcn = scaled_encimg_bchw.flatten(2)[:, :, binary_mask.flatten()]
        avg_token = masked_image_tokens_bcn.permute(1, 0, 2).flatten(start_dim=1).mean(dim=1)

        # Find cosine similarity between all tokens of comparison image compared to 'averaged' token
        normed_avg_token = avg_token / avg_token.norm()
        normed_comp_encimg = comparison_encoded_image / comparison_encoded_image.norm(dim=1)
        similarity_tensor = torch.einsum("c, bchw -> bhw", normed_avg_token, normed_comp_encimg)

    # Remove batch index for final output
    return similarity_tensor[0]


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
parser = argparse.ArgumentParser(description="Visualize similarity of SAM image tokens between two images")
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
    "--same",
    default=False,
    action="store_true",
    help="If set, the same image will be used for both prompting/masking & similarity comparison",
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
use_square_sizing = not args.use_aspect_ratio
imgenc_base_size = args.base_size_px
show_info = not args.hide_info
use_same_image = args.same
force_hstack = args.hstack
force_vstack = args.vstack

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Create history to re-use selected inputs
root_path = osp.dirname(osp.dirname(__file__))
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_crossimgpath = history.read("cross_image_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path_a = ask_for_path_if_missing(arg_image_path_a, "image", history_imgpath)
if not use_same_image:
    image_path_b = ask_for_path_if_missing(arg_image_path_b, "cross image", history_crossimgpath)
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)

# Store history for use on reload
if use_same_image:
    history.store(image_path=image_path_a, model_path=model_path)
else:
    history.store(image_path=image_path_a, model_path=model_path, cross_image_path=image_path_b)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Get the model name, for reporting
model_name = osp.basename(model_path)
print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
sammodel.to(**device_config_dict)

# Load reference image
image_bgr_a = cv2.imread(image_path_a)
if image_bgr_a is None:
    ok_video, image_bgr_a = run_video_frame_select_ui(image_path_a)
    if not ok_video:
        print("", "Unable to load image!", f"  @ {image_path_a}", sep="\n", flush=True)
        raise FileNotFoundError(osp.basename(image_path_a))

# Load comparison image (or duplicate reference image)
image_bgr_b = cv2.imread(image_path_b) if not use_same_image else image_bgr_a.copy()
if image_bgr_b is None:
    ok_video, image_bgr_b = run_video_frame_select_ui(image_path_b)
    if not ok_video:
        print("", "Unable to load image!", f"  @ {image_path_b}", sep="\n", flush=True)
        raise FileNotFoundError(osp.basename(image_path_b))

# Determine stacking direction
use_hstack_images = None
if force_hstack:
    use_hstack_images = True
elif force_vstack:
    use_hstack_images = False
else:
    img_a_h, img_a_w = image_bgr_a.shape[0:2]
    img_b_h, img_b_w = image_bgr_b.shape[0:2]
    have_narrow_img = (img_a_h > img_a_w) or (img_b_h > img_b_w)
    use_hstack_images = have_narrow_img


# ---------------------------------------------------------------------------------------------------------------------
# %% Initial model run

# Determine which image encoding function to use
is_v1_model, is_v2_model, is_v3_model = False, False, False
encode_image_func = None
if sammodel.name == "samv3":
    is_v3_model = True
    encode_image_func = encode_image_samv3
elif sammodel.name == "samv2":
    is_v2_model = True
    encode_image_func = encode_image_samv2
elif sammodel.name == "samv1":
    is_v1_model = True
    encode_image_func = encode_image_samv1
else:
    raise TypeError("Unknown model type (expecting SAMv1, v2 or v3)")

# Run Model
imgenc_config_dict = {"max_side_length": imgenc_base_size, "use_square_sizing": use_square_sizing}
print("", "Encoding image data...", sep="\n", flush=True)
t1 = perf_counter()
raw_enc_a, proj_enc_a = encode_image_func(sammodel, image_bgr_a, **imgenc_config_dict)
raw_enc_b, proj_enc_b = encode_image_func(sammodel, image_bgr_b, **imgenc_config_dict)
t2 = perf_counter()
init_time_taken_ms = round(1000 * (t2 - t1), 1)
print(f"  -> Took {init_time_taken_ms} ms", flush=True)

# For convenience, create the usual 'encoded image' value needed for mask generation
# -> We already computed this, but with support for h-flipping
# -> Here we take the non-h-flipped encodings (i.e. batch index 0) as our encoding for mask generation
encoded_img_a = [enc[[0]] for enc in proj_enc_a]
encoded_img_b = [enc[[0]] for enc in proj_enc_b]

# Remove list indexing for SAMv1 models, which normally don't have this!
if is_v1_model:
    encoded_img_a = encoded_img_a[0]
    encoded_img_b = encoded_img_b[0]

# Run model without prompts as sanity check. Also gives initial result values
encoded_prompts = sammodel.encode_prompts([], [], [])
init_mask_preds, iou_preds = sammodel.generate_masks(encoded_img_a, encoded_prompts, blank_promptless_output=True)
mask_uint8 = normalize_to_npuint8(init_mask_preds[0, 0, :, :])

# Provide some feedback about how the model is running
token_hw_a, token_hw_b = proj_enc_a[0].shape[2:], proj_enc_b[0].shape[2:]
model_device = device_config_dict["device"]
model_dtype = str(device_config_dict["dtype"]).split(".")[-1]
token_hw_a_str, token_hw_b_str = f"{token_hw_a[0]} x {token_hw_a[1]}", f"{token_hw_b[0]} x {token_hw_b[1]}"
print(
    "",
    f"Config ({model_name}):",
    f"  Device: {model_device} ({model_dtype})",
    f"  ImageA tokens HW: {token_hw_a_str}",
    f"  ImageB tokens HW: {token_hw_b_str}",
    sep="\n",
    flush=True,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up the UI

# Set up shared UI elements & control logic
ui_elems = PromptUI(image_bgr_a, init_mask_preds, tool_button_text_scale=0.5)
uictrl = PromptUIControl(ui_elems)

# Set up comparison image element
comp_img_elem = ExpandingImage(image_bgr_b)

# Set up control buttons
swap_images_btn, raw_feats_btn, hflip_btn, similarity_btn = ToggleButton.many(
    "Swap images", "Raw features", "H-flip", "Show similarity", text_scale=0.5
)
swap_images_btn.set_is_changed()
similarity_btn.toggle(True)
similarity_btn.style(on_color=(70, 30, 120), off_color=(50, 40, 60))

# Set up sliders
max_enc_idx = max(len(proj_enc_a), len(raw_enc_a)) - 1
encoding_select_slider = HSlider(
    "Encoding resolution",
    initial_value=0,
    min_value=0,
    max_value=max_enc_idx,
    step_size=1,
    marker_steps=1,
)
norm_range_slider = HMultiSlider(
    "Normalization range",
    initial_values=(0.0, 1.0),
    min_value=-1.0,
    max_value=1.0,
    step_size=0.05,
    fill_between_points=True,
)

# Set up message bars
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(model_name, device_dtype_str, space_equally=True)
footer_msgbar = StaticMessageBar(
    "[s] Swap" if not use_same_image else None,
    "[r] Raw features",
    "[h] H-Flip",
    "[tab] Similarity",
    "[, .] Change encoding" if (not is_v1_model) else None,
    text_scale=0.35,
    space_equally=True,
)

# Create bar of colormaps for adjusting display style
cmap_bar = HColormapsBar(cv2.COLORMAP_INFERNO, cv2.COLORMAP_VIRIDIS, make_spectral_colormap(), None)

# Set up full display layout
StackElem: HStack | VStack = HStack if use_hstack_images else VStack
disp_layout = VStack(
    header_msgbar if show_info else None,
    cmap_bar,
    StackElem(ui_elems.layout, comp_img_elem),
    HStack(swap_images_btn if not use_same_image else None, hflip_btn, raw_feats_btn, similarity_btn),
    encoding_select_slider if (not is_v1_model) else None,
    norm_range_slider,
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
window.attach_keypress_callback("h", hflip_btn.toggle)
window.attach_keypress_callback("s", swap_images_btn.toggle)
window.attach_keypress_callback("r", raw_feats_btn.toggle)
window.attach_keypress_callback(KEY.TAB, similarity_btn.toggle)
window.attach_keypress_callback(",", encoding_select_slider.decrement)
window.attach_keypress_callback(".", encoding_select_slider.increment)

# For clarity, some additional keypress codes
KEY_ZOOM_OUT, KEY_ZOOM_IN = ord("-"), ord("=")

# Pre-compute upper image scaling targets (this is just for display efficiency in case we get overly large input images)
disp_hw_a = get_image_hw_for_max_height(image_bgr_a, max_height_px=display_size_px)
disp_hw_b = get_image_hw_for_max_height(image_bgr_b, max_height_px=display_size_px)

# Set up copies of variables that need to exist before first run
init_preds_hw = init_mask_preds.shape[2:]
preview_preds = init_mask_preds
mask_preds = init_mask_preds
mask_contours_norm = None
sim_img_uint8 = image_bgr_b * 0

# *** Main display loop ***
try:
    while True:

        # Get current display sizing (changes if user zooms display in or out)
        ref_hw = ui_elems.image.get_render_hw()
        comp_hw = comp_img_elem.get_render_hw()
        disp_hw_comp = comp_hw

        # Read prompt input data & selected mask
        need_prompt_encode, prompts = uictrl.read_prompts()
        is_mask_changed, mselect_idx, selected_mask_btn = ui_elems.masks_constraint.read()

        # Read controls
        is_cmap_changed, _, _ = cmap_bar.read()
        is_hflip_changed, use_hflip = hflip_btn.read()
        is_swap_changed, use_image_swap = swap_images_btn.read()
        is_rawfeats_changed, use_raw_feats = raw_feats_btn.read()
        is_normrange_changed, (min_similarity, max_similarity) = norm_range_slider.read()
        is_stage_changed, encoding_select_idx = encoding_select_slider.read()
        is_sim_controls_changed = is_normrange_changed or is_hflip_changed or is_rawfeats_changed or is_stage_changed

        # Special toggle: Switch to similarity view whenever controls would adjust it's appearance
        if is_sim_controls_changed:
            similarity_btn.toggle(True)
        is_showsim_changed, show_similarity = similarity_btn.read()

        # Handle swapping of images (e.g. so comparison image becomes promptable)
        if is_swap_changed:

            # Set 'default' assignments of reference vs. comparison data
            ref_image_bgr, comp_image_bgr = image_bgr_a, image_bgr_b
            proj_enc_ref, proj_enc_comp = proj_enc_a, proj_enc_b
            raw_enc_ref, raw_enc_comp = raw_enc_a, raw_enc_b
            disp_hw_ref, disp_hw_comp = disp_hw_a, disp_hw_b
            encoded_img = encoded_img_a

            # Swap reference/comparison data if needed
            if use_image_swap:
                ref_image_bgr, comp_image_bgr = comp_image_bgr, ref_image_bgr
                proj_enc_ref, proj_enc_comp = proj_enc_comp, proj_enc_ref
                raw_enc_ref, raw_enc_comp = raw_enc_comp, raw_enc_ref
                disp_hw_ref, disp_hw_comp = disp_hw_comp, disp_hw_ref
                encoded_img = encoded_img_b

            # Wipe out existing prompts & update displayed images
            ui_elems.image.set_image(ref_image_bgr)
            comp_img_elem.set_image(comp_image_bgr * 0)

        # Compute mask predictions as needed
        need_update_prediction = need_prompt_encode or is_swap_changed
        if need_prompt_encode or is_swap_changed:
            encoded_prompts = sammodel.encode_prompts(*prompts)
            mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts, blank_promptless_output=True)
            uictrl.update_mask_previews(mask_preds)

        # Update masking + outline used to indicate segmentation result
        need_update_mask_outline = need_update_prediction or is_mask_changed
        if need_update_mask_outline:

            # Scale selected mask to match display sizing
            mask_select = mask_preds[:, mselect_idx, :, :].unsqueeze(1)
            mask_scaled = torch.nn.functional.interpolate(
                mask_select,
                size=ref_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(dim=(0, 1))

            # Process contour data for overlay
            mask_uint8 = ((mask_scaled > 0.0).byte() * 255).cpu().numpy()
            ok_contours, mask_contours_norm = get_contours_from_mask(mask_uint8, normalize=True)
            ui_elems.olays.polygon.set_polygons(mask_contours_norm)

        # Update similarity calculation/display as needed
        need_update_similarity = need_update_mask_outline or is_sim_controls_changed or is_cmap_changed
        if need_update_similarity:

            # For clarity, set up indexing into raw vs. projection encodings
            # -> When using SAMv2, we use encoding selector to choose raw-indexing,
            #    then set projection index to 1 less (there are 4 raw encodings, 3 projection encodings),
            #    as this leads to a less jarring efect when toggling raw vs. proj. display
            raw_idx = min(encoding_select_idx, len(raw_enc_ref) - 1)
            proj_idx = max(raw_idx - 1, 0) if is_v2_model else encoding_select_idx

            # Pick appropriate encodings based on control settings
            # There are 4 options: projection encodings + hflip, project + no hflip, raw + hflip, raw + no hflip
            # -> We need to pick matching (project vs. raw) encodings for reference vs. comparison image
            # -> H-flip data is held in the batch dimension of encodings (index 0 is not flipped, 1 is h-flipped)
            # -> The comparison image always uses the non-hflip encodings to match the (not flipped) image itself
            ref_enc = raw_enc_ref[raw_idx] if use_raw_feats else proj_enc_ref[proj_idx]
            comp_enc = raw_enc_comp[raw_idx][[0]] if use_raw_feats else proj_enc_comp[proj_idx][[0]]
            if not use_hflip:
                ref_enc = ref_enc[[0]]

            # Calculate similarity & limit value range, which affects colormapping
            similarity_tensor = compute_similarity(ref_enc, mask_select, comp_enc)
            similarity_tensor = torch.clamp(similarity_tensor, min_similarity, max_similarity)

            # Convert similarity to colormapped image for display
            sim_wh = disp_hw_comp[::-1]
            sim_img_uint8 = normalize_to_npuint8(similarity_tensor)
            sim_img_uint8 = cmap_bar.apply_colormap(sim_img_uint8)
            sim_img_uint8 = cv2.resize(sim_img_uint8, dsize=sim_wh, interpolation=cv2.INTER_NEAREST_EXACT)

        # Allow toggling the similarity image vs. original color image
        need_update_similarity_display = need_update_similarity or is_showsim_changed
        if need_update_similarity_display:
            comp_img_elem.set_image(sim_img_uint8 if show_similarity else comp_image_bgr)

        # Render final output
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

finally:
    cv2.destroyAllWindows()
