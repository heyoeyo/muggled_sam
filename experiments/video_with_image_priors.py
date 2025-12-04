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

import torch
import cv2

from muggled_sam.make_sam import make_sam_from_state_dict

from muggled_sam.demo_helpers.ui.video import LoopingVideoReader, LoopingVideoPlaybackSlider, ValueChangeTracker
from muggled_sam.demo_helpers.ui.window import DisplayWindow
from muggled_sam.demo_helpers.ui.layout import HStack, VStack
from muggled_sam.demo_helpers.ui.buttons import ToggleButton, ImmediateButton
from muggled_sam.demo_helpers.ui.text import TitledTextBlock, ValueBlock
from muggled_sam.demo_helpers.ui.static import StaticMessageBar
from muggled_sam.demo_helpers.shared_ui_layout import PromptUIControl, PromptUI, ReusableBaseImage

from muggled_sam.demo_helpers.video_frame_select_ui import run_video_frame_select_ui
from muggled_sam.demo_helpers.contours import get_contours_from_mask
from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults

from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from muggled_sam.demo_helpers.misc import PeriodicVRAMReport, get_default_device_string, make_device_config


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_video_path = None
default_model_path = None
default_prompts_path = None
default_display_size = 900
default_base_size = None
default_max_memory_history = 6
default_max_pointer_history = 15
default_show_iou_preds = False

# Define script arguments
parser = argparse.ArgumentParser(description="Run SAMV2 video segmentation with prompting from a separate image")
parser.add_argument("-i", "--image_path", default=default_image_path, help="Path to input image")
parser.add_argument("-v", "--video_path", default=default_video_path, help="Path to input video")
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
    "--max_memories",
    default=default_max_memory_history,
    type=int,
    help=f"Maximum number of previous-frame memory encodings to store (default {default_max_memory_history})",
)
parser.add_argument(
    "--max_pointers",
    default=default_max_pointer_history,
    type=int,
    help=f"Maximum number of previous-frame object pointers to store (default {default_max_pointer_history})",
)
parser.add_argument(
    "--discard_on_bad_objscore",
    default=False,
    action="store_true",
    help="If set, low object-score masks (during video segmentation) will be discarded",
)
parser.add_argument(
    "--disable_segment_on_startup",
    default=False,
    action="store_true",
    help="If set, segmentation will not be enabled once video playback begins",
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
arg_video_path = args.video_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
use_square_sizing = not args.use_aspect_ratio
imgenc_base_size = args.base_size_px
max_memory_history = args.max_memories
max_pointer_history = args.max_pointers
discard_on_bad_objscore = args.discard_on_bad_objscore
segment_video_on_startup = not args.disable_segment_on_startup
show_info = not args.hide_info

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Create history to re-use selected inputs
root_path = osp.dirname(osp.dirname(__file__))
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_vidpath = history.read("video_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath)
video_path = ask_for_path_if_missing(arg_video_path, "video", history_vidpath)
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path, video_path=video_path, model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Get the model name, for reporting
model_name = osp.basename(model_path)

print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
assert sammodel.name in ("samv2", "samv3"), "Only SAMv2/v3 models are supported for video predictions!"
sammodel.to(**device_config_dict)

# Load image and get shaping info for providing display
full_image_bgr = cv2.imread(image_path)
if full_image_bgr is None:
    ok_video, full_image_bgr = run_video_frame_select_ui(image_path)
    if not ok_video:
        print("", "Unable to load image!", f"  @ {image_path}", sep="\n", flush=True)
        raise FileNotFoundError(osp.basename(image_path))


# ---------------------------------------------------------------------------------------------------------------------
# %% Run image encoder

# Set up shared image encoder settings (needs to be consistent across image/video frame encodings)
imgenc_config_dict = {"max_side_length": imgenc_base_size, "use_square_sizing": use_square_sizing}

# Run Model
print("", "Encoding image data...", sep="\n", flush=True)
t1 = perf_counter()
encoded_img, token_hw, preencode_img_hw = sammodel.encode_image(full_image_bgr, **imgenc_config_dict)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
time_taken_ms = round(1000 * (t2 - t1))
print(f"  -> Took {time_taken_ms} ms", flush=True)

# Run model without prompts as sanity check. Also gives initial result values
encoded_prompts = sammodel.encode_prompts([], [], [])
mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts)
prediction_hw = mask_preds.shape[2:]

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
    peak_vram_mb = torch.cuda.max_memory_allocated() // 1_000_000
    print("  VRAM:", peak_vram_mb, "MB")


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up the UI

# Set up shared UI elements & control logic
ui_elems = PromptUI(full_image_bgr, mask_preds)
uictrl = PromptUIControl(ui_elems)

# Set up message bars to communicate data info & controls
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(model_name, f"{token_hw_str} tokens", device_dtype_str, space_equally=True)

# Controls for adding/finishing
num_prompts_textblock = TitledTextBlock("Stored Memories").set_text(0)
record_prompt_btn = ImmediateButton("Record Prompt", color=(135, 115, 35))
track_video_btn = ImmediateButton("Track Video", color=(80, 140, 20))

# Set up full display layout
imgseg_layout = VStack(
    header_msgbar if show_info else None,
    ui_elems.layout,
    HStack(record_prompt_btn, num_prompts_textblock, track_video_btn),
).set_debug_name("DisplayLayout")

# Render out an image with a target size, to figure out which side we should limit when rendering
display_image = imgseg_layout.render(h=display_size_px, w=display_size_px)
render_side = "h" if display_image.shape[1] > display_image.shape[0] else "w"
render_limit_dict = {render_side: display_size_px}
min_display_size_px = imgseg_layout._rdr.limits.min_h if render_side == "h" else imgseg_layout._rdr.limits.min_w


# ---------------------------------------------------------------------------------------------------------------------
# %% Image segmentation

# Set up display
cv2.destroyAllWindows()
window = DisplayWindow("Image Segmentation - q to quit", display_fps=60).attach_mouse_callbacks(imgseg_layout)
window.move(200, 50)

# Change tools/masks on arrow keys
uictrl.attach_arrowkey_callbacks(window)

# Keypress for clearing prompts
window.attach_keypress_callback("c", ui_elems.tools.clear.click)

# For clarity, some additional keypress codes
KEY_ZOOM_IN = ord("=")
KEY_ZOOM_OUT = ord("-")

# Set up helper objects for managing display/mask data
base_img_maker = ReusableBaseImage(full_image_bgr)

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

# Allocate storage for SAM2 video masking
objbuffer = SAMVideoObjectResults.create(
    memory_history_length=max_memory_history,
    pointer_history_length=max_pointer_history,
    prompt_history_length=32,
)

# *** Image segmentation loop ***
try:
    while True:

        # Read prompt input data & selected mask
        need_prompt_encode, prompts = uictrl.read_prompts()
        is_mask_changed, mselect_idx, selected_mask_btn = ui_elems.masks_constraint.read()

        # Record existing prompts into a single memory
        if record_prompt_btn.read():

            # Produce prompt to be recorded for video segmentation
            _, init_mem, init_ptr = sammodel.initialize_video_masking(
                encoded_img, *prompts, mask_index_select=mselect_idx
            )
            objbuffer.store_prompt_result(0, init_mem, init_ptr)

            # Report the number of store prompt memories
            num_prompt_memories, _ = objbuffer.get_num_memories()
            num_prompts_textblock.set_text(num_prompt_memories)

            # Wipe out prompts, to indicate storage
            ui_elems.tools.clear.click()
            ui_elems.tools_constraint.change_to(ui_elems.tools.hover)

        # Close image segmentation to begin tracking
        if track_video_btn.read():
            break

        # Only run the model when an input affecting the output has changed!
        if need_prompt_encode:
            encoded_prompts = sammodel.encode_prompts(*prompts)
            mask_preds, iou_preds = sammodel.generate_masks(
                encoded_img, encoded_prompts, mask_hint=None, blank_promptless_output=True
            )

        # Update mask previews & selected mask for outlines
        need_mask_update = any((need_prompt_encode, is_mask_changed))
        if need_mask_update:
            selected_mask_uint8 = uictrl.create_hires_mask_uint8(mask_preds, mselect_idx, preencode_img_hw)
            uictrl.update_mask_previews(mask_preds)

        # Process contour data
        final_mask_uint8 = selected_mask_uint8
        _, mask_contours_norm = get_contours_from_mask(final_mask_uint8, normalize=True)

        # Re-generate display image at required display size
        # -> Not strictly needed, but can avoid constant re-sizing of base image (helpful for large images)
        display_hw = ui_elems.image.get_render_hw()
        disp_img = base_img_maker.regenerate(display_hw)

        # Update the main display image in the UI
        uictrl.update_main_display_image(disp_img, final_mask_uint8, mask_contours_norm)

        # Render final output
        display_image = imgseg_layout.render(**render_limit_dict)
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
    print("", "Closed with Ctrl+C", sep="\n")

except Exception as err:
    raise err

finally:
    cv2.destroyAllWindows()

# Sanity check. If the user didn't record
has_prompts = any(len(prompt_type) > 0 for prompt_type in prompts)
no_buffer = objbuffer.get_num_memories()[0] == 0
if no_buffer and has_prompts:
    print(
        "",
        "No prompt recorded, but found an active prompt. Will automatically store...",
        "-> To avoid this auto-storage, make sure that no prompts",
        "   (including hovering) are active before closing!",
        sep="\n",
        flush=True,
    )

    # Store encoding associated with last active prompt data
    _, init_mem, init_ptr = sammodel.initialize_video_masking(encoded_img, *prompts, mask_index_select=mselect_idx)
    objbuffer.store_prompt_result(0, init_mem, init_ptr)


# ---------------------------------------------------------------------------------------------------------------------
# %% Video segmentation

# Video frame reader + playback control UI
vreader = LoopingVideoReader(video_path)
playback_slider = LoopingVideoPlaybackSlider(vreader, stay_paused_on_change=False)
sample_frame = vreader.get_sample_frame()

# Setup initial model results
encoded_img, token_hw, preencode_img_hw = sammodel.encode_image(sample_frame, **imgenc_config_dict)
encoded_prompts = sammodel.encode_prompts([], [], [])
mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts)
prediction_hw = mask_preds.shape[2:]

# Set up simpler UI for video playback
ui_elems.enable_tools(False)
ui_elems.enable_masks(False)

# Set up text-based reporting UI
has_cuda = torch.cuda.is_available()
vram_text = ValueBlock("VRAM: ", "-", "MB", max_characters=5)
num_prompts_text = ValueBlock("Prompts: ", "0", max_characters=2)
num_history_text = ValueBlock("History: ", "0", max_characters=2)
objscore_text = ValueBlock("Score: ", None, max_characters=3)
text_header = HStack(vram_text, num_prompts_text, num_history_text, objscore_text)

# Define new UI
show_preview_btn, enable_segment_btn, enable_memory_btn = ToggleButton.many(
    "Preview", "Segment", "Store Memories", default_state=True, text_scale=0.5
)
show_preview_btn.toggle(False)
enable_segment_btn.toggle(segment_video_on_startup)
reset_memory_btn = ImmediateButton("Reset memory", color=(80, 20, 150), text_scale=0.5)
vidseg_layout = VStack(
    header_msgbar if show_info else None,
    text_header,
    ui_elems.display_block,
    HStack(show_preview_btn, enable_segment_btn, enable_memory_btn, reset_memory_btn),
    playback_slider,
)

# Set up display
cv2.destroyAllWindows()
window = DisplayWindow("Video Segmentation - q to quit", display_fps=1000 // vreader.get_frame_delay_ms())
window.attach_mouse_callbacks(vidseg_layout)
window.attach_keypress_callback(" ", vreader.toggle_pause)
window.attach_keypress_callback("p", show_preview_btn.toggle)
window.move(200, 50)

# Set up storage for keeping track of last encoded frame index
idx_keeper = ValueChangeTracker()
vram_report = PeriodicVRAMReport(update_period_ms=2000)

try:
    for is_paused, frame_idx, frame in vreader:

        # Read controls
        _, show_with_alpha = show_preview_btn.read()
        _, enable_segmentation = enable_segment_btn.read()
        _, enable_prevframe_storage = enable_memory_btn.read()

        # Wipe out per-frame history on button press
        if reset_memory_btn.read():
            objbuffer.prevframe_buffer.clear()
            idx_keeper.clear()

        # Update text feedback
        vram_usage_mb = vram_report.get_vram_usage()
        vram_text.set_value(vram_usage_mb)
        num_prompt_mems, num_prev_mems = objbuffer.get_num_memories()
        num_prompts_text.set_value(num_prompt_mems)
        num_history_text.set_value(num_prev_mems)

        # Disable segmentation while paused or when adjusting playback (avoid crippling cpu/gpu)
        is_new_frame = idx_keeper.is_changed(frame_idx)
        if is_new_frame:
            mask_contours_norm = None

        # Only run segmentation on unseen frames
        if enable_segmentation and is_new_frame and (not playback_slider.is_adjusting()):
            encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)
            obj_score, best_mask_idx, video_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
                encoded_imgs_list, **objbuffer.to_dict()
            )

            # Only store history for high-scoring predictions
            obj_score = float(obj_score.squeeze().float().cpu().numpy())
            if obj_score < 0 and discard_on_bad_objscore:
                video_preds = video_preds * 0.0
            elif enable_prevframe_storage:
                objbuffer.store_frame_result(frame_idx, mem_enc, obj_ptr)
            objscore_text.set_value(round(obj_score, 1))

            # Update the mask indicator to show which mask the model has chosen each frame
            best_mask_idx = int(best_mask_idx.squeeze().cpu())
            ui_elems.masks_constraint.change_to(best_mask_idx)
            uictrl.update_mask_previews(video_preds)

            # Process contour data
            selected_mask_uint8 = uictrl.create_hires_mask_uint8(video_preds, best_mask_idx, preencode_img_hw)
            ok_contours, mask_contours_norm = get_contours_from_mask(selected_mask_uint8, normalize=True)

            # Record the fact we worked on this frame
            idx_keeper.record(frame_idx)

        # Use checker background to suggest alpha channel if needed
        uictrl.update_main_display_image(frame, selected_mask_uint8, mask_contours_norm, show_with_alpha)

        # Display final image
        display_image = vidseg_layout.render(**render_limit_dict)
        req_break, keypress = window.show(display_image, 1 if enable_segmentation else None)
        if req_break:
            break

        # Scale display size up when pressing +/- keys
        if keypress == KEY_ZOOM_IN:
            display_size_px = min(display_size_px + 50, 10000)
            render_limit_dict = {render_side: display_size_px}
        if keypress == KEY_ZOOM_OUT:
            display_size_px = max(display_size_px - 50, min_display_size_px)
            render_limit_dict = {render_side: display_size_px}

        # Updates playback indicator & allows for adjusting playback
        playback_slider.update(frame_idx)

except KeyboardInterrupt:
    print("", "Closed with Ctrl+C", sep="\n")

except Exception as err:
    raise err

finally:
    cv2.destroyAllWindows()
    vreader.release()
