#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

# Needed to make this script work from outside the root project folder (without requiring install)
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
from pathlib import Path
from time import perf_counter, sleep

import torch
import torch.nn as nn
import cv2
import numpy as np

from muggled_sam.make_sam import make_sam_from_state_dict

from muggled_sam.demo_helpers.ui.window import DisplayWindow, KEY
from muggled_sam.demo_helpers.ui.layout import HStack, VStack, OverlayStack
from muggled_sam.demo_helpers.ui.images import ExpandingImage
from muggled_sam.demo_helpers.ui.overlays import PointSelectOverlay, BoxSelectOverlay
from muggled_sam.demo_helpers.ui.buttons import ToggleButton, ImmediateButton, RadioConstraint
from muggled_sam.demo_helpers.ui.sliders import HSlider
from muggled_sam.demo_helpers.ui.text import ValueBlock, TextBlock
from muggled_sam.demo_helpers.ui.static import StaticMessageBar, HSeparator
from muggled_sam.demo_helpers.ui.plotting import LossPlot, ScoresPlot
from muggled_sam.demo_helpers.ui.base import force_same_min_width

from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing, load_init_prompts
from muggled_sam.demo_helpers.text_input import confirm_prompt
from muggled_sam.demo_helpers.mask_postprocessing import make_stacked_masks
from muggled_sam.demo_helpers.ui.helpers.images import get_image_hw_for_max_side_length
from muggled_sam.demo_helpers.misc import (
    get_default_device_string,
    make_device_config,
    get_total_cuda_vram_usage_mb,
)

import muggled_sam.demo_helpers.training.loss_functions as loss_funcs
from muggled_sam.demo_helpers.training.io import (
    TrainModulesDict,
    ImageLoader,
    load_prior_weights,
    make_save_folder,
    save_training_weights,
    load_text_list_file,
)
from muggled_sam.demo_helpers.training.layer_replacement import (
    LoraLinear,
    LoraConv2D,
    LoraConvTranspose2D,
    OffsetLayernorm,
    replace_target_modules,
    checkpoint_image_encoder_stages,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_prompts_path = None
default_teacher_path = None
default_student_path = None
default_continue_path = None
default_train_images_path = None
default_display_size = 800
default_base_size = 504
default_threshold = 0.5
default_lora_rank = 24
default_lr_low = 1e-7
default_lr_init = 1e-4
default_lr_high = 1e-1
default_max_duration = 120
default_loss_samples = 300

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to distill the SAMv3 image encoder")
parser.add_argument("-i", "--image_path", default=default_image_path, help="Path to test image")
parser.add_argument(
    "-p",
    "--prompts_path",
    default=default_prompts_path,
    type=str,
    help="Path to a json file containing prompts to use on start-up (format is: {'boxes': ..., 'fg_points': ...})",
)
parser.add_argument("--teacher_path", default=default_teacher_path, type=str, help="Path to teacher model weights")
parser.add_argument("--student_path", default=default_student_path, type=str, help="Path to student model weights")
parser.add_argument(
    "--training_images_path",
    default=default_train_images_path,
    type=str,
    help="Path to folder containing images for training",
)
parser.add_argument(
    "--continue_path", default=default_continue_path, type=str, help="Path to re-load prior training weights"
)
parser.add_argument(
    "-y",
    "--skip_cli",
    default=False,
    action="store_true",
    help="If set, cli prompts that have valid defaults will be used without prompting on start-up",
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
    help="Process images at their original aspect ratio",
)
parser.add_argument(
    "-b",
    "--base_size_px",
    default=default_base_size,
    type=int,
    help="Initial image processing size",
)
parser.add_argument("--stack_vertical", default=False, action="store_true", help="Stack displayed masks vertically")
parser.add_argument(
    "--no_convolution", default=False, action="store_true", help="Disable training of the convolution layers"
)
parser.add_argument("--no_linear", default=False, action="store_true", help="Disable training of linear layers")
parser.add_argument("--enable_layernorm", default=False, action="store_true", help="Enable training of layernorms")
parser.add_argument(
    "-r",
    "--lora_rank",
    default=default_lora_rank,
    type=int,
    help=f"LoRA rank used for training (default: {default_lora_rank})",
)
parser.add_argument(
    "--lr_init", default=default_lr_init, type=float, help=f"Initial learning rate (default: {default_lr_init:.1e})"
)
parser.add_argument(
    "--lr_high",
    default=default_lr_high,
    type=float,
    help=f"Highest allowed learning rate (default: {default_lr_high:.1e})",
)
parser.add_argument(
    "--lr_low", default=default_lr_low, type=float, help=f"Lowest allowed learning rate (default: {default_lr_low:.1e})"
)
parser.add_argument(
    "--max_duration", default=default_max_duration, type=int, help="Set maximum training duration (seconds)"
)
parser.add_argument(
    "--num_loss_samples",
    default=default_loss_samples,
    type=int,
    help=f"Max number of loss plot values to show (default: {default_loss_samples})",
)
parser.add_argument(
    "--low_memory", default=False, action="store_true", help="Enable activation checkpointing to reduce memory usage"
)

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
init_prompts_path = args.prompts_path
arg_teacher_path = args.teacher_path
arg_student_path = args.student_path
arg_train_images_path = args.training_images_path
arg_continue_path = args.continue_path
arg_skip_cli = args.skip_cli
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
use_square_sizing = not args.use_aspect_ratio
imgenc_init_size = args.base_size_px
is_masks_vert_stack = args.stack_vertical
enable_convolution_training = not args.no_convolution
enable_linear_training = not args.no_linear
enable_layernorm_training = args.enable_layernorm
lora_rank = max(args.lora_rank, 1)
lr_low = args.lr_low
lr_high = args.lr_high
lr_init = args.lr_init
duration_max_sec = max(1, int(args.max_duration))
num_loss_samples = max(5, args.num_loss_samples)
enable_checkpointing = args.low_memory

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)
is_using_cuda = "cuda" in device_config_dict["device"]

# Force checkpointing if we're likely to run out of VRAM
if is_using_cuda and not enable_checkpointing:
    _, total_mem_bytes = torch.cuda.mem_get_info()
    total_mem_gb = total_mem_bytes / 1e9
    if total_mem_gb < (24 if use_float32 else 12):
        enable_checkpointing = True
        print("", "Low-memory mode has been auto-enabled to reduce VRAM usage", sep="\n")

# Create history to re-use selected inputs
root_path = Path(__file__).parent.parent
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_trainpath = history.read("train_image_folder_path")
_, history_studentpath = history.read("distill_student_path")
_, history_teacherpath = history.read("distill_teacher_path")

# Skip prompts with history if possible
if arg_skip_cli:
    arg_image_path = history_imgpath if arg_image_path is None else arg_image_path
    arg_train_images_path = history_trainpath if arg_train_images_path is None else arg_train_images_path
    arg_student_path = history_studentpath if arg_student_path is None else arg_student_path
    arg_teacher_path = history_teacherpath if arg_teacher_path is None else arg_teacher_path

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath)
path_train_images = ask_for_path_if_missing(arg_train_images_path, "training images", history_trainpath)
path_student_model = ask_for_model_path_if_missing(
    root_path, arg_student_path, history_studentpath, message="Select student model:"
)
path_teacher_model = ask_for_model_path_if_missing(
    root_path, arg_teacher_path, history_teacherpath, message="Select teacher model:"
)

# Sanity check. Make sure we're not using the same model for student and teacher
if path_student_model == path_teacher_model:
    raise NameError(
        "Cannot use the same model as both the teacher and student! Please create a pruned model to use as the student"
    )

# Store history for use on reload
history.store(
    image_path=image_path,
    train_image_folder_path=path_train_images,
    distill_student_path=path_student_model,
    distill_teacher_path=path_teacher_model,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load model resources

# Load student model
name_student = Path(path_student_model).name
print("", "Loading student model...", f"@ {path_student_model}", sep="\n")
config_student, model_student = make_sam_from_state_dict(path_student_model)
assert model_student.name == "samv3", "Only SAMv3 is supported for fine tuning"

# Load up teacher model
name_teacher = Path(path_teacher_model).name
print("", "Loading teacher model...", f"@ {path_teacher_model}", sep="\n")
config_teacher, model_teacher = make_sam_from_state_dict(path_teacher_model)
assert model_teacher.name == "samv3", "Only SAMv3 is supported for fine tuning"

# Remove unused components to save some memory
to_delete = ["sampling_encoder", "image_exemplar_fusion", "text_encoder", "exemplar_detector", "exemplar_segmentation"]
for component_name in to_delete:
    if hasattr(model_student, component_name):
        delattr(model_student, component_name)
    if hasattr(model_teacher, component_name):
        delattr(model_teacher, component_name)

# Make sure all weights are un-trainable to begin
for p in model_teacher.parameters():
    p.requires_grad_(False)
for p in model_teacher.parameters():
    p.requires_grad_(False)
model_teacher.eval()
model_student.train()

# Move models to target devices & report sizing
model_teacher.to(**device_config_dict)
model_student.to(**device_config_dict)
if is_using_cuda:
    torch.cuda.empty_cache()

# Set up save pathing
save_folder_path = make_save_folder(root_path, path_student_model, "distill_imgenc")

# Helps avoid issues mixing inf-mode tensors with training tensors
model_student.toggle_inference_mode(False)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load training data

# Force relative path to be relative to this script
path_train_images = Path(path_train_images)
if path_train_images.parent == Path("."):
    path_train_images = Path(__file__).parent / path_train_images

print("", "Loading image paths for training...", f"@ {path_train_images}", sep="\n")
if path_train_images.is_dir():
    # If given a folder, assume all file paths inside are training images
    all_train_image_paths_list = []
    for parent_folder_path, _, files_list in path_train_images.walk():
        for file in files_list:
            if file.endswith("txt") or file.endswith("json"):
                continue
            all_train_image_paths_list.append(parent_folder_path / file)
    assert len(all_train_image_paths_list) > 0, f"Error, no images found in folder: {path_train_images}"
else:
    # Assume we got a path to a text file listing image paths directly
    all_train_image_paths_list = []
    for img_path in load_text_list_file(path_train_images):
        img_path = Path(img_path.removeprefix("'").removeprefix('"').removesuffix("'").removesuffix('"'))
        if not img_path.exists():
            continue
        all_train_image_paths_list.append(img_path)
print(f"  -> Found {len(all_train_image_paths_list)} files")
assert len(all_train_image_paths_list) > 0, "No image paths found, nothing to train on!"
img_loader = ImageLoader(all_train_image_paths_list)

# Also load Load test image and get smaller version for display (reduce resizing burden during updates)
loaded_image_bgr = cv2.imread(image_path)
scale_hw = get_image_hw_for_max_side_length(loaded_image_bgr)
scaled_img_bgr = cv2.resize(loaded_image_bgr, dsize=(scale_hw[1], scale_hw[0]))


# ---------------------------------------------------------------------------------------------------------------------
# %% Attach training layers

# Initialize references to trainable parameters
training_modules = TrainModulesDict()
num_linear_params = 0
num_layernorm_params = 0
num_conv2d_params = 0
num_proj_conv2d_params = 0
num_convtranspose_params = 0
IMGENC_SUBMOD_STR = "image_encoder"
IMGPRJ_SUBMOD_STR = "image_projection"

# Train linear layers
if enable_linear_training:
    linear_refs, num_linear_params = replace_target_modules(
        model_student, IMGENC_SUBMOD_STR, nn.Linear, lambda layer: LoraLinear(layer, lora_rank)
    )
    training_modules.store_training_modules("lora_linear", linear_refs)

# Train convolution layers (these exist on both the encoder and projection components!)
if enable_convolution_training:
    enc_conv2d_refs, num_conv2d_params = replace_target_modules(
        model_student, IMGENC_SUBMOD_STR, nn.Conv2d, lambda layer: LoraConv2D(layer, lora_rank)
    )
    proj_conv2d_refs, num_proj_conv2d_params = replace_target_modules(
        model_student, IMGPRJ_SUBMOD_STR, nn.Conv2d, lambda layer: LoraConv2D(layer, lora_rank)
    )
    proj_convtranspose_refs, num_convtranspose_params = replace_target_modules(
        model_student, IMGPRJ_SUBMOD_STR, nn.ConvTranspose2d, lambda layer: LoraConvTranspose2D(layer, lora_rank)
    )

    training_modules.store_training_modules("lora_conv2d", enc_conv2d_refs)
    training_modules.store_training_modules("lora_conv2d", proj_conv2d_refs, replace=False)
    training_modules.store_training_modules("lora_convtranspose2d", proj_convtranspose_refs, replace=False)

# Train layernorms
if enable_layernorm_training:
    layernorm_refs, num_layernorm_params = replace_target_modules(
        model_student, IMGENC_SUBMOD_STR, nn.LayerNorm, OffsetLayernorm
    )
    training_modules.store_training_modules("offset_layernorm", layernorm_refs)

# Load existing weights, if needed
prior_weights_dict = None
need_load_weights = arg_continue_path is not None
if need_load_weights:
    print("", "Loading prior training weights...", f"@ {arg_continue_path}", sep="\n")
    need_module_resizing = load_prior_weights(
        arg_continue_path,
        name_student,
        training_modules,
    )
    if need_module_resizing:
        print("Resizing needed to load layer data (likely a lora rank mismatch)")
    prior_weights_dict = training_modules.record_state_dict()

# Get block mappings
num_blocks_student = config_student["imgencoder_num_blocks"]
num_blocks_teacher = config_teacher["imgencoder_num_blocks"]

# Report training parameter count
num_student_params = sum(p.numel() for p in model_student.image_encoder.parameters() if p.requires_grad)
num_teacher_params = sum(p.numel() for p in model_teacher.image_encoder.parameters())
pct_train_param = round(100 * num_student_params / num_teacher_params, 1)
report_lora_rank = f"{lora_rank if (num_conv2d_params + num_linear_params) > 0 else 'not used'}"
if need_load_weights:
    report_lora_rank = f"{report_lora_rank} (may be altered by loaded weights)"
print(
    "",
    "Training stats:",
    f" Total param count: {num_student_params} from {num_teacher_params} ({pct_train_param}%)",
    f"     Linear params: {num_linear_params}",
    f"     Conv2D params: {num_conv2d_params + num_proj_conv2d_params}",
    f"   Conv2D.T params: {num_convtranspose_params}",
    f"  Layernorm params: {num_layernorm_params}",
    f"       Block count: {num_blocks_student} from {num_blocks_teacher}",
    f"         Lora rank: {report_lora_rank}",
    sep="\n",
    flush=True,
)
sleep(0.5)

# Warn if student is same/bigger than teacher
if num_blocks_student >= num_blocks_teacher:
    print(
        "",
        "WARNING:",
        "  The student model is not smaller than the teacher model!",
        "  Training may not work properly...",
        sep="\n",
        flush=True,
    )
    sleep(5)

# Set up checkpointing if needed
if enable_checkpointing:
    print("", "Checkpointing model to reduce memory usage...", sep="\n")
    checkpoint_image_encoder_stages(model_student)

# Set up initial 'backup' weights, just so we don't get errors if user tries to restore
backup_weights_dict = {ltype: {} for ltype in training_modules.get_layer_types()}
last_backup_name, last_backup_iters = None, 0
make_backup_name = lambda loss, name, iters: f"{loss:.2e} ({name}) : {iters}"


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up the UI

# Set up message bars to communicate config info
model_device = device_config_dict["device"]
model_dtype = str(device_config_dict["dtype"]).split(".")[-1]
blocks_str = f"{num_blocks_teacher} -> {num_blocks_student} blocks"
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(blocks_str, device_dtype_str, space_equally=True)

# Set up main displays
main_img_elem = ExpandingImage(loaded_image_bgr)
mask_img_elem = ExpandingImage(np.zeros_like(loaded_image_bgr))
point_select_olay = PointSelectOverlay(color=(0, 255, 0))
box_select_olay = BoxSelectOverlay(thickness=2)
olay_elem = OverlayStack(main_img_elem, point_select_olay, box_select_olay)
plot_loss_elem = LossPlot("Loss", min_side_length=256)
plot_scores_elem = ScoresPlot("IoU Scores", bar_width_pct=(90, 60), use_log_scale=True, min_side_length=256)

# Disable box input on start up
point_select_olay.enable(True)
box_select_olay.enable(False)

# Set backup controls
backup_btn = ImmediateButton("Backup", (60, 120, 150), button_height=30, text_scale=0.35)
restore_btn = ImmediateButton("Restore", (150, 120, 60), button_height=30, text_scale=0.35)
last_backup_name_block = TextBlock("no backup", block_height=20)
force_same_min_width(restore_btn, backup_btn)

# Set up weight controls
reset_btn = ImmediateButton("Reset", (60, 20, 170))
save_to_disk_btn = ImmediateButton("Save", (60, 170, 20))
force_same_min_width(reset_btn, save_to_disk_btn)

# Set up loss function selector
loss_functions_list = [
    ("MSE", loss_funcs.mse_loss),
    ("L1", loss_funcs.l1_loss),
    ("Angle", loss_funcs.angle_loss),
    ("Scale", loss_funcs.scale_loss),
    ("Polar", loss_funcs.polar_loss),
]
loss_btns_list = ToggleButton.many(*[name for name, _ in loss_functions_list], button_height=35, text_scale=0.5)
radio_loss = RadioConstraint(*loss_btns_list)
radio_loss.change_to(2)

# Elements used to report training state
vram_text_block = ValueBlock("VRAM: ", "-", "MB", max_characters=5, outline_color=None)
iters_txt_block = ValueBlock("Iter: ", "0", max_characters=6, outline_color=None)
loss_txt_block = ValueBlock("Loss: ", "-", max_characters=6, outline_color=None)
lr_txt_block = ValueBlock("LR: ", "-", max_characters=6, outline_color=None)
force_same_min_width(iters_txt_block, vram_text_block, lr_txt_block, loss_txt_block)

# Interpret learning rate config
lr_low_log10, lr_init_log10, lr_high_log10 = [np.log10(val) for val in sorted([lr_low, lr_init, lr_high])]
lr_init_pct = float(round(100 * (lr_init_log10 - lr_low_log10) / (lr_high_log10 - lr_low_log10)))
update_lr_from_pct = lambda lr_pct: float(10 ** (((lr_pct / 100.0) * (lr_high_log10 - lr_low_log10)) + lr_low_log10))

# Set up slider controls
duration_max_sec = 120
train_btn = ToggleButton("Train", on_color=(0, 80, 220), off_color=(60, 75, 100))
duration_slider = HSlider("Train duration (s)", 20, 0, duration_max_sec, step_size=1, marker_steps=15)
lr_slider = HSlider("Learning rate", lr_init_pct, 0, 100, step_size=1, marker_steps=10)
accum_slider = HSlider("Gradient accumulation", 8, 1, 128, step_size=1, marker_steps=16)
disp_n_slider = HSlider("Display updates (every N)", 1, 0, 8, step_size=1, marker_steps=2)
imgsize_slider = HSlider(
    "Process Resolution (px)", imgenc_init_size, 336, 1176, step_size=24, marker_steps=7, bar_bg_color=(55, 50, 55)
)
force_same_min_width(duration_slider, train_btn, disp_n_slider, min_w=100)

# Set up full display layout
disp_layout = VStack(
    header_msgbar,
    HStack(*radio_loss),
    imgsize_slider,
    HStack(
        VStack(olay_elem, mask_img_elem) if is_masks_vert_stack else HStack(olay_elem, mask_img_elem),
        VStack(plot_loss_elem, plot_scores_elem, HStack(backup_btn, restore_btn), last_backup_name_block),
    ),
    HStack(iters_txt_block, vram_text_block if is_using_cuda else None, lr_txt_block, loss_txt_block),
    HStack(duration_slider, train_btn, disp_n_slider),
    lr_slider,
    accum_slider,
    HStack(reset_btn, HSeparator(40), save_to_disk_btn),
)

# Render out an image with a target size, to figure out which side we should limit when rendering
display_image = disp_layout.render(h=display_size_px, w=display_size_px)
render_limit_dict = {"h": display_size_px}
min_display_size_px = min(600, disp_layout._rdr.limits.min_h)

# Create hidden controls (easier to bind to keypresses this way)
hidden_box_prompt_mode_btn = ToggleButton("Box prompt mode", default_state=False)
hidden_optim_btn = ImmediateButton("Reset optimizer")
hidden_mask_stack_btn = ToggleButton("Use vertical masks", default_state=False)
mask_rc_list = [(1, 4), (2, 2)] if is_masks_vert_stack else [(4, 1), (2, 2)]
num_mask_rowcol = mask_rc_list[0]

# Load initial prompts if provided
have_init_prompts, init_prompts_dict = load_init_prompts(init_prompts_path)
if have_init_prompts:
    box_select_olay.add_boxes(*init_prompts_dict.get("boxes", []))
    point_select_olay.add_points(*init_prompts_dict.get("fg_points", []))
    if len(init_prompts_dict.get("bg_points", [])) > 0:
        print(
            "",
            "Warning:",
            "  Loaded prompt contains background points, which are not supported in this script",
            "These will be ignored...",
            sep="\n",
        )


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Display loop ***


def run_full(model: nn.Module, image_tensor_bchw: torch.Tensor):
    """Helper used to run full image encoding for training"""
    encoded_img = model.image_encoder(image_tensor_bchw)
    v3_features_list = model.image_projection.v2_projection(encoded_img)
    v2_features_list = model.image_projection.v3_projection(encoded_img)
    return [*v2_features_list, *v3_features_list]


def save_weights(
    save_folder_path: str | Path,
    modules_to_save: TrainModulesDict,
    name_teacher: str,
    name_student: str,
    config_student: dict,
    total_iterations: int,
):
    """Helper used to save training weights"""
    save_weights_dict = modules_to_save.record_state_dict()
    save_data_dict = {"weights": save_weights_dict}
    save_data_dict["teacher_name"] = name_teacher
    save_data_dict["student_name"] = name_student
    save_data_dict["student_config"] = config_student
    save_path, save_size_mb = save_training_weights(save_folder_path, save_data_dict, total_iterations)
    print("", f"Saved training weights: ({save_size_mb:.1f} MB)", f"@ {save_path}", "", sep="\n", flush=True)


# Set up display
cv2.destroyAllWindows()
window = DisplayWindow("Display - q to quit", display_fps=30).attach_mouse_callbacks(disp_layout)
window.move(200, 50)

# Keypress controls
window.attach_keypress_callback(KEY.SPACEBAR, train_btn.toggle)
window.attach_keypress_callback(KEY.LEFT_ARROW, radio_loss.previous)
window.attach_keypress_callback(KEY.RIGHT_ARROW, radio_loss.next)
window.attach_keypress_callback("[", lr_slider.decrement)
window.attach_keypress_callback("]", lr_slider.increment)
window.attach_keypress_callback("b", backup_btn.click)
window.attach_keypress_callback("s", save_to_disk_btn.click)
window.attach_keypress_callback("m", hidden_mask_stack_btn.toggle)
window.attach_keypress_callback("r", hidden_optim_btn.click)
window.attach_keypress_callback(KEY.TAB, hidden_box_prompt_mode_btn.toggle)

# For clarity, some additional keypress codes
KEY_ZOOM_IN = ord("=")
KEY_ZOOM_OUT = ord("-")

# Some feedback
print(
    "",
    "Note:",
    "  The training objective is to match the encoded image tokens of the student to the teacher model.",
    "  The mask and IoU predictions are not directly optimized and can even degrade as loss improves!",
    "",
    "- Use spacebar to start/stop training",
    "- Try different loss functions from the top bar",
    "- Try adjusting the learning rate to see how it affects the loss",
    "- Use backup/restore to keep copies of weights before making extreme changes to settings",
    "- Use reset to wipe out training progress",
    "- Click on the image to adjust the point prompt used for testing",
    "",
    "Mask prediction coloring:",
    "- Purple areas are false negatives (predicted by teacher, not by student)",
    "- Red areas are false positives (predicted by student, not by teacher)",
    "- Green areas are true positives (predicted by teacher and student",
    "",
    "Keypress controls:",
    "Left/right arrows: Change selected loss function",
    "[ or ]: Adjust learning rate",
    "b: Create a backup",
    "s: Save the current training weights to disk",
    "m: Toggle mask stacking",
    "r: Manually reset optimizer state",
    "TAB: Toggle between point/box prompting (use right-click to delete prompts before switching)",
    "",
    "Use -/+ keys to change display sizing",
    "Press q or esc to close the window",
    "",
    sep="\n",
    flush=True,
)

# Initialize training loop data/state
remake_optimizer = lambda lr: torch.optim.AdamW((p for p in model_student.parameters() if p.requires_grad), lr=lr)
need_optim_reset = True
is_nan_loss = False
time_train_start_sec = 0
total_iters = 0
optim_step_count = 0
avg_loss = None
loss_name, loss_func = loss_functions_list[0]
plot_loss_list = []
optim = remake_optimizer(0)
request_display_only_update = True
need_save_on_crash = False
loss_scaling = [1.0, (1 / 2), (1 / 2), 1.0, (1 / 2), (1 / 2)]
imgenc_config_dict = {"max_side_length": imgenc_init_size, "use_square_sizing": use_square_sizing}
prompts_dict = {"box_tlbr_norm_list": [], "fg_xy_norm_list": [], "bg_xy_norm_list": []}

# Force ui updates on start
radio_loss.set_is_changed(True)
lr_slider.set_is_changed(True)
imgsize_slider.set_is_changed(True)

# Add an initial prompt if needed
_, point_xy_norm_list = point_select_olay.read()
_, box_xy1xy2_norm_list = box_select_olay.read()
if len(point_xy_norm_list) == 0 and len(box_xy1xy2_norm_list) == 0:
    point_select_olay.add_points((0.5, 0.5))

# *** Main display loop ***
try:
    while True:
        # Read buttons
        need_reset_weights = reset_btn.read()
        need_restore_weights = restore_btn.read()
        need_backup_weights = backup_btn.read()
        need_save_to_disk = save_to_disk_btn.read()
        need_manual_optim_reset = hidden_optim_btn.read()
        is_prompt_mode_changed, use_box_prompt = hidden_box_prompt_mode_btn.read()
        is_train_changed, is_training = train_btn.read()
        is_loss_changed, loss_select_idx, loss_func_btn = radio_loss.read()
        is_mask_stacking_changed, is_stack_vertical = hidden_mask_stack_btn.read()

        # Read sliders
        is_imgsize_changed, imgsize_px = imgsize_slider.read()
        is_duration_changed, duration_sec = duration_slider.read()
        is_lr_changed, learning_rate_pct = lr_slider.read()
        is_accum_changed, accum_after_n = accum_slider.read()
        is_disprate_changed, display_every_n = disp_n_slider.read()

        # Read prompt inputs
        if is_prompt_mode_changed:
            print(f"Enabling {'box' if use_box_prompt else 'point'} prompts")
            point_select_olay.enable(not use_box_prompt)
            box_select_olay.enable(use_box_prompt)
        is_point_prompt_changed, point_xy_norm_list = point_select_olay.read()
        is_box_prompt_changed, box_xy1xy2_norm_list = box_select_olay.read()

        # Force re-prediction from teacher, so we have new data for comparisons
        is_prompt_changed = is_point_prompt_changed or is_box_prompt_changed
        if is_imgsize_changed or is_prompt_changed:
            imgenc_config_dict = {"max_side_length": imgsize_px, "use_square_sizing": use_square_sizing}
            prompts_dict["fg_xy_norm_list"] = point_xy_norm_list
            prompts_dict["box_tlbr_norm_list"] = box_xy1xy2_norm_list

            # Re-predict teacher results
            true_encimg, _, _ = model_teacher.encode_image(loaded_image_bgr, **imgenc_config_dict)
            true_encpmt = model_teacher.encode_prompts(**prompts_dict)
            true_mask_preds, true_iou_scores = model_teacher.generate_masks(true_encimg, true_encpmt)
            plot_scores_elem.set_true_data(true_iou_scores.squeeze(0).float().cpu().numpy())
            request_display_only_update = not is_training

        # Force display update when switching mask stacking
        if is_mask_stacking_changed:
            num_mask_rowcol = mask_rc_list[int(is_stack_vertical)]
            request_display_only_update |= not is_training

        # Switch loss functions if needed and force display update to show new loss
        if is_loss_changed:
            avg_loss = None
            request_display_only_update = True
            loss_name, loss_func = loss_functions_list[loss_select_idx]
            plot_loss_list = []

        # Clear all training weights
        if need_reset_weights:
            if prior_weights_dict is None:
                training_modules.reset_all_weights()
            else:
                training_modules.load_state_dict(prior_weights_dict)
            need_optim_reset = True

        # Record current weights
        if need_backup_weights:
            backup_weights_dict = training_modules.record_state_dict()
            last_backup_name = make_backup_name(avg_loss, loss_name, total_iters)
            last_backup_iters = total_iters
            last_backup_name_block.set_text(last_backup_name)
            request_display_only_update = True
            print("  Backup -", last_backup_name)

        # Load previously stored weights
        if need_restore_weights:
            training_modules.load_state_dict(backup_weights_dict)
            total_iters = last_backup_iters
            need_optim_reset = True
            print("Restored -", last_backup_name)

        # Reset optimizer & training state
        if need_optim_reset or need_manual_optim_reset:
            if need_manual_optim_reset:
                print("Manually reset optimizer!")
            need_optim_reset = False
            lr = update_lr_from_pct(learning_rate_pct)
            optim = remake_optimizer(lr)
            if is_using_cuda:
                torch.cuda.empty_cache()

            # Trigger train update to report new loss/detection results
            total_iters = 0 if not need_restore_weights else last_backup_iters
            avg_loss = None
            plot_loss_list = []
            request_display_only_update = True

        # Update learning rate if needed
        if is_lr_changed or need_reset_weights:
            lr = update_lr_from_pct(learning_rate_pct)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
            optim.zero_grad()
            lr_txt_block.set_value(f"{lr:.1e}")

        # Running training updates
        if is_training or request_display_only_update:

            # Check if we should stop training by elapsed time
            if is_train_changed:
                time_train_start_sec = perf_counter() if duration_sec < duration_max_sec else 1e9
            elapsed_time = perf_counter() - time_train_start_sec
            is_last_example = elapsed_time > duration_sec

            # Run example through teacher & student model (or re-use teacher cache)
            next_img_uint8 = img_loader.get_next_image(request_display_only_update)
            with torch.no_grad():
                img_tensor = model_teacher.image_encoder.prepare_image(next_img_uint8, **imgenc_config_dict)
                out_target = run_full(model_teacher, img_tensor)
            out_pred = run_full(model_student, img_tensor)

            # Calculate loss (prediction have shape: BxNxC -> B batch, N num tokens, C channels)
            all_losses = [loss_func(targ, pred, channel_dim=1) for targ, pred in zip(out_target, out_pred)]
            loss_train = sum(scale * lval for scale, lval in zip(loss_scaling, all_losses))
            loss_item = loss_train.item()
            avg_loss = loss_item if avg_loss is None else (avg_loss * 0.9 + loss_item * 0.1)
            plot_loss_list.append(avg_loss)
            is_nan_loss = np.isnan(loss_item)
            if is_nan_loss:
                plot_loss_list = []
                train_btn.toggle(False)
                print("Got NaN loss! Reset weights to continue...")

            # Run training updates
            if not request_display_only_update:
                (loss_train / accum_after_n).backward()
                optim.step()
                total_iters += 1
                optim_step_count += 1

                # Stop accumulating gradients
                if is_last_example or optim_step_count >= accum_after_n:
                    optim.zero_grad()
                    optim_step_count = 0

            # Update display
            need_display_update = (display_every_n > 0) and (total_iters % display_every_n) == 0
            if need_display_update or is_last_example or request_display_only_update:

                # Get mask predictions using updated image encoder
                with torch.no_grad():
                    # This no_grad block is needed because we disabled the built-in inference_mode earlier
                    # The reason for disabling inference_mode is because we may need to re-gen cached
                    # internal data (e.g. position encodings) as user changes inference settings, but
                    # we get an error if the cache is re-generated inside of inference_mode, hence no_grad!
                    # (mainly a problem due to image size/aspect ratio changes)
                    test_encimg, _, _ = model_student.encode_image(loaded_image_bgr, **imgenc_config_dict)
                    test_encpmt = model_student.encode_prompts(**prompts_dict)
                    test_mask_preds, test_iou_preds = model_student.generate_masks(test_encimg, test_encpmt)

                    # Draw mask predictions showing matching pixels and false positives/negatives
                    test_mask_img = make_stacked_masks(test_mask_preds[0] > 0, num_mask_rowcol).cpu().numpy()
                    true_mask_img = make_stacked_masks(true_mask_preds[0] > 0, num_mask_rowcol).cpu().numpy()
                    mask_img = np.zeros((*true_mask_img.shape, 3), dtype=np.uint8)
                    match_mask_img = np.bitwise_and(test_mask_img, true_mask_img)
                    false_pos_img = np.bitwise_and(test_mask_img, ~true_mask_img)
                    false_neg_img = np.bitwise_and(~test_mask_img, true_mask_img)
                    mask_img[false_neg_img > 0] = (250, 10, 110)
                    mask_img[false_pos_img > 0] = (70, 40, 255)
                    mask_img[match_mask_img > 0] = (80, 255, 0)

                    # Get plottable IoU scores
                    test_iou_np = test_iou_preds[0].float().cpu().numpy()

                # Update displayed image & plot data
                mask_img_elem.set_image(mask_img)
                plot_loss_elem.set_plot_data(plot_loss_list[-num_loss_samples:], 0)
                plot_scores_elem.set_test_data(test_iou_np)

                # Report loss & iterations
                loss_txt_block.set_value(f"{avg_loss:.2e}")
                iters_txt_block.set_value(total_iters)

            # Stop training
            if is_last_example or request_display_only_update:
                train_btn.toggle(False)
                vram_text_block.set_value(get_total_cuda_vram_usage_mb() if is_using_cuda else "-")
            request_display_only_update = False

        # Render final output
        display_image = disp_layout.render(**render_limit_dict)
        req_break, keypress = window.show(display_image, frame_delay_ms=1 if is_training else None)
        if req_break:
            break

        # Scale display size up when pressing +/- keys
        if keypress == KEY_ZOOM_IN:
            display_size_px = min(display_size_px + 50, 10000)
            render_limit_dict = {"h": display_size_px}
        if keypress == KEY_ZOOM_OUT:
            display_size_px = max(display_size_px - 50, min_display_size_px)
            render_limit_dict = {"h": display_size_px}

        # Save current training weights
        if need_save_to_disk:
            save_weights(
                save_folder_path,
                training_modules,
                name_teacher,
                name_student,
                config_student,
                total_iters,
            )

        pass

except KeyboardInterrupt:
    print("")

except torch.OutOfMemoryError as err:
    need_save_on_crash = True
    print("", "Error - Out of memory!", "Consider enabling low memory mode (see script flags)...", "", sep="\n")
    sleep(1)
    raise err

except Exception as err:
    need_save_on_crash = True
    print("", "An unexpected error occurred!", "", sep="\n", flush=True)
    raise err

finally:
    cv2.destroyAllWindows()

    if need_save_on_crash:
        user_confirm = confirm_prompt("Save current training weights?", is_yes_by_default=True)
        if user_confirm:
            save_weights(
                save_folder_path,
                training_modules,
                name_teacher,
                name_student,
                config_student,
                total_iters,
            )
