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
from muggled_sam.demo_helpers.ui.overlays import DrawBoxOverlay
from muggled_sam.demo_helpers.ui.buttons import ToggleButton, ImmediateButton, RadioConstraint
from muggled_sam.demo_helpers.ui.sliders import HSlider
from muggled_sam.demo_helpers.ui.text import ValueBlock, TextBlock
from muggled_sam.demo_helpers.ui.static import StaticMessageBar, HSeparator
from muggled_sam.demo_helpers.ui.plotting import LossPlot, ScoresPlot
from muggled_sam.demo_helpers.ui.base import force_same_min_width

from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import (
    ask_for_path_if_missing,
    ask_for_model_path_if_missing,
    ask_for_value_if_missing,
)
from muggled_sam.demo_helpers.text_input import read_user_text_input, confirm_prompt
from muggled_sam.demo_helpers.mask_postprocessing import draw_combined_mask
from muggled_sam.demo_helpers.misc import (
    get_default_device_string,
    make_device_config,
    get_total_cuda_vram_usage_mb,
)

import muggled_sam.demo_helpers.training.loss_functions as loss_funcs
from muggled_sam.demo_helpers.training.default_data import make_default_training_text_list, save_unnested_json
from muggled_sam.demo_helpers.training.io import (
    TrainModulesDict,
    ShuffleList,
    load_prior_weights,
    make_save_folder,
    save_training_weights,
    load_text_list_file,
)
from muggled_sam.demo_helpers.training.layer_replacement import (
    LoraLinear,
    LoraEmbedding,
    OffsetLayernorm,
    replace_target_modules,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_text_prompt = None
default_teacher_path = None
default_student_path = None
default_continue_path = None
default_train_text_path = "distill_training_text.json"
default_display_size = 800
default_threshold = 0.5
default_lora_rank = 32
default_lr_low = 1e-7
default_lr_init = 1e-4
default_lr_high = 1e-1
default_max_duration = 60
default_loss_samples = 600

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to distill the SAMv3 text encoder")
parser.add_argument("-i", "--image_path", default=default_image_path, help="Path to test image")
parser.add_argument("-t", "--text_prompt", default=default_text_prompt, help="Text prompt for testing")
parser.add_argument("--teacher_path", default=default_teacher_path, type=str, help="Path to teacher model weights")
parser.add_argument("--student_path", default=default_student_path, type=str, help="Path to student model weights")
parser.add_argument(
    "--continue_path", default=default_continue_path, type=str, help="Path to re-load prior training weights"
)
parser.add_argument(
    "--training_text_path",
    default=default_train_text_path,
    type=str,
    help="Path to file containing text prompts for training",
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
    "--detection_threshold",
    default=default_threshold,
    type=float,
    help=f"Detection threshold used for reporting results (default: {default_threshold})",
)
parser.add_argument(
    "--no_embedding", default=False, action="store_true", help="Disable training of the text embeddings"
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

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_text_prompt = args.text_prompt
arg_teacher_path = args.teacher_path
arg_student_path = args.student_path
arg_continue_path = args.continue_path
arg_train_text_path = args.training_text_path
arg_skip_cli = args.skip_cli
arg_display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
detection_threshold = args.detection_threshold
enable_embedding_training = not args.no_embedding
enable_linear_training = not args.no_linear
enable_layernorm_training = args.enable_layernorm
lora_rank = max(args.lora_rank, 1)
lr_low = args.lr_low
lr_high = args.lr_high
lr_init = args.lr_init
duration_max_sec = max(1, int(args.max_duration))
num_loss_samples = max(5, args.num_loss_samples)

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)
is_using_cuda = "cuda" in device_config_dict["device"]

# Create history to re-use selected inputs
root_path = Path(__file__).parent.parent
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_txtprompt = history.read("distill_txt_prompt")
_, history_studentpath = history.read("distill_student_path")
_, history_teacherpath = history.read("distill_teacher_path")

# Use history values to fill in cli prompts
if arg_skip_cli:
    arg_image_path = history_imgpath if arg_image_path is None else arg_image_path
    arg_text_prompt = history_txtprompt if arg_text_prompt is None else arg_text_prompt
    arg_student_path = history_studentpath if arg_student_path is None else arg_student_path
    arg_teacher_path = history_teacherpath if arg_teacher_path is None else arg_teacher_path

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath)
text_prompt = ask_for_value_if_missing(arg_text_prompt, "Enter text prompt: ", history_txtprompt)
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
    distill_txt_prompt=text_prompt,
    distill_student_path=path_student_model,
    distill_teacher_path=path_teacher_model,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load model resources

# Load student model
name_student = Path(path_student_model).name
print("", "Loading student model...", f"@ {path_student_model}", sep="\n")
config_student, base_model_student = make_sam_from_state_dict(path_student_model)
assert base_model_student.name == "samv3", "Only SAMv3 is supported for fine tuning"
model_student = base_model_student.make_detector_model()

# Load up teacher model
name_teacher = Path(path_teacher_model).name
print("", "Loading teacher model...", f"@ {path_teacher_model}", sep="\n")
config_teacher, base_model_teacher = make_sam_from_state_dict(path_teacher_model)
assert base_model_teacher.name == "samv3", "Only SAMv3 is supported for fine tuning"
model_teacher = base_model_teacher.make_detector_model()

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

# Set up save pathing
save_folder_path = make_save_folder(root_path, path_student_model, "distill_txtenc")


# ---------------------------------------------------------------------------------------------------------------------
# %% Load test image

# Load test image
loaded_image_bgr = cv2.imread(image_path)

# Run 'true' image encoding & mask generation for reporting
exemplar_prompt = {"text": text_prompt}
true_encimg, _, _ = model_teacher.encode_detection_image(loaded_image_bgr)
true_encexm = model_teacher.encode_exemplars(true_encimg, **exemplar_prompt)
_, _, true_score_preds, _ = model_teacher.generate_detections(true_encimg, true_encexm)

# Figure out how many detections to display based on 'true' result
num_true_detections = (true_score_preds > detection_threshold).count_nonzero()
use_top_n = num_true_detections if num_true_detections > 0 else 5

# Delete unusued components save memory
del base_model_student
del base_model_teacher
del model_teacher.image_encoder
del model_teacher.image_projection
del model_student.image_encoder
del model_student.image_projection
if is_using_cuda:
    torch.cuda.empty_cache()

# Helps avoid issues mixing inf-mode tensors with training tensors
model_student.toggle_inference_mode(False)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load training data

# Force relative path to be relative to this script
arg_train_text_path = Path(arg_train_text_path)
if arg_train_text_path.parent == Path("."):
    arg_train_text_path = Path(__file__).parent / arg_train_text_path

# For convenience, check for a .txt version of the given training text file
if not arg_train_text_path.exists() and arg_train_text_path.with_suffix(".txt").exists():
    arg_train_text_path = arg_train_text_path.with_suffix(".txt")

# Create training text file if we don't find one
if not arg_train_text_path.exists():
    default_train_text = make_default_training_text_list()
    save_unnested_json(arg_train_text_path, default_train_text)
    print(
        "",
        "Training text file not found! Creating default file:",
        f"@ {arg_train_text_path}",
        "Add/remove text prompts from this file to guide the model distillation!",
        "Including more training prompts will lead to better generalization, but will be harder to train.",
        sep="\n",
    )

# Load training text data
print("", "Loading text for training...", f"@ {arg_train_text_path.name}", sep="\n")
all_train_text_list = load_text_list_file(arg_train_text_path)
all_train_text_list = ShuffleList(all_train_text_list, shuffle_on_init=True)

# Convert all input text to vocabulary indexing ahead-of-time
with torch.no_grad():
    teacher_tokenizer = model_teacher.text_encoder.tokenizer
    txt_to_idx_cache = {txt: teacher_tokenizer.text_to_vocab_index(txt) for txt in all_train_text_list}
    num_unique_txt = len(txt_to_idx_cache)
    example_txt = ", ".join(list(txt_to_idx_cache.keys())[0:5])
    txt_cache_kb = 8 * sum(idx_tensor.shape[-1] for idx_tensor in txt_to_idx_cache.values()) / 1000
print(f"  -> Found {num_unique_txt} unique text entries (cached {txt_cache_kb:.1f} KB of token indices)")
print("  -> Examples:", example_txt, "...")


# ---------------------------------------------------------------------------------------------------------------------
# %% Attach training layers

# Initialize references to trainable parameters
training_modules = TrainModulesDict()
num_linear_params = 0
num_embedding_params = 0
num_layernorm_params = 0
TXTENC_SUBMOD_STR = "text_encoder"

# Train linear layers
if enable_linear_training:
    linear_refs, num_linear_params = replace_target_modules(
        model_student,
        TXTENC_SUBMOD_STR,
        nn.Linear,
        lambda layer: LoraLinear(layer, lora_rank),
    )
    training_modules.store_training_modules("lora_linear", linear_refs)

# Train text embedding
if enable_embedding_training:
    embedding_refs, num_embedding_params = replace_target_modules(
        model_student,
        TXTENC_SUBMOD_STR,
        nn.Embedding,
        lambda layer: LoraEmbedding(layer, lora_rank),
    )
    training_modules.store_training_modules("lora_embedding", embedding_refs)

# Train layernorms
if enable_layernorm_training:
    layernorm_refs, num_layernorm_params = replace_target_modules(
        model_student, TXTENC_SUBMOD_STR, nn.LayerNorm, OffsetLayernorm
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
        print("*** Resizing was needed to load layer data (likely a lora rank mismatch)")
    prior_weights_dict = training_modules.record_state_dict()

# Get block mappings
num_layers_student = config_student["txtencoder_num_blocks"]
num_layers_teacher = config_teacher["txtencoder_num_blocks"]

# Report training parameter count
num_student_params = sum(p.numel() for p in model_student.text_encoder.parameters() if p.requires_grad)
num_teacher_params = sum(p.numel() for p in model_teacher.text_encoder.parameters())
pct_train_param = round(100 * num_student_params / num_teacher_params, 1)
report_lora_rank = f"{lora_rank if (num_embedding_params + num_linear_params) > 0 else 'not used'}"
if need_load_weights:
    report_lora_rank = f"{report_lora_rank} (may be altered by loaded weights)"
print(
    "",
    "Training stats:",
    f" Total param count: {num_student_params} from {num_teacher_params} ({pct_train_param}%)",
    f"  Embedding params: {num_embedding_params}",
    f"     Linear params: {num_linear_params}",
    f"  Layernorm params: {num_layernorm_params}",
    f"       Layer count: {num_layers_student} from {num_layers_teacher}",
    f"         Lora rank: {report_lora_rank}",
    sep="\n",
    flush=True,
)
sleep(0.5)

# Warn if student is same/bigger than teacher
if num_layers_student >= num_layers_teacher:
    print(
        "",
        "WARNING:",
        "  The student model is not smaller than the teacher model!",
        "  Training may not work properly...",
        sep="\n",
        flush=True,
    )
    sleep(5)

# Set up initial 'backup' weights, just so we don't get errors if user tries to restore
backup_weights_dict = {ltype: {} for ltype in training_modules.get_layer_types()}
last_backup_name, last_backup_iters = None, 0
make_backup_name = lambda loss, name, iters: f"{loss:.2e} ({name}) : {iters}"


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up the UI

# Set up message bars to communicate config info
model_device = device_config_dict["device"]
model_dtype = str(device_config_dict["dtype"]).split(".")[-1]
layers_str = f"{num_layers_teacher} -> {num_layers_student} layers"
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(layers_str, device_dtype_str, space_equally=True)

# Set up main displays
main_img_elem = ExpandingImage(loaded_image_bgr)
bounding_box_olay = DrawBoxOverlay()
olay_elem = OverlayStack(main_img_elem, bounding_box_olay)

# Set plots & backup controls
plot_loss_elem = LossPlot("Loss", min_side_length=128)
plot_scores_elem = ScoresPlot("Detection Scores", use_log_scale=True, min_side_length=128)
backup_btn = ImmediateButton("Backup", (60, 120, 150), button_height=30, text_scale=0.35)
restore_btn = ImmediateButton("Restore", (150, 120, 60), button_height=30, text_scale=0.35)
last_backup_name_block = TextBlock("no backup", block_height=20)
force_same_min_width(restore_btn, backup_btn, min_w=110)

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
train_btn = ToggleButton("Train", on_color=(0, 80, 220), off_color=(60, 75, 100))
duration_slider = HSlider("Train duration (s)", 30, 0, duration_max_sec, step_size=1, marker_steps=10)
lr_slider = HSlider("Learning rate", lr_init_pct, 0, 100, step_size=1, marker_steps=10)
accum_slider = HSlider("Gradient accumulation", 32, 1, 128, step_size=1, marker_steps=16)
disp_n_slider = HSlider("Display updates (every N)", 1, 0, 8, step_size=1, marker_steps=2)
force_same_min_width(duration_slider, train_btn, disp_n_slider)

# Set up full display layout
disp_layout = VStack(
    header_msgbar,
    HStack(*radio_loss),
    HStack(
        olay_elem, VStack(plot_loss_elem, plot_scores_elem, HStack(backup_btn, restore_btn), last_backup_name_block)
    ),
    HStack(iters_txt_block, vram_text_block if is_using_cuda else None, lr_txt_block, loss_txt_block),
    HStack(duration_slider, train_btn, disp_n_slider),
    lr_slider,
    accum_slider,
    HStack(reset_btn, HSeparator(40), save_to_disk_btn),
)

# Render out an image with a target size, to figure out which side we should limit when rendering
display_size_px = arg_display_size_px
display_image = disp_layout.render(h=display_size_px, w=display_size_px)
render_limit_dict = {"h": display_size_px}
min_display_size_px = min(400, disp_layout._rdr.limits.min_h, arg_display_size_px)

# Create hidden controls (easier to bind to keypresses this way)
hidden_text_input = ImmediateButton("Enter prompt")
hidden_show_mask_btn = ToggleButton("Show masks")
hidden_list_scores_btn = ImmediateButton("Blame")
hidden_optim_btn = ImmediateButton("Reset optimizer")


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Display loop ***


def run_full(model: nn.Module, idx_tokens_bn: torch.Tensor) -> torch.Tensor:
    """Helper used to run full text encoding for training"""
    encoded_text_bnc = model.text_encoder.vocab_embeddings(idx_tokens_bn)
    encoded_text_bnc = model.text_encoder.transformer(encoded_text_bnc)
    encoded_text_bnc = model.text_encoder.out_proj(encoded_text_bnc)
    return encoded_text_bnc


def save_weights(
    save_folder_path: Path,
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
target_fps = 30
frame_delay_sec_target = (1.0 / target_fps) * 0.75
cv2.destroyAllWindows()
window = DisplayWindow("Display - q to quit", display_fps=target_fps).attach_mouse_callbacks(disp_layout)
window.move(200, 50)

# Keypress controls
window.attach_keypress_callback(KEY.ENTER, hidden_text_input.click)
window.attach_keypress_callback(KEY.SPACEBAR, train_btn.toggle)
window.attach_keypress_callback(KEY.LEFT_ARROW, radio_loss.previous)
window.attach_keypress_callback(KEY.RIGHT_ARROW, radio_loss.next)
window.attach_keypress_callback("[", lr_slider.decrement)
window.attach_keypress_callback("]", lr_slider.increment)
window.attach_keypress_callback("b", backup_btn.click)
window.attach_keypress_callback("s", save_to_disk_btn.click)
window.attach_keypress_callback("m", hidden_show_mask_btn.toggle)
window.attach_keypress_callback("r", hidden_optim_btn.click)
window.attach_keypress_callback("l", hidden_list_scores_btn.click)

# For clarity, some additional keypress codes
KEY_ZOOM_IN = ord("=")
KEY_ZOOM_OUT = ord("-")

# Some feedback
print(
    "",
    "Note:",
    "  The training objective is to match the encoded tokens of the student to the teacher model.",
    "  The box, mask and score predictions are not directly optimized and can even degrade as loss improves!",
    "",
    "- Use spacebar to start/stop training",
    "- Try different loss functions from the top bar",
    "- Try adjusting the learning rate to see how it affects the loss",
    "- Use backup/restore to keep copies of weights before making extreme changes to settings",
    "- Use reset to wipe out training progress",
    "- Press enter to return to the terminal to enter a new text prompt",
    "",
    "Keypress controls:",
    "Left/right arrows: Change selected loss function",
    "[ or ]: Adjust learning rate",
    "b: Create a backup",
    "s: Save the current training weights to disk",
    "m: Toggle mask prediction view",
    "r: Manually reset optimizer state",
    "l: Print out top-20 worst scoring text prompts",
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

# Load true prediction scores for plotting comparison
plot_scores_elem.set_true_data(true_score_preds.squeeze(0).float().cpu().numpy())

# Force ui updates on start
radio_loss.set_is_changed(True)
lr_slider.set_is_changed(True)

# *** Main display loop ***
try:
    while True:
        # Read buttons
        need_new_prompt = hidden_text_input.read()
        need_reset_weights = reset_btn.read()
        need_restore_weights = restore_btn.read()
        need_backup_weights = backup_btn.read()
        need_save_to_disk = save_to_disk_btn.read()
        need_manual_optim_reset = hidden_optim_btn.read()
        is_train_changed, is_training = train_btn.read()
        is_loss_changed, loss_select_idx, loss_func_btn = radio_loss.read()
        is_show_masks_changed, show_masks = hidden_show_mask_btn.read()

        # Read sliders
        is_duration_changed, duration_sec = duration_slider.read()
        is_lr_changed, learning_rate_pct = lr_slider.read()
        is_accum_changed, accum_after_n = accum_slider.read()
        is_disprate_changed, display_every_n = disp_n_slider.read()

        # Switch to terminal to enter new text prompt
        if need_new_prompt:
            _, _, new_prompt = read_user_text_input(prompt_for_exiting=None, allow_numeric_inputs=False)
            exemplar_prompt = {"text": new_prompt}

            # Update 'true' predictions for comparisons
            true_encexm = model_teacher.encode_exemplars(true_encimg, **exemplar_prompt)
            _, _, true_score_preds, _ = model_teacher.generate_detections(true_encimg, true_encexm)
            num_true_detections = (true_score_preds > detection_threshold).count_nonzero()
            use_top_n = num_true_detections if num_true_detections > 0 else 5
            plot_scores_elem.set_true_data(true_score_preds.squeeze(0).float().cpu().numpy())

            request_display_only_update |= not is_training

        # Force display update to switch mask vs. image display
        if is_show_masks_changed:
            request_display_only_update |= not is_training

        # Switch loss functions if needed and force display update to show new loss
        if is_loss_changed:
            avg_loss = None
            loss_name, loss_func = loss_functions_list[loss_select_idx]
            plot_loss_list = []
            request_display_only_update = True

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

            # On start of training, re-sample data
            if is_train_changed:
                time_train_start_sec = perf_counter() if duration_sec < duration_max_sec else 1e9

            # Run a 'inter-display' loop. This allows for multiple model runs in-between display updates
            # -> Useful for when the model can run much faster than the display update rate
            iter_start_time = perf_counter()
            for inter_idx in range(1 if request_display_only_update else 50):

                # Stop inter-display updates if next display update is ready
                iter_time = perf_counter()
                if (iter_time - iter_start_time) > frame_delay_sec_target:
                    break

                # Check if we should stop training by elapsed time
                elapsed_time = iter_time - time_train_start_sec
                is_last_example = elapsed_time > duration_sec

                # Run example through teacher & student model
                _, next_txt_prompt = all_train_text_list.get_next(request_display_only_update)
                with torch.no_grad():
                    input_vocab_index = txt_to_idx_cache.get(next_txt_prompt, None)
                    if input_vocab_index is None:
                        input_vocab_index = teacher_tokenizer.text_to_vocab_index(next_txt_prompt)
                    out_target = run_full(model_teacher, input_vocab_index)
                out_pred = run_full(model_student, input_vocab_index)

                # Calculate loss (prediction have shape: BxNxC -> B batch, N num tokens, C channels)
                loss_train = loss_func(out_target, out_pred, channel_dim=-1)
                loss_item = loss_train.item()
                avg_loss = loss_item if avg_loss is None else (avg_loss * 0.9 + loss_item * 0.1)
                plot_loss_list.append(avg_loss)
                is_nan_loss = np.isnan(loss_item)
                if is_nan_loss:
                    plot_loss_list = []
                    train_btn.toggle(False)
                    print("Got NaN loss! Reset weights to continue...")
                    break

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

                # Stop inter-display loop if we hit our last example
                if is_last_example:
                    break

            # Update display
            need_display_update = (display_every_n > 0) and (total_iters % display_every_n) == 0
            if need_display_update or is_last_example or request_display_only_update:

                # Report loss & iterations
                loss_txt_block.set_value(f"{avg_loss:.2e}")
                iters_txt_block.set_value(total_iters)

                with torch.no_grad():

                    # Get detections from student model
                    test_encexm = model_student.encode_exemplars(true_encimg, **exemplar_prompt)
                    test_mask_preds, test_box_preds, test_score_preds, _ = model_student.generate_detections(
                        true_encimg, test_encexm
                    )

                    # Figure out how many valid detections we have for display
                    test_score_preds = test_score_preds[0].float().cpu()
                    is_valid_detection = (test_score_preds > detection_threshold).cpu()
                    num_valid_detections = is_valid_detection.count_nonzero()

                    # Set up display filtering + special handling for NaN states
                    has_detections = num_valid_detections > 0
                    test_sorted_idx = None if has_detections else test_score_preds.sort(descending=True)[-1]
                    filter_idx = is_valid_detection if has_detections else test_sorted_idx[0:use_top_n]
                    if is_nan_loss:
                        filter_idx = torch.zeros_like(is_valid_detection)
                        test_score_preds = torch.zeros_like(test_score_preds)
                        num_valid_detections = 0

                    # Generate mask or image + box display
                    disp_img = loaded_image_bgr
                    if show_masks:
                        test_mask_preds = test_mask_preds[0].float().cpu()
                        filtered_masks = test_mask_preds[filter_idx]
                        disp_img = draw_combined_mask(filtered_masks, main_img_elem.get_render_hw())
                        bounding_box_olay.clear()
                    else:
                        test_box_preds = test_box_preds[0].float().cpu()
                        filtered_boxes = test_box_preds[filter_idx]
                        bounding_box_olay.style(color=(0, 255, 0) if num_valid_detections > 0 else (0, 200, 255))
                        bounding_box_olay.set_boxes(filtered_boxes.float().cpu().numpy())

                    # Update displayed image & plot data
                    main_img_elem.set_image(disp_img)
                    plot_loss_elem.set_plot_data(plot_loss_list[-num_loss_samples:], 0)
                    plot_scores_elem.set_test_data(test_score_preds.numpy())

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

        # Print out scores for every text entry to see what's causing the model the most trouble
        if hidden_list_scores_btn.read():
            print("", f"Computing scores... ({loss_name})", sep="\n", flush=True)
            all_scores_list = []
            with torch.no_grad():
                for txt in all_train_text_list:
                    input_vocab_index = txt_to_idx_cache[txt]
                    out_target = run_full(model_teacher, input_vocab_index)
                    out_pred = run_full(model_student, input_vocab_index)
                    loss_score = loss_func(out_target, out_pred, channel_dim=-1).item()
                    all_scores_list.append((loss_score, txt))
            for blame_score, txt in sorted(all_scores_list, reverse=True)[:20]:
                print(f"{blame_score:.2e} - {txt}")
            print("")

        # Save current training weights
        if need_save_to_disk:
            save_weights(save_folder_path, training_modules, name_teacher, name_student, config_student, total_iters)

        pass

except KeyboardInterrupt:
    print("")

except torch.OutOfMemoryError as err:
    need_save_on_crash = True
    print("", "Error - Out of memory!", "", sep="\n")
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
            save_weights(save_folder_path, training_modules, name_teacher, name_student, config_student, total_iters)
