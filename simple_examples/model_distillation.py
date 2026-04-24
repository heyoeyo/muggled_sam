#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import muggled_sam  # NOQA
except ModuleNotFoundError:
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "muggled_sam" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to muggled_sam folder!")
from datetime import datetime as dt
import cv2
import torch
from muggled_sam.make_sam import make_sam_from_state_dict

# Define pathing
training_images_folder_path = "/path/to/training/images/folder"
student_model_path = "/path/to/student.pth"
teacher_model_path = "/path/to/teacher.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Training config
imgenc_config_dict = {"max_side_length": 336, "use_square_sizing": True}
num_epochs = 2
learning_rate = 5e-5

# Set up training data
print("Reading training data")
train_image_paths_list = []
for parent_folder_path, sub_folders, files_list in os.walk(training_images_folder_path):
    train_image_paths_list.extend([os.path.join(parent_folder_path, filename) for filename in files_list])
print(f"-> Found {len(train_image_paths_list)} files")
assert len(train_image_paths_list) > 0, "No training data found!"

# Set up student
print("Loading student model...")
_, model_student = make_sam_from_state_dict(student_model_path)
model_student.to(device=device, dtype=dtype)
model_student.toggle_inference_mode(False)
model_student.train()

# Set up teacher
print("Loading teacher model...")
_, model_teacher = make_sam_from_state_dict(teacher_model_path)
model_teacher.to(device=device, dtype=dtype)
model_teacher.toggle_inference_mode(False)
model_teacher.eval()

# Sanity check. Start with all weights being un-trainable
for p in model_student.parameters():
    p.requires_grad_(False)
for p in model_teacher.parameters():
    p.requires_grad_(False)

# Enable training of only desired components
components_to_train_list = [model_student.image_encoder]
for component in components_to_train_list:
    for p in component.parameters():
        p.requires_grad_(True)

# *** Training loop ***
loss_func = torch.nn.MSELoss()
optim = torch.optim.AdamW((p for p in model_student.parameters() if p.requires_grad), lr=learning_rate)
for epoch_idx in range(num_epochs):
    epoch_loss, iters_epoch = 0, 0
    print("", "*" * 32, f"Epoch: {epoch_idx}", sep="\n", flush=True)
    for train_path in train_image_paths_list:

        # Load image for processing
        img_uint8 = cv2.imread(train_path)
        is_ok_img = img_uint8 is not None
        if not is_ok_img:
            print("Skipping bad image path:", train_path)
            continue

        # Run teacher to produce ground-truth and run student for corresponding prediction
        with torch.no_grad():
            ground_truth, _, _ = model_teacher.encode_image(img_uint8, **imgenc_config_dict)
        prediction, _, _ = model_student.encode_image(img_uint8, **imgenc_config_dict)

        # Flatten outputs (v3 models output list of lists of tensors)
        if not isinstance(prediction[0], torch.Tensor):
            flat_truth, flat_pred = [], []
            for targ, pred in zip(ground_truth, prediction):
                for targ_item, pred_item in zip(targ, pred):
                    flat_truth.append(targ_item)
                    flat_pred.append(pred_item)
            ground_truth, prediction = flat_truth, flat_pred

        # Update student weights
        loss = sum(loss_func(targ, pred) for targ, pred in zip(ground_truth, prediction))
        loss.backward()
        optim.step()
        optim.zero_grad()

        # Keep track of running average loss
        iters_epoch += 1
        epoch_loss += loss.item()
        if iters_epoch % 10 == 0:
            print(f"E{epoch_idx} | {iters_epoch:>4} | Avg loss: {epoch_loss / max(1, iters_epoch): .3e}")

    # Some feedback
    print(f"Finished! Avg loss: {epoch_loss / max(1, iters_epoch): .3e}", sep="\n")

# Save trained model weights
print("", "", sep="\n")
if "n" not in input("Save result? [Y/n] ").lower():
    timestamp = round(dt.now().timestamp())
    save_path = os.path.join(os.path.dirname(student_model_path), f"simple_distillation_{timestamp}.pt")
    torch.save(model_student.state_dict(), save_path)
    print("Saved training result:", save_path, sep="\n")
