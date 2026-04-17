#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import cv2
import torch
from muggled_sam.make_sam import make_sam_from_state_dict

# Define pathing
image_path = "/path/to/image.jpg"
model_path = "/path/to/sam3.pt"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Define inference config
text_prompts_list = ["face", "arm", "hand", "invisible object", "leg", "foot", "ground"]
max_side_length = 1008
use_square_sizing = True
detection_score_threshold = 0.5

# Load image
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Error loading image from: {image_path}")

# Load and set up detector model
print("Loading model...", flush=True)
model_config_dict, full_model = make_sam_from_state_dict(model_path)
full_model.to(device=device, dtype=dtype)
detmodel = full_model.make_detector_model()

# Run detection
print("Running detections...", flush=True)
encoded_imgs, _, _ = detmodel.encode_detection_image(img_bgr, max_side_length, use_square_sizing)
exemplars_list = [detmodel.encode_exemplars(encoded_imgs, txt) for txt in text_prompts_list]
exemplar_batch, exemplar_padding_mask = detmodel.make_exemplar_batch(*exemplars_list)
mask_preds_bnhw, box_preds_bn22, detection_scores_bn, presence_scores_b = detmodel.generate_detections(
    encoded_imgs,
    encoded_exemplars_bnc=exemplar_batch,
    exemplar_padding_mask_bn=exemplar_padding_mask,
)
print("Raw mask prediction shape:", tuple(mask_preds_bnhw.shape))
print("Exemplar batch shape:", tuple(exemplar_batch.shape))
print("Padding mask shape:", tuple(exemplar_padding_mask.shape))

# Display results per-batch entry
batch_size = mask_preds_bnhw.shape[0]
for b_idx, txt in enumerate(text_prompts_list):

    # Filter out 'good' detections (model always produces 200 guesses, we keep only the ones with high scores)
    # -> Cannot be done as a batch operation, since results are usually different shapes (i.e. different 'N')
    mask_nhw, boxes_n22, scores_n22, presence_score = detmodel.filter_results(
        mask_preds_bnhw[b_idx],
        box_preds_bn22[b_idx],
        detection_scores_bn[b_idx],
        presence_scores_b[b_idx],
        score_threshold=detection_score_threshold,
    )

    # Some feedback + skip display if we don't have any masks
    num_results = mask_nhw.shape[0]
    print(f"Found {num_results:>3} results for: {txt}")
    if num_results == 0:
        continue

    # Mask out sections of the original image for display
    scaled_mask = torch.nn.functional.interpolate(mask_nhw.unsqueeze(0), img_bgr.shape[0:2], mode="bilinear")
    mask_uint8 = (scaled_mask[0] > 0).any(dim=0).byte().cpu().numpy()
    mask_uint8 = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
    new_img = cv2.copyTo(img_bgr, mask_uint8)
    cv2.putText(new_img, f"Prompt: {txt}", (5, 28), 0, 1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Mask results", new_img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
