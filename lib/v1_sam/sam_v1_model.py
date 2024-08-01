#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV1Model(nn.Module):
    """
    Wrapper around separated SAM model components, so that the model can be used as a singular entity
    """

    # .................................................................................................................

    def __init__(self, image_encoder_model, coordinate_encoder, prompt_encoder_model, mask_decoder_model):

        # Inherit from parent
        super().__init__()

        # Store SAM model components
        self.image_encoder = image_encoder_model
        self.coordinate_encoder = coordinate_encoder
        self.prompt_encoder = prompt_encoder_model
        self.mask_decoder = mask_decoder_model

        # Default to eval mode, expecting to use inference only
        self.eval()

    # .................................................................................................................

    def forward(self, image_rgb_normalized_bchw, boxes_tensor, fg_tensor, bg_tensor, mask_hint=None):
        """
        Main functionality of the SAM model, bundled into a single function.
        Takes an image and set of prompts and produces several candidate segmentation masks.

        Note that in practice, it makes more sense to call the component pieces of the model,
        rather than using this function so that image & prompt encoding can happen independently.
        See the 'encode_prompts', 'encode_image' and 'generate_masks' functions for more info

        Returns:
            mask_predictions, iou_predictions
        """

        # Encode prompts & image inputs
        box_posenc, fg_posenc, bg_posenc = self.coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
        encoded_prompts = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)
        encoded_image = self.image_encoder(image_rgb_normalized_bchw)

        # Combine encodings to generate mask output
        patch_grid_hw = encoded_image.shape[2:]
        grid_posenc = self.coordinate_encoder.get_full_grid_encoding(patch_grid_hw)
        mask_preds, iou_preds = self.mask_decoder(encoded_image, encoded_prompts, grid_posenc, mask_hint)

        return mask_preds, iou_preds

    # .................................................................................................................

    def encode_prompts(self, box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list):
        """
        Function used to encode prompt coordinates. Inputs should be given as lists
        of prompts. The length of each list does not need to match. Enter either
        None or an empty list ([]) to disable any of the prompts.

        Box prompt formatting:
            Each entry should be in top-left/bottom-right form: ((x1, y1), (x2, y2))
            For example:
                [
                    [(0.1, 0.5), (0.3, 0.7)], # Box 1
                    [(0.6, 0.2), (0.8, 0.4)], # Box 2
                    ... etc ...
                ]

        FG/BG prompt formatting:
            Each entry should be a single (x, y) point
            For example:
                [
                    (0.2, 0.6), # Point 1
                    (0.5, 0.4), # Point 2
                    (0.7, 0.7), # Point 3
                    (0.1, 0.9), # Point 4
                    ... etc ..
                ]

        Returns:
            encoded_prompts (shape: 1 x N x F, where N is number of prompt points, F is features per prompt)
        """

        with torch.inference_mode():
            boxes_tensor = self.coordinate_encoder.prepare_boxes(box_tlbr_norm_list)
            fg_tensor, bg_tensor = self.coordinate_encoder.prepare_points(fg_xy_norm_list, bg_xy_norm_list)
            box_posenc, fg_posenc, bg_posenc = self.coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
            encoded_prompts = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)

        return encoded_prompts

    # .................................................................................................................

    def encode_image(self, image_bgr, max_side_length=1024, window_size=None):

        with torch.inference_mode():
            image_rgb_normalized_bchw = self.image_encoder.prepare_image(image_bgr, max_side_length)
            image_preenc_hw = image_rgb_normalized_bchw.shape[2:]
            encoded_image = self.image_encoder(image_rgb_normalized_bchw, window_size)

        patch_grid_hw = encoded_image.shape[2:]
        return encoded_image, patch_grid_hw, image_preenc_hw

    # .................................................................................................................

    def generate_masks(self, encoded_image, encoded_prompts, mask_hint=None):

        with torch.inference_mode():
            patch_grid_hw = encoded_image.shape[2:]
            grid_posenc = self.coordinate_encoder.get_full_grid_encoding(patch_grid_hw)
            mask_preds, iou_preds = self.mask_decoder(encoded_image, encoded_prompts, grid_posenc, mask_hint)

        return mask_preds, iou_preds

    # .................................................................................................................

    def get_best_mask_index(self, iou_predictions):
        """Returns the index of the highest IoU prediction score"""
        return self.mask_decoder.get_best_mask_index(iou_predictions)

    # .................................................................................................................

    @staticmethod
    def normalize_xy(xy_px_list, frame_shape, use_original_SAM_method=True):
        """Helper used to normalize (x,y) pixel coordinates to the 0-to-1 range used by SAM"""

        # For convenience
        frame_h, frame_w = frame_shape[0:2]

        # Original method 'centers' pixel coordinates when normalizing, see:
        # https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/prompt_encoder.py#L80
        # https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/prompt_encoder.py#L95
        if use_original_SAM_method:
            x_scale, y_scale = np.float32(1 / frame_w), np.float32(1 / frame_h)
            return [((x + 0.5) * x_scale, (y + 0.5) * y_scale) for x, y in xy_px_list]

        # Natural way to normalize coords (imo), maps 0.0 to left/top-most pixel index, 1.0 to right/bottom most index
        x_scale, y_scale = np.float32(1 / (frame_w - 1)), np.float32(1 / (frame_h - 1))
        return [(x * x_scale, y * y_scale) for x, y in xy_px_list]
