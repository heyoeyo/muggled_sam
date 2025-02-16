#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.cross_attention_transformer import CrossAttentionTransformer
from .components.shared import LayerNorm2d, Conv1x1

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV1MaskDecoder(nn.Module):
    """
    Simplified implementation of the 'mask-decoder' model/component described in:
        "Segment Anything"
        By: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
            Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
        @ https://arxiv.org/abs/2304.02643

    The code here is adapted from the original segment-anything repo:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py

    This model takes in image and prompt encodings and creates segmentation masks.

    In the original implementation, the model returned either 1 or 3 masks, was which
    meant to vary based on whether many prompts were given (return only 1 mask) or
    only 1 point or box prompt was given (return 3 masks). In this implementation,
    the model always returns all 4 masks, and it is left up to the end user to
    decide what to do with them.

    This implementation also includes a 'MaskHintEncoder' which is used to encode masks
    provided as inputs. In the original implementation, this component was part of
    the prompt-encoder model.
    """

    # .................................................................................................................

    def __init__(
        self,
        input_channels=256,
        downsample_dim=128,
        num_layers=2,
        num_heads=8,
        num_mask_tokens=4,
    ):

        # Inherit from parent
        super().__init__()

        # Set up model for handling input mask hints for improving segmentation results
        self.maskhint_encoder = MaskHintEncoder(input_channels)

        # Set up mask & iou token embeddings, which are combined with prompt embeddings as input into transformer
        self.cls_mask_tokens = nn.Parameter(torch.empty(num_mask_tokens, input_channels))
        self.cls_iou_token = nn.Parameter(torch.empty(1, input_channels))

        # Create transformer decoder
        self.transformer = CrossAttentionTransformer(num_layers, num_heads, input_channels, downsample_dim)

        # Create layers for generating final outputs
        self.maskgen = MaskGen(input_channels, num_mask_tokens)
        self.iou_token_mlp = MLP3Layers(input_channels, num_mask_tokens)

    # .................................................................................................................

    def forward(
        self,
        encoded_image_bchw: Tensor,
        encoded_prompts_bnc: Tensor,
        grid_positional_encoding: Tensor,
        mask_hint: Tensor | None = None,
        blank_promptless_output=True,
    ) -> tuple[Tensor, Tensor]:
        """
        Generates multiple candidate segmentation masks given an image encoding and
        encoded prompts (which specific the part of the image to segment).
        Also returns estimates for the 'quality' of each segmentation mask.

        The mask_hint input can be provided to help 'refine' the model output in some cases.
        It is expected to be of the same shape as a single mask output by the model.

        If 'blank_promptless_output' is true and no prompts are given, then a fully
        'blank' result will be returned instead of running the full decoder.

        Returns:
            mask_predictions, iou_predictions, encoded_cls_tokens
            -> Mask prediction has shape: Bx4xHxW, IoU has shape: Bx4, cls has shape: Bx5xF
            -> H & W are 4x the size of the image patch encoding size
            -> The 0-th mask was originally intended to be used in cases with
               many prompts, and not to be used with single points/box prompts
            -> The 'cls' tokens are included for matching SAMv2 outputs,
               they contain information about the iou & mask results but
               are not normally needed by an end user
        """

        # For clarity
        batch_size_prompts, num_prompts, enc_dim = encoded_prompts_bnc.shape

        # Special case, return blank masks if no prompt is given
        if blank_promptless_output and (num_prompts == 0) and (mask_hint is None):
            return self.maskgen.make_blank_results(encoded_image_bchw, batch_size_prompts)

        # Concatenate learned 'cls' tokens to prompts
        cls_tokens = torch.cat([self.cls_iou_token, self.cls_mask_tokens], dim=0).unsqueeze(0)
        cls_tokens = cls_tokens.expand(batch_size_prompts, -1, -1)
        num_cls_tokens = cls_tokens.shape[1]

        # Expand per-image data in batch direction to be per-mask, as well as the position encoding
        img_tokens_bchw = self.maskhint_encoder(encoded_image_bchw, mask_hint)
        img_tokens_bchw = torch.repeat_interleave(img_tokens_bchw, batch_size_prompts, dim=0)
        img_posenc_bchw = torch.repeat_interleave(grid_positional_encoding, batch_size_prompts, dim=0)

        # Cross-encode image tokens with prompt tokens
        prompt_tokens = torch.cat((cls_tokens, encoded_prompts_bnc), dim=1)
        prompt_tokens, img_tokens_bchw = self.transformer(prompt_tokens, img_tokens_bchw, img_posenc_bchw)

        # Extract the (now-encoded) 'cls' tokens by undoing the earlier cls concatenation step
        encoded_cls_tokens = prompt_tokens[:, :num_cls_tokens, :]
        iou_token_out = encoded_cls_tokens[:, 0, :]
        mask_tokens_out = encoded_cls_tokens[:, 1:, :]

        # Produce final output mask & quality predictions
        mask_preds = self.maskgen(img_tokens_bchw, mask_tokens_out)
        iou_preds = self.iou_token_mlp(iou_token_out)

        return mask_preds, iou_preds, encoded_cls_tokens

    # .................................................................................................................

    @staticmethod
    def get_best_mask_index(iou_predictions) -> int:
        """Helper used to select the index of the 'best' output, based on the highest IoU prediction score"""
        return int(iou_predictions.cpu().argmax())

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Model components


class MaskGen(nn.Module):
    """
    Sub-module of the SAM mask decoder which is responsible for generating the final output masks
    Performs upscaling of image embeddings (back to original input resolution) as well
    as a projection of encoder mask ('cls') tokens before dot-producting with image tokens
    to get final masks.

    The original implementation was directly part of the mask decoder model. Here, it's been split
    out into it's own module for clarity. For the original implementation details see:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L53-L65
    """

    # .................................................................................................................

    def __init__(self, input_channels, num_mask_tokens):

        # Inherit from parent
        super().__init__()

        hidden_channels = input_channels // 4
        upscaler_channels = input_channels // 8
        self.img_patch_upscaler = nn.Sequential(
            nn.ConvTranspose2d(input_channels, hidden_channels, kernel_size=2, stride=2),
            LayerNorm2d(hidden_channels),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_channels, upscaler_channels, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.mask_token_mlps = nn.ModuleList(
            [MLP3Layers(input_channels, upscaler_channels) for i in range(num_mask_tokens)]
        )

        # Create helper tensor for recording the model device/dtype information
        self.register_buffer("device_info", torch.empty(0), persistent=False)

    # .................................................................................................................

    def forward(self, image_tokens_bchw: Tensor, cls_mask_tokens: Tensor) -> Tensor:

        # Process/upscale image tokens to final output size
        upscaled_img_tokens = self.img_patch_upscaler(image_tokens_bchw)

        # Further encode cls mask tokens
        encoded_mask_tokens_list = [mlp(cls_mask_tokens[:, i, :]) for i, mlp in enumerate(self.mask_token_mlps)]
        encoded_mask_tokens = torch.stack(encoded_mask_tokens_list, dim=1)

        # Take dot product (along channel dimension) of cls tokens with image patches for final masks
        return torch.einsum("bnc, bchw -> bnhw", encoded_mask_tokens, upscaled_img_tokens)

    # .................................................................................................................

    def make_blank_results(self, image_tokens_bchw: Tensor, batch_size_prompts: int) -> tuple[Tensor, Tensor, Tensor]:
        """Helper used to generate a 'blank' mask, meant for cases where inputs aren't available"""

        # For clarity
        _, c, h, w = image_tokens_bchw.shape

        # Due to upscaler layer, the normal mask output should be 4 times larger than the image encoding size!
        mask_h, mask_w = (4 * h, 4 * w)
        mask_shape = (batch_size_prompts, 4, mask_h, mask_w)
        iou_shape = (batch_size_prompts, 4)
        cls_shape = (batch_size_prompts, 5, c)

        # Fill in empty mask and IoU prediction values
        device, dtype = self.device_info.device, self.device_info.dtype
        blank_mask_preds = torch.full(mask_shape, -100, device=device, dtype=dtype, requires_grad=False)
        blank_iou_preds = torch.ones(iou_shape, device=device, dtype=dtype, requires_grad=False)
        blank_cls_tokens = torch.zeros(cls_shape, device=device, dtype=dtype, requires_grad=False)

        return blank_mask_preds, blank_iou_preds, blank_cls_tokens

    # .................................................................................................................


class MaskHintEncoder(nn.Module):
    """
    Sub-module of the SAM mask decoder which is responsible for encoding input masks,
    which can act as a hint for where the output segmentation mask should be generated.
    If a hint isn't given, a learned 'no-mask' embedding will be used instead.

    This functionality was originally implemented as part of the prompt Encoder,
    but structurally it's simpler if implemented as a component of the decoder model,
    mostly because it needs access to the image token shape, whereas the prompt encoder doesn't.

    The original implementation details can be found here:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/prompt_encoder.py#L51-L60
    """

    # .................................................................................................................

    def __init__(self, output_channels=256):

        # Inherit from parent
        super().__init__()

        # Define a fall-back embedding to use when no mask hint is given
        self.no_mask_embed = nn.Parameter(torch.empty(1, output_channels))

        # For clarity. These values are hard-coded from original SAM implementation
        input_channels = 1
        num_hidden_ch_1 = 4
        num_hidden_ch_2 = 16
        self.downscaler = nn.Sequential(
            nn.Conv2d(input_channels, num_hidden_ch_1, kernel_size=2, stride=2),
            LayerNorm2d(num_hidden_ch_1),
            nn.GELU(),
            nn.Conv2d(num_hidden_ch_1, num_hidden_ch_2, kernel_size=2, stride=2),
            LayerNorm2d(num_hidden_ch_2),
            nn.GELU(),
            Conv1x1(num_hidden_ch_2, output_channels),
        )

        # Helper variable used to keep track of the model device/dtype (need for mask hints)
        self.register_buffer("device_info", torch.empty(1), persistent=False)

    # .................................................................................................................

    def forward(self, image_tokens_bchw: Tensor, mask_hint_bhw: Tensor | None) -> Tensor:
        """
        If a mask hint is provided, it will be encoded and added to the image tokens
        If a hint isn't given, then a learned 'no-mask' embedding is used instead.

        The mask hint should ideally be 4x the height & width of the patch grid,
        corresponding to the size of the outputs of the mask decoder itself
        -> This seems to be the original intended usage, to feed back output
           masks as prompts for refinement
        -> For example, with default (1024x1024px) sizing, the mask should be 256x256px

        Returns:
            hint_encoded_image_tokens_bchw
            -> Same shape as input image tokens (BxCxHxW)
        """

        # Create no-mask embedding if no hint is provided otherwise process mask to form encoded embedding
        grid_h, grid_w = image_tokens_bchw.shape[2:]
        if mask_hint_bhw is None:
            encoded_mask_hint = self.no_mask_embed.reshape(1, -1, 1, 1).expand(1, -1, grid_h, grid_w)

        else:
            # Add batch dimension to incoming mask, if needed
            if mask_hint_bhw.ndim == 2:
                mask_hint_bhw = mask_hint_bhw.unsqueeze(0)

            # Encode hint and scale to target size if needed
            encoded_mask_hint = self.downscaler(mask_hint_bhw.to(self.device_info))
            _, _, hint_h, hint_w = encoded_mask_hint.shape
            if hint_h != grid_h or hint_w != grid_w:
                encoded_mask_hint = nn.functional.interpolate(encoded_mask_hint, size=(grid_h, grid_w))

        return image_tokens_bchw + encoded_mask_hint

    # .................................................................................................................


class MLP3Layers(nn.Module):
    """
    Simplified implementation of the 3-layer MLP model used by the mask-decoder of SAM:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L154

    Simple 3-layer linear/feed-forward network using relu activations on layer 1 & 2,
    the final layer does not apply an activation function. All layers have dimensions equal to
    the input dimension, except the very last layer which is sized to the output dimension
    (i.e. dimension doesn't change through layers 1 & 2)
    """

    # .................................................................................................................

    def __init__(self, input_dim, output_dim):

        # Inherit from parent
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    # .................................................................................................................

    def forward(self, x):
        return self.layers(x)

    # .................................................................................................................
