#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.cross_attention_transformer import CrossAttentionTransformer
from .components.shared import LayerNorm2d

# For type hints
from torch import Tensor

# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV2MaskDecoder(nn.Module):
    """
    Simplified implementation of the 'mask-decoder' model/component originally described in:
        "Segment Anything"
        By: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
            Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
        @ https://arxiv.org/abs/2304.02643

    This version is updated to support SAMV2:
        "SAM 2: Segment Anything in Images and Videos"
        By: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma,
        Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan,
        Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
        Piotr Dollár, Christoph Feichtenhofer
        @ https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

    This model takes in image and prompt encodings and creates segmentation masks.
    Compared to the V1 implementation, the V2 mask decoder includes extra 'object'
    outputs (an object pointer and object score, used for video processing). It
    also includes extra processing of image encodings, which makes use of the
    hi-res image features generated by the new V2 image encoder.

    In the original implementation, the model returned either 1 or 3 masks, was which
    meant to vary based on whether many prompts were given (return only 1 mask) or
    only 1 point or box prompt was given (return 3 masks). In this implementation,
    the model always returns all 4 masks, and it is left up to the end user to
    decide what to do with them.
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

        # Set up cls-like embeddings, which are combined with prompt embeddings as input into transformer
        self.cls_obj_token = nn.Parameter(torch.empty(1, input_channels))
        self.cls_mask_tokens = nn.Parameter(torch.empty(num_mask_tokens, input_channels))
        self.cls_iou_token = nn.Parameter(torch.empty(1, input_channels))

        # Create transformer decoder
        self.transformer = CrossAttentionTransformer(num_layers, num_heads, input_channels, downsample_dim)

        # Create layers for generating final outputs
        self.maskgen = MaskGen(input_channels, num_mask_tokens)
        self.iou_token_mlp = MLP3Layers(input_channels, num_mask_tokens, use_sigmoid_output=True)
        self.objptrgen = ObjectPointerGen(input_channels)

    # .................................................................................................................

    def forward(
        self,
        encoded_image_tokens_list_bchw: Tensor,
        encoded_prompts_bnc: Tensor,
        grid_positional_encoding: Tensor,
        mask_hint: Tensor | int | None = None,
        blank_promptless_output=True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Generates multiple candidate segmentation masks given an image encoding and
        encoded prompts (which specific the part of the image to segment).
        Also returns estimates for the 'quality' of each segmentation mask.

        The mask_hint input can be provided to help 'refine' the model output in some cases.
        It is expected to be of the same shape as a single mask output by the model. If
        the mask_hint argument is given as an integer, this is interpretted to mean to
        run the model twice, once to generate onc eset of masks and then to index out
        one of those masks to use as a hint for re-running the model.

        If 'blank_promptless_output' is true and no prompts are given, then a fully
        'blank' result will be returned instead of running the full decoder.

        Returns:
            mask_predictions, iou_predictions, object_pointers, object_score
            -> Mask prediction has shape: Bx4xHxW
            -> IoU has shape: Bx4
            -> Object pointer has shape: Bx4xF (F is features per token, default 256)
            -> Object score has shape Bx1 and indicates (score > 0) if an object is masked
            -> Mask H & W are 4x the size of the image patch encoding size
            -> The 0-th mask was originally intended to be used in cases with
               many prompts, and not to be used with single points/box prompts
        """

        # For clarity
        batch_size, num_prompts, enc_dim = encoded_prompts_bnc.shape
        lowres_tokens, hires_tokens_x2, hires_tokens_x4 = encoded_image_tokens_list_bchw
        patch_grid_hw = lowres_tokens.shape[2:]

        # Special case, return blank masks if no prompt is given
        if blank_promptless_output and (num_prompts == 0) and mask_hint is None:
            mask_preds, iou_preds = self.maskgen.make_blank_results(patch_grid_hw, batch_size)
            obj_score, obj_ptrs = self.objptrgen.make_blank_results(self.cls_mask_tokens, batch_size)
            return mask_preds, iou_preds, obj_ptrs, obj_score

        # Concatenate learned 'cls' tokens to prompts
        cls_tokens = torch.cat([self.cls_obj_token, self.cls_iou_token, self.cls_mask_tokens], dim=0)
        cls_tokens = cls_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        num_cls_tokens = cls_tokens.shape[1]

        # Expand per-image data in batch direction to be per-mask, as well as the position encoding
        img_tokens_bchw = lowres_tokens + self.maskhint_encoder(patch_grid_hw, mask_hint)
        img_tokens_bchw = torch.repeat_interleave(img_tokens_bchw, batch_size, dim=0)
        img_posenc_bchw = torch.repeat_interleave(grid_positional_encoding, batch_size, dim=0)

        # Cross-encode image tokens with prompt tokens
        prompt_tokens = torch.cat((cls_tokens, encoded_prompts_bnc), dim=1)
        encoded_prompt_tokens, encoded_img_tokens = self.transformer(prompt_tokens, img_tokens_bchw, img_posenc_bchw)

        # Extract the (now-encoded) 'cls' tokens by undoing the earlier cls concatenation step
        encoded_cls_tokens = encoded_prompt_tokens[:, :num_cls_tokens, :]
        obj_token_out = encoded_cls_tokens[:, 0, :]
        iou_token_out = encoded_cls_tokens[:, 1, :]
        mask_tokens_out = encoded_cls_tokens[:, 2:, :]

        # Produce final output mask & quality predictions
        mask_preds = self.maskgen(encoded_img_tokens, hires_tokens_x2, hires_tokens_x4, mask_tokens_out)
        iou_preds = self.iou_token_mlp(iou_token_out)

        # Generate 'object pointer' output
        obj_score, obj_ptrs = self.objptrgen(obj_token_out, mask_tokens_out)

        return mask_preds, iou_preds, obj_ptrs, obj_score

    # .................................................................................................................

    @staticmethod
    def get_best_mask_index(iou_predictions) -> int:
        """Helper used to select the index of the 'best' output, based on the highest IoU prediction score"""
        return int(iou_predictions.cpu().argmax())

    # .................................................................................................................

    @staticmethod
    def get_best_decoder_results(
        mask_preds, iou_preds, obj_ptrs, exclude_0th_index=True
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Helper used to keep only the 'best' result from the mask decoder predictions.
        The 'exclude_0th_index' flag is used to ignore the mask normally associated
        with 'multi-prompt' masking (which is stored in the 0th index of the results)

        Returns:
            best_index, best_mask_prediction, best_iou_score, best_object_pointer
            -> Best index is a tensor (!) with shape: B (i.e. 1 index for each batch entry)
            -> Mask prediction has shape: Bx1xHxW
            -> IoU has shape: Bx1
            -> Object pointer has shape: Bx1xF (F features, 256 by default)
        """

        # Use highest iou prediction as indicator of the 'best' results
        # -> Optionally exclude the 0th index, which is normally used when multiple-prompts are given
        best_idx = 1 + torch.argmax(iou_preds[:, 1:], dim=-1) if exclude_0th_index else torch.argmax(iou_preds, dim=-1)

        best_mask_pred = mask_preds[:, best_idx, :, :]
        best_iou_pred = iou_preds[:, best_idx]
        best_obj_ptr = obj_ptrs[:, best_idx, :]

        return best_idx, best_mask_pred, best_iou_pred, best_obj_ptr

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Model components


class MaskGen(nn.Module):
    """
    Sub-module of the SAM mask decoder which is responsible for generating the final output masks
    Performs upscaling of image embeddings (back to original input resolution) as well
    as a projection of encoder mask ('cls') tokens before dot-producting with image tokens
    to get final masks.

    The one difference between the V2 implementation (compared to SAM V1), is that the
    upscaling step involves additional hi-res copies of the encoded image tokens,
    which come from the hierarchical image encoder in SAMV2 (not present in V1).

    The original implementation was directly part of the mask decoder model. Here, it's been split
    out into it's own module for clarity. For the original implementation details see:
    https://github.com/facebookresearch/segment-anything-2/blob/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2/modeling/sam/mask_decoder.py#L65-L75
    """

    # .................................................................................................................

    def __init__(self, input_channels, num_mask_tokens):

        # Inherit from parent
        super().__init__()

        # Create model for handling image token upscaling
        upscaler_channels = input_channels // 8
        self.img_patch_upscaler = MaskUpscalerWithHiresSupport(input_channels, upscaler_channels)

        # Create model for further processing encoded mask 'cls' tokens
        self.mask_token_mlps = nn.ModuleList(
            [MLP3Layers(input_channels, upscaler_channels) for i in range(num_mask_tokens)]
        )

        # Create helper tensor for recording the model device/dtype information
        self.register_buffer("device_info", torch.empty(0), persistent=False)

    # .................................................................................................................

    def forward(
        self,
        lowres_image_tokens: Tensor,
        hires_tokens_x2: Tensor,
        hires_tokens_x4: Tensor,
        cls_mask_tokens: Tensor,
    ) -> Tensor:
        """
        Produces (non-thresholded) segmentation masks based on
        the encoded image & mask token data from earlier
        transformer layers. The output contains 4 (by default)
        masks, which are meant to help cover ambiguity
        (i.e. one mask is for 'whole object' another for 'sub-part').

        The 0-th index mask is 'special' in the original implementation,
        as it is used (exclusively) when more than one prompt is
        given (a.k.a. multimask_output=False in the original code base).

        Returns:
            segmentation_masks
            -> shape: Bx4xHxW
            -> B batch size
            -> H & W are based on the input image height & width
               (will be Bx4x256x256 using default 1024x1024 input sizing)
        """

        # Convert image tokens to 'image-like' shape and upscale them to final output size
        upscaled_img_tokens = self.img_patch_upscaler(lowres_image_tokens, hires_tokens_x2, hires_tokens_x4)

        # Further encode cls mask tokens
        encoded_mask_tokens_list = [mlp(cls_mask_tokens[:, i, :]) for i, mlp in enumerate(self.mask_token_mlps)]
        encoded_mask_tokens = torch.stack(encoded_mask_tokens_list, dim=1)

        # Take dot product (along channel dimension) of cls tokens with image patches for final masks
        return torch.einsum("bnc, bchw -> bnhw", encoded_mask_tokens, upscaled_img_tokens)

    # .................................................................................................................

    def make_blank_results(self, patch_grid_hw: tuple[int, int], batch_size: int) -> tuple[Tensor, Tensor]:
        """Helper used to generate a 'blank' mask, meant for cases where inputs aren't available"""

        # Due to upscaler layer, the normal mask output should be 4 times larger than the image encoding size!
        mask_h, mask_w = [4 * size for size in patch_grid_hw]
        mask_shape = (batch_size, 4, mask_h, mask_w)
        iou_shape = (batch_size, 4)

        # Fill in empty mask and IoU prediction values
        device, dtype = self.device_info.device, self.device_info.dtype
        blank_mask_preds = torch.full(mask_shape, -100, device=device, dtype=dtype, requires_grad=False)
        blank_iou_preds = torch.ones(iou_shape, device=device, dtype=dtype, requires_grad=False)

        return blank_mask_preds, blank_iou_preds

    # .................................................................................................................


class MaskUpscalerWithHiresSupport(nn.Module):
    """
    Helper module used to handle mask upscaling while incorporating the
    hi-res image feature maps available from the output of the SAMV2 image encoder

    This module does not exist in the original implementation, which instead unpacks the
    existing sequential model (from the SAMV1 implementation) and uses each component separately:
    https://github.com/facebookresearch/segment-anything-2/blob/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2/modeling/sam/mask_decoder.py#L222-L225
    """

    # .................................................................................................................

    def __init__(self, input_channels, output_channels):

        # Inherit from parent
        super().__init__()

        hidden_channels = input_channels // 4
        self.upscale_1 = nn.ConvTranspose2d(input_channels, hidden_channels, kernel_size=2, stride=2)
        self.norm = LayerNorm2d(hidden_channels)
        self.upscale_2 = nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=2, stride=2)
        self.gelu = nn.GELU()

    # .................................................................................................................

    def forward(self, lowres_image_tokens: Tensor, hires_tokens_x2: Tensor, hires_tokens_x4: Tensor) -> Tensor:
        """
        Upscales the provided low-resolution image tokens (by a factor of 4),
        while incorporating the 2x hi-res and 4x hi-res features
        Returns:
            upscaled_4x_image_tokens
        """

        # Upscale once by a factor of 2, and include hires 2x tokens
        upscaled_tokens_x2 = self.upscale_1(lowres_image_tokens)
        upscaled_tokens_x2 = self.norm(upscaled_tokens_x2 + hires_tokens_x2)
        upscaled_tokens_x2 = self.gelu(upscaled_tokens_x2)

        # Upscale again by a factor of 2, and include hires 4x tokens
        upscaled_tokens_x4 = self.upscale_2(upscaled_tokens_x2)
        upscaled_tokens_x4 = self.gelu(upscaled_tokens_x4 + hires_tokens_x4)

        return upscaled_tokens_x4

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
            nn.Conv2d(num_hidden_ch_2, output_channels, kernel_size=1),
        )

        # Helper variable used to keep track of the model device/dtype (need for mask hints)
        self.register_buffer("device_info", torch.empty(1), persistent=False)

    # .................................................................................................................

    def forward(self, patch_grid_hw: tuple[int, int], mask_hint_bhw: Tensor | None) -> Tensor:
        """
        If a mask hint is provided, it will be encoded to help adjust the 'prompt'
        when running the mask decoder. If a hint isn't given, then a learned
        'no-mask' embedding is used instead.

        The mask hint should ideally be 4x the height & width of the patch grid,
        corresponding to the size of the outputs of the mask decoder itself
        -> This seems to be the original intended usage, to feed back output
           masks as prompts for refinement
        """

        # Return no-mask embedding if no hint is provided
        grid_h, grid_w = patch_grid_hw
        if mask_hint_bhw is None:
            return self.no_mask_embed.reshape(1, -1, 1, 1).expand(1, -1, grid_h, grid_w)

        # Add batch dimensions, if needed
        if mask_hint_bhw.ndim == 2:
            mask_hint_bhw = mask_hint_bhw.unsqueeze(0)

        # Encode hint and scale to target size if needed
        encoded_mask_hint = self.downscaler(mask_hint_bhw.to(self.device_info))
        _, _, hint_h, hint_w = encoded_mask_hint.shape
        if hint_h != grid_h or hint_w != grid_w:
            encoded_mask_hint = nn.functional.interpolate(encoded_mask_hint, size=patch_grid_hw)

        return encoded_mask_hint

    # .................................................................................................................


class ObjectPointerGen(nn.Module):
    """
    Helper module of the SAMV2 mask decoder. This module handles the
    computation of object pointers and the 'object score', which are
    generated from encoded mask tokens and the encoded object token,
    respectively, from the mask decoder.
    """

    # .................................................................................................................

    def __init__(self, features_per_token=256):

        # Inherit from parent
        super().__init__()

        # Projection layers to convert encoded tokens to score & pointer
        # -> The 'no_ptr' parameter is the pointer given for low object scores
        self.no_ptr = nn.Parameter(torch.zeros(1, features_per_token))
        self.score_mlp = MLP3Layers(features_per_token, 1)
        self.pointer_mlp = MLP3Layers(features_per_token, features_per_token)

    # .................................................................................................................

    def forward(self, encoded_object_token, encoded_mask_tokens) -> tuple[Tensor, Tensor]:
        """
        Produces an object score and object pointer from encoded tokens
        from the mask decoder.

        The object score is a single number, which indicates how likely
        it is that a segmentated object is 'present'. This is meant for
        use with video segmentation, where the model is 'auto-prompting'
        and may not always segment an object. The score will typically be
        +5 or more when an object is present and take on negative values
        when there is no object.

        The object pointer is a 'memory' representation of the object,
        used to help to continue to segment the same object on future
        frames when running on videos. It's used by the memory fusion
        model. There is one pointer for each mask token (4 by default).

        Returns:
            object_score, object_pointers
            -> score shape is: Bx1
            -> pointer shape is: Bx4xF
            -> For B batch size, F features per token (256 by default)
        """

        # Compute object score (indicator of whether there is a valid object being masked)
        objscore = self.score_mlp(encoded_object_token)

        # Get pointer for each batch
        objptrs_list = []
        for batch_idx in range(encoded_mask_tokens.shape[0]):
            tokens = encoded_mask_tokens[batch_idx]
            ptr = self.pointer_mlp(tokens) if objscore[batch_idx] > 0 else self.no_ptr.expand_as(tokens)
            objptrs_list.append(ptr)
        objptrs = torch.stack(objptrs_list)

        return objscore, objptrs

    # .................................................................................................................

    def make_blank_results(self, mask_tokens, batch_size):
        """Helper used to produce 'no object' output when needing blank results"""
        no_obj_score = torch.tensor(-10).to(mask_tokens)
        no_obj_ptr = self.no_ptr.expand_as(mask_tokens).expand(batch_size, -1, -1)
        return no_obj_score, no_obj_ptr

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

    def __init__(self, input_dim, output_dim, use_sigmoid_output=False):

        # Inherit from parent
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid() if use_sigmoid_output else nn.Identity(),
        )

    # .................................................................................................................

    def forward(self, x):
        return self.layers(x)

    # .................................................................................................................
