#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3PromptEncoder(nn.Module):
    """
    Simplified implementation of the 'prompt-encoder' model/component originally described in:
        "Segment Anything"
        By: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
            Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
        @ https://arxiv.org/abs/2304.02643

    The original SAMv1, v2 & v3 code can be found here:
    https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/prompt_encoder.py#L16
    https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam/prompt_encoder.py#L17
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/sam/prompt_encoder.py#L12

    While this version is used in SAMV3, the implementation is identical to SAMV1 & SAMV2.
    (This code has been copied verbatim from the existing MuggledSAM implementation of SAMv2)

    The prompt encoder is is used to encode points prompts given as
    foreground points, background points or 2-point-bounding-boxes.
    The results from this model, along with the results from the corresponding image encoder are
    passed along to a mask decoder model to create segmentation masks.
    """

    # .................................................................................................................

    def __init__(self, output_channels=256):

        # Inherit from parent
        super().__init__()

        self.point_encoder = PointEncoder(output_channels)
        self.box_encoder = BoxEncoder(output_channels)

        # Create cache for storing a special encoding repeatedly used during video segmentation
        self.register_buffer("_cached_no_prompt", torch.zeros((1, 0, output_channels)), persistent=False)

    # .................................................................................................................

    def forward(self, posenc_boxes: Tensor, posenc_fg_pts: Tensor, posenc_bg_pts: Tensor) -> Tensor:
        """
        Encode all point-based prompts (FG/BG/Boxes) into a single 'prompt tensor'
        for use by the SAM mask decoder. Each of the inputs is expected to already be
        a position-encoded (by the coordinate-encoder model) tensor.
        FG/BG tensors should have a shape of: BxNxF
        The box tensor should have a shape of: BxNx2xF
        -> Where B is batch size, N is number of prompts, F is features per prompt

        Returns:
            prompt_tensor
            -> Has shape: BxN'xF
            -> Where N' is total number of prompt points = 2*N_boxes + N_FG + N_BG + 1
               (+1 is due to padding point added whenever a prompt is given)
        """

        # The original implementation adds an extra padding point when no boxes are given:
        # https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/sam/prompt_encoder.py#L184
        # But at the same time, never passes boxes as inputs! (it always treats them as points):
        # https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam1_task_predictor.py#L400
        # So use 1 padding point (if any points or boxes are given) or 0 otherwise
        device = self._cached_no_prompt.device
        num_total_prompts = posenc_fg_pts.shape[1] + posenc_bg_pts.shape[1] + posenc_boxes.shape[1]
        num_padding_points = torch.min(torch.tensor(num_total_prompts), torch.tensor(1)).to(device)

        fg_pt, bg_pt, pad_pt = self.point_encoder(posenc_fg_pts, posenc_bg_pts, num_padding_points)
        boxes_as_pts = self.box_encoder(posenc_boxes)

        # Merge all encodings together
        return torch.cat((boxes_as_pts, fg_pt, bg_pt, pad_pt), dim=1)

    # .................................................................................................................

    def create_video_no_prompt_encoding(self, batch_size=1) -> Tensor:
        """
        Helper used to create a 'no prompt' encoding for use during video segmentation.
        In SAMv2, this was somewhat elaborate due to the conditional use of padding points.
        In SAMv3, the model simply doesn't include point encodings during video segmentation,
        so a 0-length tensor is generated to maintain compatibility with existing processing.

        See the SAMv3 behavior here:
        https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam1_task_predictor.py#L379-L402

        Returns:
            no_prompt_encoding
            (shape: Bx0xF, B batch size, F features per token)
        """

        cache_batch_size = self._cached_no_prompt.shape[0]
        if cache_batch_size != batch_size:
            _, _, num_channels = self._cached_no_prompt.shape
            with torch.inference_mode():
                self._cached_no_prompt = torch.zeros((batch_size, 0, num_channels))

        return self._cached_no_prompt

    # .................................................................................................................

    @staticmethod
    def check_have_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list) -> bool:
        """Helper used to check if there are any prompts (i.e. check for at least one non-empty list)"""
        return any((len(items) for items in (box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)))

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Model components


class PointEncoder(nn.Module):
    """Simple helper component, used to handle foreground/background embeddings for the prompt encoder"""

    # .................................................................................................................

    def __init__(self, output_channels=256):

        # Inherit from parent
        super().__init__()

        # Set up point embeddings for background/foreground & box top-left/bottom-right points
        self.fg_embed = nn.Parameter(torch.empty(1, output_channels))
        self.bg_embed = nn.Parameter(torch.empty(1, output_channels))
        self.not_a_point_embed = nn.Parameter(torch.empty(1, output_channels))

    # .................................................................................................................

    def forward(
        self, posenc_fg_pts: Tensor, posenc_bg_pts: Tensor, num_padding_points=0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Adds foreground/background embedding to provided points as well as creating padding embedding, if needed.
        Both the fg & bg inputs are expected to have a shape of: BxNxF
        -> B is batch size, N is number of points, F is features per point
        -> N can be different for the fg & bg inputs
        -> Use N = 0 to disable the inputs

        Returns:
            fg_pts_embedding, bg_pts_embedding, padding_pts_embedding
            -> Each tensor is of shape: BxNxF
        """

        # Add foreground encoding data to output
        fg_out = posenc_fg_pts + self.fg_embed
        bg_out = posenc_bg_pts + self.bg_embed

        # Create 'not a point' encoding to pad the prompt if needed
        batch_size = max(1, fg_out.shape[0], bg_out.shape[0])
        pad_out = self.create_padding_point_encoding(batch_size, num_padding_points)

        return fg_out, bg_out, pad_out

    # .................................................................................................................

    def create_padding_point_encoding(self, batch_size=1, num_padding_points=1):
        """Helper used to standardize padding point creation"""
        return self.not_a_point_embed.tile(batch_size, num_padding_points, 1)

    # .................................................................................................................


class BoxEncoder(nn.Module):
    """Simple helper component, used to handle bounding-box embeddings for the prompt encoder"""

    # .................................................................................................................

    def __init__(self, output_channels=256):

        # Inherit from parent
        super().__init__()

        self.tl_embed = nn.Parameter(torch.empty(1, output_channels))
        self.br_embed = nn.Parameter(torch.empty(1, output_channels))

    # .................................................................................................................

    def forward(self, posenc_tlbr_points: Tensor) -> Tensor:
        """
        Adds top-left/bottom-right embedding to provided bounding boxes.
        The input is expected to have a shape of: BxNx2xF
        -> B is batch size, N is number of boxes, F is features per point
        -> The '2' dimension is meant to hold the encoding for the top-left, then bottom-right point encodings!
        -> Use N = 0 to disable box encoding

        Returns:
            box_pts_embedding
            -> Has shape: Bx(2N)xF
        """

        posenc_tlbr_points[:, :, 0, :] += self.tl_embed
        posenc_tlbr_points[:, :, 1, :] += self.br_embed

        # Stack top-left/bot-right embeddings together, so shape becomes: BxNx2xF -> Bx(2N)xF
        return posenc_tlbr_points.flatten(1, 2)

    # .................................................................................................................
