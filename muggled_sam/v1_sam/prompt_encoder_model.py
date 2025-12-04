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


class SAMV1PromptEncoder(nn.Module):
    """
    Simplified implementation of the 'prompt-encoder' model/component described in:
        "Segment Anything"
        By: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
            Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
        @ https://arxiv.org/abs/2304.02643

    The code here is adapted from the original segment-anything repo:
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py

    The prompt encoder is is used to encode points prompts given as
    foreground points, background points or 2-point-bounding-boxes.
    The results from this model, along with the results from the corresponding image encoder are
    passed along to a mask decoder model to create segmentation masks.

    The mechanism behind this model is extremely simple. It takes position-encoded xy cordinates
    (handled by another model) and merely 'adds' a learned embedding.
    There are separate learned embeddings for background & foreground points
    as well as the top-left and bottom-right points for bounding boxes.
    Additionally there is a 'not-a-point' embedding which is used to pad
    out prompts that do not include boxes.
    (padding is not strictly neccessary, but comes from the original implementation)

    The original implementation of the prompt encoder was more complex and included
    encoding of 'mask' inputs as well as positional encoding of prompt points.
    In this implementation, mask inputs are now handled by the mask decoder model, while
    the positional encoding has been separated out into it's own full module
    (called the 'coordinate encoder).

    In addition, the original model only supported having a single box prompt, while
    any number of box prompts can be provided in this version, though performance
    seems poor, likely because the original wasn't trained for this!
    """

    # .................................................................................................................

    def __init__(self, output_channels=256):

        # Inherit from parent
        super().__init__()

        self.point_encoder = PointEncoder(output_channels)
        self.box_encoder = BoxEncoder(output_channels)

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
            -> Where N' is total number of prompt points = N_FG + N_BG + 2*N_boxes
        """

        # The original implementation added extra padding points when no boxes where given:
        # https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/prompt_encoder.py#L155
        # -> From brief testing, this isn't strictly required, but does
        #    seem to give slightly nicer results, at least qualitatively.
        #    The behavior has been replicated here for consistency
        no_points = posenc_fg_pts.shape[1] == 0 and posenc_bg_pts.shape[1] == 0
        no_boxes = posenc_boxes.shape[1] == 0
        num_padding_points = 1 if (no_boxes and not no_points) else 0
        fg_pt, bg_pt, pad_pt = self.point_encoder(posenc_fg_pts, posenc_bg_pts, num_padding_points)
        boxes_as_pts = self.box_encoder(posenc_boxes)

        # Merge all encodings together
        return torch.cat((fg_pt, bg_pt, pad_pt, boxes_as_pts), dim=1)

    # .................................................................................................................

    @staticmethod
    def check_have_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list):
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
        pad_out = self.not_a_point_embed.tile(batch_size, num_padding_points, 1)

        return fg_out, bg_out, pad_out

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
