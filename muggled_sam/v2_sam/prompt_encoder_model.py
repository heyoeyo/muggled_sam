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


class SAMV2PromptEncoder(nn.Module):
    """
    Simplified implementation of the 'prompt-encoder' model/component originally described in:
        "Segment Anything"
        By: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
            Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
        @ https://arxiv.org/abs/2304.02643

    The code here is adapted from the original segment-anything v1 & v2 repos:
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py
    https://github.com/facebookresearch/segment-anything-2/blob/main/sam2/modeling/sam/prompt_encoder.py

    Note that the SAMv2 implementation is identical to v1!

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
        self.register_buffer("_cached_no_prompt", torch.empty((0, 1, 1)), persistent=False)

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
        # https://github.com/facebookresearch/segment-anything-2/blob/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2/modeling/sam/prompt_encoder.py#L169
        # But at the same time, never passes boxes as inputs! (it always treats them as points):
        # https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/sam2_image_predictor.py#L406
        # So a padding point is always added, as long as either a point or box prompt is given
        # (this is slightly different from the v1 implementation, which does handle boxes separately)
        # -> From brief testing, this isn't strictly required, but does
        #    seem to give slightly nicer results, at least qualitatively.
        #    The behavior has been replicated here for consistency
        no_points = posenc_fg_pts.shape[1] == 0 and posenc_bg_pts.shape[1] == 0
        no_boxes = posenc_boxes.shape[1] == 0
        num_padding_points = 0 if (no_boxes and no_points) else 1
        fg_pt, bg_pt, pad_pt = self.point_encoder(posenc_fg_pts, posenc_bg_pts, num_padding_points)
        boxes_as_pts = self.box_encoder(posenc_boxes)

        # Merge all encodings together
        return torch.cat((boxes_as_pts, fg_pt, bg_pt, pad_pt), dim=1)

    # .................................................................................................................

    def create_video_no_prompt_encoding(self, batch_size=1) -> Tensor:
        """
        Helper used to mimic the 'no prompt' encoding used by SAMv2 during video segmentation
        which uses 2 padding points. To see how this happens in the SAMv2 model, note that first,
        if no prompt is given then a single padding point is used as the prompt:
        https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L316-L318

        Also note that for the box prompt, a 'None' value is used:
        https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L342

        Due to the way the prompt encoder is structured, if a point prompt is given with a box input of 'None',
        then the point encoding gets a padding point:
        https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam/prompt_encoder.py#L169
        https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam/prompt_encoder.py#L87-L91

        So the result is one explicitly given padding point + an additional padding point due to a box input of 'None'.
        This has a shockingly large impact on the results, at least for the object score prediction
        (which tends to be lower without this encoding, most notably with the v2.1 tiny model).

        Returns:
            no_prompt_encoding
            (shape: Bx2xF, B batch size, F features per token)
        """

        cache_batch_size = self._cached_no_prompt.shape[0]
        if cache_batch_size != batch_size:
            with torch.inference_mode():
                self._cached_no_prompt = self.point_encoder.create_padding_point_encoding(
                    batch_size, num_padding_points=2
                )

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
        """Helper used to standardize padding point creation (important for SAMv2 video processing!)"""
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
