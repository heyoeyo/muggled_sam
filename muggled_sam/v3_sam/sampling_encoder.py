#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.sampling_encoder_components import BoxSampleEncoder, PointSampleEncoder
from .components.sampling_encoder_attention import SelfAttentionBlock, CrossAttentionBlock
from .components.position_encoding import SinusoidalPE2D
from .components.shared import MLP2LayersPreNorm, imagelike_to_rows_of_tokens, rows_of_tokens_to_imagelike

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3SamplingEncoder(nn.Module):
    """
    Simplified implementation of the 'SequenceGeometryEncoder' model described in:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This model is responsible for taking in coordinates (in the form of points or boxes)
    and using them to 'sample' from image tokens, as well as performing some additional
    encoding steps and (optionally) encoding the coordinate data itself.

    The combination of these tokens with text tokens are referred to as 'exemplar' tokens.

    See the original implementation:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/geometry_encoders.py#L481
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int = 256, num_layers: int = 3, num_heads: int = 8):

        # Inherit from parent
        super().__init__()

        # Image pre-processing norm
        self.img_pre_norm = nn.LayerNorm(features_per_token)
        self.img_posenc = SinusoidalPE2D(features_per_token)

        # Storage for learned tokens/embeddings
        self.positive_coord_label = nn.Parameter(torch.empty(1, 1, features_per_token))
        self.negative_coord_label = nn.Parameter(torch.empty(1, 1, features_per_token))
        self.cls_token = nn.Parameter(torch.empty(1, 1, features_per_token))

        # Modules used to handle point/box inputs
        self.point_encoder = PointSampleEncoder(features_per_token)
        self.box_encoder = BoxSampleEncoder(features_per_token)

        # Set up pre-processing of tokens used prior to transformer layers
        self.layer_pre_norm = nn.Sequential(
            nn.Linear(features_per_token, features_per_token),
            nn.LayerNorm(features_per_token),
        )

        # Create transformer layers (bulk of the model)
        self.fusion_layers = nn.ModuleList(
            (SamplingFusionLayer(features_per_token, num_heads) for idx in range(num_layers))
        )

        # Final output norm
        self.output_norm = nn.LayerNorm(features_per_token)

        # Set up storage for keeping track of the model device
        self.register_buffer("missing_token_bnc", torch.zeros(1, 0, features_per_token), persistent=False)

    # .................................................................................................................

    def forward(
        self,
        image_tensor_bchw: Tensor,
        boxes_bn22: Tensor | None = None,
        points_bn2: Tensor | None = None,
        negative_boxes_bn22: Tensor | None = None,
        negative_points_bn2: Tensor | None = None,
        include_coordinate_encodings: bool = True,
    ) -> Tensor:

        # Pre-process image tokens
        img_tensor_bnc, img_hw = imagelike_to_rows_of_tokens(image_tensor_bchw)
        encimg_bnc = self.img_pre_norm(img_tensor_bnc)
        encimg_bchw = rows_of_tokens_to_imagelike(encimg_bnc, img_hw)
        img_pos_enc_bnc, _ = imagelike_to_rows_of_tokens(self.img_posenc(*img_hw))

        # Encode whichever coordinates we received
        sampling_tokens_list = []
        if boxes_bn22 is not None:
            box_enc_bnc = self.box_encoder(encimg_bchw, boxes_bn22, include_coordinate_encodings)
            box_enc_bnc = box_enc_bnc + self.positive_coord_label
            sampling_tokens_list.append(box_enc_bnc)
        if points_bn2 is not None:
            pt_enc_bnc = self.point_encoder(encimg_bchw, points_bn2, include_coordinate_encodings)
            pt_enc_bnc = pt_enc_bnc + self.positive_coord_label
            sampling_tokens_list.append(pt_enc_bnc)
        if negative_boxes_bn22 is not None:
            neg_box_enc_bnc = self.box_encoder(encimg_bchw, negative_boxes_bn22, include_coordinate_encodings)
            neg_box_enc_bnc = neg_box_enc_bnc + self.negative_coord_label
            sampling_tokens_list.append(neg_box_enc_bnc)
        if negative_points_bn2 is not None:
            neg_pt_enc_bnc = self.point_encoder(encimg_bchw, negative_points_bn2, include_coordinate_encodings)
            neg_pt_enc_bnc = neg_pt_enc_bnc + self.negative_coord_label
            sampling_tokens_list.append(neg_pt_enc_bnc)

        # If we don't end up with any inputs, return a missing token (avoids returning just the cls token)
        if len(sampling_tokens_list) == 0:
            return self.missing_token_bnc

        # Combine cls into tokens
        max_b = max(tokens.shape[0] for tokens in sampling_tokens_list)
        cls_token_bnc = self.cls_token.repeat(max_b, 1, 1) if max_b > 1 else self.cls_token
        sampling_tokens_list.append(cls_token_bnc)
        sampling_tokens_bnc = torch.cat(sampling_tokens_list, dim=1)

        # Mix image feature information into sampling tokens
        sampling_tokens_bnc = self.layer_pre_norm(sampling_tokens_bnc)
        for fusion_layer in self.fusion_layers:
            sampling_tokens_bnc = fusion_layer(
                sampling_tokens_bnc=sampling_tokens_bnc,
                image_tokens_bnc=img_tensor_bnc,
                image_posenc_bnc=img_pos_enc_bnc,
            )

        return self.output_norm(sampling_tokens_bnc)

    # .................................................................................................................

    def prepare_box_input(
        self,
        box_xy1xy2_norm_list: (
            list[tuple[tuple[float, float], tuple[float, float]]] | list[tuple[float, float, float, float]] | None
        ),
    ) -> Tensor | None:
        """
        Helper used to convert box inputs into the tensor format needed by the model.
        Boxes can be given as either a list of [(x1,y1), (x2,y2)] pairs or
        as a flat list of [x1,y1,x2,y2] coordinates, for example:

            Format 1 (xy1xy2): box_xy1xy2_norm_list = [[(0.4, 0.2), (0.6, 0.5)], [(0.1, 0.5), (0.7, 0.8)], ...]
            -or-
            Format 2 (xyxy):   box_xyxy_norm_list = [(0.4,0.2,0.6,0.5), (0.1,0.5,0.7,0.8), ...]

        Note: (x1,y1) corresponds to the top-left of the box, while (x2,y2) is the bottom right.
        """

        # Bail on non-inputs
        if box_xy1xy2_norm_list is None:
            return None
        if len(box_xy1xy2_norm_list) == 0:
            return None

        # Convert to torch data
        device, dtype = self.missing_token_bnc.device, self.missing_token_bnc.dtype
        boxes_tensor_bn22 = torch.tensor(box_xy1xy2_norm_list, device=device, dtype=dtype)
        input_shape = boxes_tensor_bn22.shape

        # Force into [(x1,y1), (x2,y2)] and force x1 < x2, y1 < y2
        if boxes_tensor_bn22.shape[-1] == 4:
            boxes_tensor_bn22 = boxes_tensor_bn22.unflatten(-1, (2, 2))
        assert boxes_tensor_bn22.shape[-2] == 2, f"Boxes must be given as Nx2x2 or Nx4 format (Got: {input_shape})"
        boxes_tensor_bn22, _ = torch.sort(boxes_tensor_bn22, dim=-2)

        # Force into BxNx2x2 shape
        if boxes_tensor_bn22.ndim == 2:
            boxes_tensor_bn22 = boxes_tensor_bn22.unsqueeze(0).unsqueeze(0)
        elif boxes_tensor_bn22.ndim == 3:
            boxes_tensor_bn22 = boxes_tensor_bn22.unsqueeze(0)

        # Sanity check
        assert boxes_tensor_bn22.ndim == 4, f"Unexpected box shape: {input_shape}, should be Nx2x2 or Nx4 for N boxes"
        return boxes_tensor_bn22

    def prepare_point_input(self, point_xy_norm_list: list[tuple[float, float]] | None) -> Tensor | None:
        """
        Helper used to convert points inputs into the tensor format needed by the model.
        Points should be given as a list of (x,y) coordinates, for example:
            point_xy_norm_list = [(0.25, 0.5), (0.1, 0.33), (0.25, 0.8), ...]
        """

        # Bail on non-inputs
        if point_xy_norm_list is None:
            return None
        if len(point_xy_norm_list) == 0:
            return None

        # Convert to torch data
        device, dtype = self.missing_token_bnc.device, self.missing_token_bnc.dtype
        points_tensor_bn2 = torch.tensor(point_xy_norm_list, device=device, dtype=dtype)
        input_shape = points_tensor_bn2.shape

        # Force into BxNx2 shape
        assert points_tensor_bn2.shape[-1] == 2, f"Points must be given in Nx2 format (Got: {input_shape})"
        if points_tensor_bn2.ndim == 1:
            points_tensor_bn2 = points_tensor_bn2.unsqueeze(0).unsqueeze(0)
        if points_tensor_bn2.ndim == 2:
            points_tensor_bn2 = points_tensor_bn2.unsqueeze(0)

        # Sanity check
        assert points_tensor_bn2.ndim == 3, f"Unexpected points shape: {input_shape}, should be Nx2x2 for N boxes"
        return points_tensor_bn2

    # .................................................................................................................


class SamplingFusionLayer(nn.Module):
    """
    Simplified implementation of the 'TransformerEncoderLayer' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This is a single transformer-like block which fuses information from image
    tokens into a provided set of sampling tokens.

    See original implementation:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/encoder.py#L13
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int = 256, num_heads: int = 8, mlp_ratio: float = 8.0):

        # Inherit from parent
        super().__init__()

        # Set up model components
        self.selfattn = SelfAttentionBlock(features_per_token, num_heads)
        self.img_crossattn = CrossAttentionBlock(features_per_token, num_heads)
        self.mlp = MLP2LayersPreNorm(features_per_token, mlp_ratio)

    def forward(
        self,
        sampling_tokens_bnc: Tensor,
        image_tokens_bnc: Tensor,
        image_posenc_bnc: Tensor,
        sampling_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Fuse information from the image tokens into the given sampling tokens
        Returns:
            fused_sampling_tokens_bnc (same shape)
        """
        enc_sampling_tokens = self.selfattn(sampling_tokens_bnc, sampling_key_padding_mask)
        enc_sampling_tokens = self.img_crossattn(enc_sampling_tokens, image_tokens_bnc, image_posenc_bnc)
        enc_sampling_tokens = self.mlp(enc_sampling_tokens)
        return enc_sampling_tokens

    # .................................................................................................................
