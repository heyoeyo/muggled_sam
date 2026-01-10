#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.shared import MLPMultiLayer, imagelike_to_rows_of_tokens
from .components.position_encoding import SinusoidalPE2D
from .components.exemplar_detector_components import PresenceScoreMLP, DetectionScoring
from .components.exemplar_detector_attention import (
    SelfAttentionBlock,
    ExemplarCrossAttentionBlock,
    ImageCrossAttentionBlock,
    MLP2LayersPostNorm,
)

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3ExemplarDetector(nn.Module):
    """
    Simplified implementation of the 'TransformerDecoder' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This model is responsible for generating bounding-box predictions for a given set of
    image tokens and corresponding exemplar tokens. It also generates a single score
    predicting whether an exemplar is present at all in the image, along with a
    per-detection 'confidence score'. It always generates 200 (by default) predictions.

    Unlike SAMv1 or SAMv2, this module allows SAMv3 to predict 'all instances' of an object
    from a single 'prompt' (which is what the encoded exemplar tokens represent).

    This function is mostly equivalent to the 'run_decoder' function in the original implementation:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/sam3_image.py#L254
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int = 256,
        num_detections: int = 200,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 8.0,
    ):
        # Inherit from parent
        super().__init__()

        # Set up xy coordinate encoder. This doesn't have any learned weights
        self.coord_posenc = SinusoidalPE2D(features_per_token)

        # Set up learned tokens/embeddings
        self.detection_tokens = nn.Parameter(torch.empty(1, num_detections, features_per_token))
        self.anchor_boxes_cxcywh = nn.Parameter(torch.empty(1, num_detections, 4))
        self.presence_token = nn.Parameter(torch.empty(1, 1, features_per_token))

        # Set up MLPs
        xy_features = 2 * features_per_token
        self.mlp_detection_posenc = MLPMultiLayer(xy_features, features_per_token, features_per_token, num_layers=2)
        self.mlp_box_relpos_dx = MLPMultiLayer(2, features_per_token, num_heads, num_layers=2)
        self.mlp_box_relpos_dy = MLPMultiLayer(2, features_per_token, num_heads, num_layers=2)
        self.mlp_detection_to_box = MLPMultiLayer(features_per_token, features_per_token, 4, num_layers=3)
        self.mlp_presence_score = PresenceScoreMLP(features_per_token)

        # Set up transformer fusion layers (bulk of model)
        self.fusion_layers = nn.ModuleList(
            (DetectionFusionLayer(features_per_token, num_heads, mlp_ratio) for _ in range(num_layers))
        )

        # Set up post-processing components
        self.out_norm_detections = nn.LayerNorm(features_per_token)
        self.detection_scoring = DetectionScoring(features_per_token, mlp_ratio)

        # Allocate storage for caching calculations used in generating position bias
        self.register_buffer("_img_y_norm_cache", torch.zeros(1, 1, 1, 1), persistent=False)
        self.register_buffer("_img_x_norm_cache", torch.zeros(1, 1, 1, 1), persistent=False)

    # .................................................................................................................

    def forward(
        self,
        image_tokens_bchw: Tensor,
        exemplar_tokens_bnc: Tensor,
        exemplar_mask_bn: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Predicts bounding box detections along with a per-detection confidence score
        and a single 'presence' score (indicates if exemplars are present in image).

        Returns:
            detection_tokens_bnc, box_xy1xy2_bn22, detection_scores_bn, presence_score
            -> Detection tokens are meant for internal use. They have shape: BxNxC (N is 200 by default)
            -> box_xy1xy2_bn22 are bounding box predictions in [(x1,y1), (x2,y2)] format
            -> detection_scores_bn holds a 0-to-1 'confidence score' for each box detection
            -> presence_score holds a 0-to-1 score for whether at least 1 object is present in image
        """

        # For clarity
        img_b, _, img_h, img_w = image_tokens_bchw.shape
        num_detections = self.detection_tokens.shape[1]

        # Set up image position encoding
        img_tokens_bnc, _ = imagelike_to_rows_of_tokens(image_tokens_bchw)
        img_pos_bnc, _ = imagelike_to_rows_of_tokens(self.coord_posenc(img_h, img_w))

        # Set up initial box predictions from learned anchors
        anc_boxes_bn4 = self.anchor_boxes_cxcywh.repeat(img_b, 1, 1)
        box_pred_cxcywh_bn4 = anc_boxes_bn4.sigmoid()
        box_pred_xyxy_bn4 = self._box_cxcywh_to_xyxy(box_pred_cxcywh_bn4)

        # Iteratively encode query & box predictions
        detection_tokens_bnc = self.detection_tokens.repeat(img_b, 1, 1)
        presence_token_b1c = self.presence_token.repeat(img_b, 1, 1)
        for fusion_layer in self.fusion_layers:

            # Compute position encoding for queries, follows odd ordering (yxwh) of original code
            # see: https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/model_misc.py#L271
            pos_x, pos_y, pos_w, pos_h = self.coord_posenc.encode_tensor(box_pred_cxcywh_bn4).unbind(-2)
            pos_yxwh_bnf = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)  # Shape is: BxNxf, with f=(C/2)
            det_posenc_bnc = self.mlp_detection_posenc(pos_yxwh_bnf)

            # Iteratively encode presence & detection tokens
            relposbias_attn_mask = self._make_relative_position_bias_attn_mask(box_pred_xyxy_bn4, img_h, img_w)
            presence_token_b1c, detection_tokens_bnc = fusion_layer(
                presence_token_b1c,
                detection_tokens_bnc,
                det_posenc_bnc,
                exemplar_tokens_bnc,
                exemplar_mask_bn,
                img_tokens_bnc,
                img_pos_bnc,
                relposbias_attn_mask,
            )

            # Update box prediction by iteratively offseting initial anchor predictions
            anc_offsets_bn4 = self.mlp_detection_to_box(detection_tokens_bnc)
            anc_boxes_bn4 = anc_boxes_bn4 + anc_offsets_bn4
            box_pred_cxcywh_bn4 = anc_boxes_bn4.sigmoid()
            box_pred_xyxy_bn4 = self._box_cxcywh_to_xyxy(box_pred_cxcywh_bn4)

        # Gather outputs
        out_detection_tokens_bnc = self.out_norm_detections(detection_tokens_bnc)
        out_presence_score_b1 = self.mlp_presence_score(presence_token_b1c)
        out_box_xy1xy2_bn22 = box_pred_xyxy_bn4.view(img_b, num_detections, 2, 2)

        # Compute per-detection confidence scores
        out_det_scores_bn = self.detection_scoring(out_detection_tokens_bnc, exemplar_tokens_bnc, exemplar_mask_bn)
        out_det_scores_bn = out_det_scores_bn * out_presence_score_b1
        out_presence_scores = out_presence_score_b1.squeeze(-1)

        return out_detection_tokens_bnc, out_box_xy1xy2_bn22, out_det_scores_bn, out_presence_scores

    # .................................................................................................................

    def _make_relative_position_bias_attn_mask(self, boxes_xyxy_bn4: Tensor, image_h: int, image_w: int):
        """
        Computes a relative position bias for each every box and image-token position.
        More specifically, the bias is computed based on the relative position between each image token and
        each set of box coordinates. Expects box shape: BxNx4 (B batches, N number of boxes, 4 for x1,y1,x2,y2).

        The output of this function is meant to act as a position bias within an attention calculation:
            attn = softmax(Q*K + bias) * V

        This is called the 'rpb matrix' in the original implementation, see:
        https://github.com/facebookresearch/sam3/blob/7b89b8fc3fa0ae8d09d9b17a284b5299e238b1b0/sam3/model/decoder.py#L333

        Returns:
            relative_position_bias_attn_mask
            -> Has weird shape: (B*C)xNx(H*W)
            -> B batches (matching box batches)
            -> C is a channel count, which should match the number of heads in attn where this mask is used
            -> N is number of boxes
            -> H & W are image token sizing

        Shape is weird due to use of 'attn_mask' in pytorch multihead attention module, see:
        https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/nn/modules/activation.py#L1290
        This is also why it's called a 'mask' even though it's really a bias term.
        """

        # Get image token xy coords
        device, dtype = boxes_xyxy_bn4.device, boxes_xyxy_bn4.dtype
        if image_h != self._img_y_norm_cache.shape[-2]:
            img_tokens_y_norm = torch.arange(0, image_h, device=device, dtype=dtype) / image_h
            self._img_y_norm_cache = img_tokens_y_norm[None, None, :, None]
        if image_w != self._img_x_norm_cache.shape[-2]:
            img_tokens_x_norm = torch.arange(0, image_w, device=device, dtype=dtype) / image_w
            self._img_x_norm_cache = img_tokens_x_norm[None, None, :, None]
        img_tokens_x_norm = self._img_x_norm_cache
        img_tokens_y_norm = self._img_y_norm_cache

        # For clarity, split box x/y and adjust shape to BxNx1x2
        box_x1x2_bn12 = boxes_xyxy_bn4[:, :, 0:3:2].unsqueeze(-2)
        box_y1y2_bn12 = boxes_xyxy_bn4[:, :, 1:4:2].unsqueeze(-2)

        # Compute difference between image x/y coords and box x/y coords
        # -> This gives dx1,dx2,dy1,dy2 for all 'image rows to box y coords' and 'image columns to box x coords'
        dx_bnw2 = img_tokens_x_norm - box_x1x2_bn12
        dy_bnh2 = img_tokens_y_norm - box_y1y2_bn12

        # Apply log-normalization to coordinate differences
        # -> Normalization maps results to -1 to +1 range, use of log redistributes more values to lower deltas
        # -> Idea seems to be to have 'more resolution' for describing positions that are nearby vs. far away
        dx_bnw2 = torch.sign(dx_bnw2) * (torch.log2(8 * torch.abs(dx_bnw2) + 1.0) / 3.0)
        dy_bnh2 = torch.sign(dy_bnh2) * (torch.log2(8 * torch.abs(dy_bnh2) + 1.0) / 3.0)

        # Encode deltas into tokens (also has effect of merging separate dx1 & dx2, likewise for y deltas)
        # -> The channel count needs to match the number of heads in multi-head attention where mask is used
        dx_bias_bn1wc = self.mlp_box_relpos_dx(dx_bnw2).unsqueeze(2)
        dy_bias_bnh1c = self.mlp_box_relpos_dy(dy_bnh2).unsqueeze(3)

        # Combine bias terms into an image-like shape, then convert to 'attn_mask' shape: (B*C)xNx(H*W)
        relpos_bias_bnhwc = dx_bias_bn1wc + dy_bias_bnh1c
        relpos_bias_attn_mask = relpos_bias_bnhwc.permute(0, 4, 1, 2, 3).flatten(3, 4).flatten(0, 1)
        return relpos_bias_attn_mask.contiguous()

    # .................................................................................................................

    @staticmethod
    def _box_cxcywh_to_xyxy(box_cxcywh_bn4: Tensor) -> Tensor:
        """Helper used to convert (x-center, y-center, width, height) box format to (x1, y1, x2, y2)"""
        x_cen = box_cxcywh_bn4[:, :, 0]
        y_cen = box_cxcywh_bn4[:, :, 1]
        half_w = box_cxcywh_bn4[:, :, 2] * 0.5
        half_h = box_cxcywh_bn4[:, :, 3] * 0.5
        return torch.stack((x_cen - half_w, y_cen - half_h, x_cen + half_w, y_cen + half_h), dim=-1)

    # .................................................................................................................


class DetectionFusionLayer(nn.Module):
    """
    Simplified implementation of the 'TransformerDecoderLayer' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    Helper module used to group transformer-like operations used within the detector.
    This model is a bit interesting in the sense that it fuses information from *both*
    exemplar tokens as well as image tokens into the detection/presence tokens.

    See original implementation:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/decoder.py#L31
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 8.0,
    ):

        # Inherit from parent
        super().__init__()

        # Set up model components
        self.query_selfattn = SelfAttentionBlock(features_per_token, num_heads)
        self.exemplar_crossattn = ExemplarCrossAttentionBlock(features_per_token, num_heads)
        self.image_crossattn = ImageCrossAttentionBlock(features_per_token, num_heads)
        self.query_mlp = MLP2LayersPostNorm(features_per_token, mlp_ratio)

    # .................................................................................................................

    def forward(
        self,
        presence_token_bnc: Tensor,
        detection_tokens_bnc: Tensor,
        detection_posenc_bnc: Tensor,
        exemplar_tokens_bnc: Tensor,
        exemplar_mask_bn: Tensor | None,
        image_tokens_bnc: Tensor,
        image_posenc_bnc: Tensor,
        image_attn_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """
        Updates presence & detection tokens through cross-attention with exemplar tokens,
        followed by cross-attention with image tokens.

        Note: Although most inputs have same 'rows-of-tokens' shape (BxNxC), the number
        of tokens (N) is generally different for presence token (N=1), detections (N=200),
        exemplars (N ~ 1 to 10) and image tokens (N ~ 1000's).

        The attn_mask should have a shape of: (B*D)xNdxNi,
        where B is batches, D is number of heads of attention module (8 by default),
        Nd is number of detections and Ni is number of image tokens.

        Returns:
            encoded_presence_token_bnc, encoded_detection_tokens_bnc
            (same shapes as inputs)
        """

        # Add presence as 'cls' token to detection tokens to form 'query' tokens for all attention ops
        q_tokens_bnc = torch.cat((presence_token_bnc, detection_tokens_bnc), dim=1)
        q_posenc_bnc = torch.cat((torch.zeros_like(presence_token_bnc), detection_posenc_bnc), dim=1)

        # Pad image attention mask to account for added presence token
        # -> Note attn mask has strange shape: (B*heads)x(NumDetections)x(ImgH*ImgW)
        # -> We're padding it from: NumDetections -to-> (1 + NumDetections)
        pres_token_mask = torch.zeros_like(image_attn_mask[:, :1, :])
        attn_mask = torch.cat((pres_token_mask, image_attn_mask), dim=1)

        # Fuse exemplar & image information into presence/detection tokens
        q_tokens_bnc = self.query_selfattn(q_tokens_bnc, q_posenc_bnc)
        q_tokens_bnc = self.exemplar_crossattn(q_tokens_bnc, q_posenc_bnc, exemplar_tokens_bnc, exemplar_mask_bn)
        q_tokens_bnc = self.image_crossattn(q_tokens_bnc, q_posenc_bnc, image_tokens_bnc, image_posenc_bnc, attn_mask)
        q_tokens_bnc = self.query_mlp(q_tokens_bnc)

        # Split presence & detection tokens
        out_presence_token_bnc, out_detection_tokens_bnc = q_tokens_bnc[:, :1], q_tokens_bnc[:, 1:]
        return out_presence_token_bnc, out_detection_tokens_bnc

    # .................................................................................................................
