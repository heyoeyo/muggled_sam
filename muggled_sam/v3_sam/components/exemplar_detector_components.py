#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .exemplar_detector_attention import MLP2LayersPostNorm

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class PresenceScoreMLP(nn.Module):
    """
    Simplified implementation of the layernorm + MLP model + clamping used by the
    SAM3 'transformer.decoder' to process the final 'presence token' output.
    This model takes in a high-dimensional token and produces a single value (score), per batch.
    This score is meant to be an indicator for whether exemplars are present in the image,
    independent of the per-detection confidence score.

    The original implementation handles this as separate components but it's combined here for clarity.
    Also note, the original implementation clamps the values but does not store the result! This
    version does apply clamping, though it doesn't seem important in typical usage.

    See the original implementation here:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/decoder.py#L582-L591
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, clamp_bound: float = 10.0):

        # Inherit from parent
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(features_per_token),
            nn.Linear(features_per_token, features_per_token),
            nn.ReLU(),
            nn.Linear(features_per_token, features_per_token),
            nn.ReLU(),
            nn.Linear(features_per_token, 1),
        )

        # Store both bounds for ease of use
        self._min_clamp = -clamp_bound
        self._max_clamp = clamp_bound

    def forward(self, tokens_bnc: Tensor) -> Tensor:
        """Reduces input token channels to a single value (e.g. score), also alters shape from: BxNxC -> BxN"""
        out_tokens_bn = self.layers(tokens_bnc).squeeze(-1)
        return out_tokens_bn.clamp(min=self._min_clamp, max=self._max_clamp).sigmoid()

    # .................................................................................................................


class DetectionScoring(nn.Module):
    """
    Simplified implementation of the 'DotProductScoring' model from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This model is responsible for computing a per-detection confidence score.
    It works somewhat like the first part of an attention calculation:
        score = sigmoid(q*k * scale) -> (analogous to: attn = softmax(q*k * scale) * v)
    Here 'q' is the encoded detection tokens and 'k' is an average of the exemplar tokens, 'v' isn't used.

    See original implementation:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/model_misc.py#L37
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int = 256,
        mlp_ratio: float = 8.0,
        clamp_bound: float = 12.0,
    ):
        # Inherit from parent
        super().__init__()

        # Main model components
        self.exemplar_mlp = MLP2LayersPostNorm(features_per_token, mlp_ratio)
        self.exemplar_proj = nn.Linear(features_per_token, features_per_token)
        self.detection_token_proj = nn.Linear(features_per_token, features_per_token)

        # Store values for use at runtime
        self.register_buffer("_scale", torch.tensor(1.0 / features_per_token).sqrt(), persistent=False)
        self._min_clamp = -clamp_bound
        self._max_clamp = clamp_bound

    # .................................................................................................................

    def forward(
        self,
        detection_tokens_bnc: Tensor,
        exemplar_tokens_bnc: Tensor,
        exemplar_mask_bn: Tensor | None = None,
    ) -> Tensor:
        """
        Returns:
            detection_confidence_scores (shape: BxN, B batches, N detections)
        """

        # Preprocess tokens before averaging
        exm_tokens_mlp_bnc = self.exemplar_mlp(exemplar_tokens_bnc)

        # Fill in missing mask before computing averaged exemplar token
        # -> Note: In this case, could just directly compute average with exm_tokens_mlp.mean(...)
        #    however, this leads to a major numerical difference (not sure why?) and ends
        #    up having a *substantial* negative impact on detection scores
        # -> For example: torch.allclose(tokens.sum(0)/num_tokens, tokens.mean(0)) comes out False!
        if exemplar_mask_bn is None:
            exm_b, exm_n, _ = exemplar_tokens_bnc.shape
            exemplar_mask_bn = torch.zeros((exm_b, exm_n), dtype=torch.bool, device=exemplar_tokens_bnc.device)
        exemplar_mask_bn = exemplar_mask_bn.to(dtype=torch.bool)
        inv_mask_bn1 = (~exemplar_mask_bn).to(exm_tokens_mlp_bnc).unsqueeze(-1)
        num_valid_exm_b1 = torch.clamp(inv_mask_bn1.sum(1), min=1.0)
        averaged_exm_bc = (exm_tokens_mlp_bnc * inv_mask_bn1).sum(dim=1) / num_valid_exm_b1

        # Do attention-like softmax(q*k*scale)
        det_tokens_proj_bnc = self.detection_token_proj(detection_tokens_bnc)
        avg_exm_proj_bc1 = self.exemplar_proj(averaged_exm_bc).unsqueeze(-1)
        scores_bn1 = torch.matmul(det_tokens_proj_bnc, avg_exm_proj_bc1) * self._scale
        return scores_bn1.squeeze(-1).clamp(min=self._min_clamp, max=self._max_clamp).sigmoid()

    # .................................................................................................................
