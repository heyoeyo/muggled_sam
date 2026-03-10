#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def polar_loss(
    target_tensor: Tensor,
    predicted_tensor: Tensor,
    channel_dim: int = 1,
    scale_weight: float = 1e-1,
):

    # Get L2 norms for normalizing image tokens
    targ_mag = target_tensor.norm(p=2, dim=channel_dim, keepdim=True)
    pred_mag = predicted_tensor.norm(p=2, dim=channel_dim, keepdim=True)

    # Compute scale & angular error between target/prediction
    scale_log_error = (torch.log1p(targ_mag) - torch.log1p(pred_mag)) ** 2
    angle_error = ((target_tensor / targ_mag) - (predicted_tensor / pred_mag)) ** 2

    return (scale_log_error * scale_weight + angle_error).mean()


def outlier_loss(target_tensor, predicted_tensor, channel_dim: int = 1):
    delta = target_tensor - predicted_tensor
    del_max, _ = delta.max(channel_dim)
    del_min, _ = delta.min(channel_dim)
    return (del_max.abs() + del_min.abs()).mean()


def angle_loss(target_tensor, predicted_tensor, channel_dim: int = 1):

    # Numerically stable version of cosine similarity (which tends to fail on f16)
    targ_norm = torch.nn.functional.normalize(target_tensor, p=2, dim=channel_dim)
    pred_norm = torch.nn.functional.normalize(predicted_tensor, p=2, dim=channel_dim)
    angle_error = ((targ_norm - pred_norm) ** 2).sum(dim=channel_dim)
    return angle_error.mean()


def mse_loss(target_tensor, predicted_tensor, channel_dim: int = 1):
    return (target_tensor - predicted_tensor).square().mean()


def l1_loss(target_tensor, predicted_tensor, channel_dim: int = 1):
    return (target_tensor - predicted_tensor).abs().mean()


def scale_loss(target_tensor, predicted_tensor, channel_dim: int = 1):
    targ_mag = target_tensor.norm(dim=channel_dim, keepdim=True)
    pred_mag = predicted_tensor.norm(dim=channel_dim, keepdim=True)
    return l1_loss(targ_mag, pred_mag)
