#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

# For type hints
from torch import Tensor
from torch.nn import Module
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def get_token_hw(image_encoding: list[Tensor]) -> tuple[int, int]:
    """
    Helper used to figure out the token height & width of a SAM image encoding.
    SAMv1 & v2 are both set up to produce image encodings which are lists of tensors,
    SAMv3 & v3.1 produce lists-of-lists-of-tensors. This function takes care of
    figuring out the lowest-resolution token sizing, regardless of which SAM version is used.

    Returns:
        token_hw (height and width)
    """
    lowres_enc_bchw = image_encoding[0]
    if not isinstance(lowres_enc_bchw, Tensor):
        lowres_enc_bchw = image_encoding[0][0]
    return tuple(lowres_enc_bchw.shape[-2:])


def get_preencoding_hw(
    model: Module,
    image_bgr: ndarray,
    max_side_length: int | None = None,
    use_square_sizing: bool = True,
) -> tuple[int, int]:
    """
    Helper used to figure out the scaling the given SAM model would apply to
    the given image, prior to computing the image encoding. This scaling is
    generally needed to make the input image compatible with the sizing of
    the model patch embedding step (and possibly other processing constraints)

    This function relies on 'reaching into' the model, which is typically error
    prone. In case this cannot be done, the model will try to return (1234,1234)

    Returns:
        preencode_hw (height and width)
    """
    # Sanity check
    assert hasattr(model, "image_encoder"), "Error! Expecting model to have an 'image_encoder'"

    # Try to use the built-in image encoder prep function to figure out pre-encoding size
    preencode_hw = (1234, 1234)
    if hasattr(model.image_encoder, "prepare_image"):
        img_tensor_bchw = model.image_encoder.prepare_image(image_bgr, max_side_length, use_square_sizing)
        preencode_hw = tuple(img_tensor_bchw.shape[2:])

    return preencode_hw
