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


class RPEComplex(nn.Module):
    """
    Rotational-Position-Encoder which uses complex numbers.
    The encoding uses RoPE, which encodes the position of tokens
    using rotation offsets.

    This implementation represents the rotations using complex
    numbers (as in the original implementation), but it could
    also be done using 2D matrix operations for wider compatibility.

    This module doesn't exist in the original sam3 implementation, but
    is instead handled through 2 functions: 'compute_axial_cis'
    and 'apply_rotary_enc' (which themselves depend on other functions), see:
    https://github.com/facebookresearch/sam3/blob/2d1cbaeac7b52ca64baf61e58973d0940ae843d0/sam3/model/vitdet.py#L41
    https://github.com/facebookresearch/sam3/blob/2d1cbaeac7b52ca64baf61e58973d0940ae843d0/sam3/model/vitdet.py#L68

    Note that a separate (mostly identical) copy of these implementations elsewhere in the codebase:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/sam/rope.py#L24
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/sam/rope.py#L56

    Note: This module does not contain any learnable parameters!
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, theta: float = 10000.0, rope_hw: tuple[int, int] | None = None):

        # Inherit from parent
        super().__init__()

        # Store rope sizing (needed when re-computing rotation vectors)
        self._use_hw_scaling = rope_hw is not None
        if rope_hw is None:
            rope_hw = (None, None)
        self._rope_h = rope_hw[0]
        self._rope_w = rope_hw[1]

        # Pre-compute base angles used to construct rotation amounts
        powers = torch.linspace(0, 1, 1 + (features_per_token // 4))[:-1]
        base_angles = torch.pow(theta, -powers)
        self.register_buffer("base_angles", base_angles, persistent=False)

        # Allocate storage for caching results, so we don't re-compute position encodings repeatedly
        # (encodings don't change for a fixed sized input, e.g. video frames)
        self._cache_hw = (0, 0)
        self.register_buffer("clx_rotors_bdnc", torch.empty(1, 1, 1, 1), persistent=False)

    # .................................................................................................................

    def forward(self, q: Tensor, k: Tensor, q_tokens_hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """
        Applies rotational position encodings to both query & key tokens.
        Every pair of feature values of each token is interpreted as a
        2D vector, 2D-rotated by some amount based on the (x,y) position
        associated with the token, and then converted back to the original
        non-paired representation.

        For example, if a token comes is as:
            [1,2,3,4,5,6,7,8],
        it first gets broken into xy pairs:
            [[1,2], [3,4], [5,6], [7,8]]

        Then each of these gets rotated based on it's indexing.
        For the sake of convenience, let's say the rotation amount
        for these four indexes are: [0, 90degrees, 180degrees, 270degrees]
        Then the rotation result would be:
            [[1,2], [-4,3], [-5,-6], [8,-7]]
        The final output un-pairs these values:
            [1,2,-4,3,-5,-6,8,-7]

        Returns:
            q_out, k_out
        """

        # For clarity
        # -> Tokens assumed to have shape: BxDxNxC (B is batch, D is num heads, N is number tokens, C is channels)
        bq, dq, nq, cq = q.shape
        bk, dk, nk, ck = k.shape

        # Re-build rotation vectors if needed
        if self._cache_hw != q_tokens_hw:
            self.clx_rotors_bdnc = self.get_complex_rotors(q_tokens_hw)
            self._cache_hw = q_tokens_hw

        # Convert each consecutive feature value pairs into 'xy' format and rotate by 'rotvectors'
        # -> Features are converted to xy simply by stacking pairs of values in 'channel' dimension,
        #    for example, for a single token with features: [-10, 20, 5, -60, 32, 11, etc.]
        #    these get stacked in pairs to form 'xy' format for rotation: [(-10, 20), (5, -60), (32, 11), etc.]
        # -> Notice reshaping: BxDxNxC -> BxDxNx(C/2)x2,
        #    however, 'view_as_complex' gets rid of separate xy dimension giving just: BxDxNx(C/2)
        q_as_xy = torch.view_as_complex(q.float().reshape(bq, dq, nq, cq // 2, 2))
        q_out = torch.view_as_real(q_as_xy * self.clx_rotors_bdnc)

        # Do the same for key tokens, but must handle potential mismatching number of q vs k tokens
        k_as_xy = torch.view_as_complex(k.float().reshape(bk, dk, nk, ck // 2, 2))
        k_rot_amt = self.clx_rotors_bdnc if nk == nq else self.clx_rotors_bdnc.repeat(1, 1, nk // nq, 1)
        k_out = torch.view_as_real(k_as_xy * k_rot_amt)

        # Restore to original (non-xy pairing) shape: BxDxNx(C/2)x2 -> BxDxNxC
        q_out = q_out.to(q.device, q.dtype).flatten(3)
        k_out = k_out.to(q.device, q.dtype).flatten(3)
        return q_out, k_out

    # .................................................................................................................

    def get_complex_rotors(self, tokens_hw: tuple[int, int]) -> Tensor:
        """
        In the original implementation, this functionality is called 'compute_axial_cis', see:
        https://github.com/facebookresearch/sam3/blob/2d1cbaeac7b52ca64baf61e58973d0940ae843d0/sam3/model/vitdet.py#L41
        -> Not sure, but 'cis' probably refers to 'cos + i * sin'
        -> 'axial' seems like a strange name, but might refer to the fact that these are applied 'per-token'

        This function calculates a rotation amount for all possible (x,y) positions,
        which is how RoPE encodes positioning of tokens. The rotation amounts are
        represented using 2D unit vectors stored as complex numbers ('rotors').

        There are a few simple but uncommon math things being used here...
            1. Computes (x,y) indexing in a 1D format
               e.g. For a 2D matrix converted to 1D (by flattening)
                   [1  2]
                   [3, 4]  -> [1, 2, 3, 4, 5, 6]
                   [5, 6]
               The x-/column-indexing for the 1D sequence is: [0,1,0,1,0,1]
               The y-/row-indexing is: [0,0,1,1,2,2]
               These are called 'x_mults' and 'y_mults' in the code

            2. Multiplying vectors using an 'outer' product
               Given vector A: [1,2,3] and vector B: [1,10]
               The outer product gives a 2D matrix:
                   [1, 10]
                   [2, 20]
                   [3, 30]
               e.g. the rows hold values of A, with each column multiplied by values of B

            3. The use of torch.polar(...), which converts a distance & angle
               into a single complex number with a matching magnitude & phase.
               However, the magnitudes are always 1, so it's just being used
               to create 'rotated' complex numbers. Multiplying another
               complex number by one of these is equivalent to a simple 2D rotation.

        Returns:
            rotation_vectors (as complex numbers)
        """

        # For clarity
        h, w = tokens_hw
        x_mult_scale = self._rope_h / h if self._use_hw_scaling else 1.0
        y_mult_scale = self._rope_w / w if self._use_hw_scaling else 1.0
        device, dtype = self.base_angles.device, self.base_angles.dtype

        # Determine xy token indexing, in 1D, to use as angle multipliers
        # -> x_mults looks like: [0,1,2,0,1,2,0,1,2,0,1,2] (for w=3, h=4)
        # -> y_mults looks like: [0,0,0,1,1,1,2,2,2,3,3,3] (for w=3, h=4)
        # -> Except these may be multiplied by a scalar if the input sizing doesn't match rope sizing
        x_mults = torch.arange(w, device=device, dtype=dtype).repeat(h) * x_mult_scale
        y_mults = torch.arange(h, device=device, dtype=dtype).repeat_interleave(w) * y_mult_scale

        # Calculate angles as multiples of (pre-computed) base angles
        # -> Base angles have length equal to 1/4 of the features per token/head (64/4 = 16 by default)
        # -> x/y_mults have length equal to h*w = number of tokens
        # -> So result has shape: Nx(f/4)
        x_angles = torch.outer(x_mults, self.base_angles).float()
        y_angles = torch.outer(y_mults, self.base_angles).float()

        # Form final complex rotors (unit magnitude complex numbers)
        x_rotors = torch.polar(torch.ones_like(x_angles), x_angles)
        y_rotors = torch.polar(torch.ones_like(y_angles), y_angles)

        # The final output has 1D 'rows-of-tokens' shape: 1x1xNx(2*f/4)
        # -> This is expected to match the input tokens (with channels halved due to (x,y) formatting)
        return torch.cat([x_rotors, y_rotors], dim=-1).unsqueeze(0).unsqueeze(0)

    # .................................................................................................................


class RPEMatrix(nn.Module):
    """
    Rotational-Position-Encoder which uses 2x2 rotation matrices.
    This is meant as an alternative to the RPEComplex module.

    While using rotation matrices is conceptually simpler, it's far
    slower than using complex numbers...

    Note: This module does not contain any learnable parameters!
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, theta: float = 10000.0, rope_hw: tuple[int, int] | None = None):

        # Inherit from parent
        super().__init__()

        # Store rope sizing (needed when re-computing rotation vectors)
        self._use_hw_scaling = rope_hw is not None
        if rope_hw is None:
            rope_hw = (None, None)
        self._rope_h = rope_hw[0]
        self._rope_w = rope_hw[1]

        # Pre-compute base angles used to construct rotation amounts
        powers = torch.linspace(0, 1, 1 + (features_per_token // 4))[:-1]
        base_angles = torch.pow(theta, -powers)
        self.register_buffer("base_angles", base_angles, persistent=False)

        # Allocate storage for caching results, so we don't re-compute position encodings repeatedly
        # (encodings don't change for a fixed sized input, e.g. video frames)
        self._cache_hw = (0, 0)
        self.register_buffer("rotmats", torch.empty(1, 1, 1, 1, 1, 1), persistent=False)

    # .................................................................................................................

    def forward(self, q: Tensor, k: Tensor, q_tokens_hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """
        Applies rotational position encodings to both query & key tokens.
        Every pair of feature values of each token is interpreted as a
        2D vector, 2D-rotated by some amount based on the (x,y) position
        associated with the token, and then converted back to the original
        non-paired representation.

        Returns:
            q_out, k_out
        """

        # For clarity
        bq, dq, nq, cq = q.shape
        bk, dk, nk, ck = k.shape

        # Re-build rotations if needed
        if self._cache_hw != q_tokens_hw:
            self.rotmats = self.get_rotation_matrices(q_tokens_hw)
            self._cache_hw = q_tokens_hw

        # Convert each consecutive feature value pair into 'xy' format and rotate
        q_out = torch.matmul(self.rotmats, q.reshape(bq, dq, nq, cq // 2, 2, 1))

        # Convert key tokens and handle potential mismatched q vs. k sizing
        k_rotmat = self.rotmats if nk == nq else self.rotmats.repeat(1, 1, nk // nq, 1, 1, 1)
        k_out = torch.matmul(k_rotmat, k.reshape(bk, dk, nk, ck // 2, 2, 1))

        # Convert back to non-paired format for output
        q_out = q_out.flatten(3)
        k_out = k_out.flatten(3)
        return q_out, k_out

    # .................................................................................................................

    def get_rotation_matrices(self, tokens_hw: tuple[int, int]) -> Tensor:
        """
        This function is equivalent to a function called 'compute_axial_cis' from the original implementation:
        https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/vitdet.py#L41

        Unlike the original function, here we compute rotation matrices rather than
        their complex number equivalents. Each possible (x,y) position is represented
        by a 2x2 rotation matrix, where the rotation amount 'IS' the position encoding.

        Returns:
            rotation_matrices (shape: 1x1xNxDx2x2)
        """

        # For clarity
        h, w = tokens_hw
        x_mult_scale = self._rope_h / h if self._use_hw_scaling else 1.0
        y_mult_scale = self._rope_w / w if self._use_hw_scaling else 1.0
        device, dtype = self.base_angles.device, self.base_angles.dtype

        # Determine xy token indexing, in 1D, to use as angle multipliers
        # -> x_mults looks like: [0,1,2,0,1,2,0,1,2,0,1,2] (for w=3, h=4)
        # -> y_mults looks like: [0,0,0,1,1,1,2,2,2,3,3,3] (for w=3, h=4)
        # -> Except these may be multiplied by a scalar if the input sizing doesn't match rope sizing
        x_mults = torch.arange(w, device=device, dtype=dtype).repeat(h) * x_mult_scale
        y_mults = torch.arange(h, device=device, dtype=dtype).repeat_interleave(w) * y_mult_scale

        # Calculate angles as multiples of (pre-computed) base angles
        # -> Base angles have length equal to 1/4 of the features per token/head (64/4 = 16 by default)
        # -> x/y_mults have length equal to h*w = number of tokens
        # -> So result has shape: Nx(f/4)
        x_angles = torch.outer(x_mults, self.base_angles).float()
        y_angles = torch.outer(y_mults, self.base_angles).float()

        # Form final 2x2 rotation matrices
        x_rotmat = self.make_rotation_matrix(x_angles).to(dtype=dtype)
        y_rotmat = self.make_rotation_matrix(y_angles).to(dtype=dtype)

        # Combine x and y matrices into a single: 1x1xNxdx2x2 tensor
        # -> Extra leading 1's are for batch & head dimensions
        return torch.cat([x_rotmat, y_rotmat], dim=1).unsqueeze(0).unsqueeze(0)

    # .................................................................................................................

    @staticmethod
    def make_rotation_matrix(angles_tensor) -> Tensor:
        """Helper used to create a rotation matrix of shape: Nxdx2x2, assuming the given angles are of shape Nxd"""

        sin_term, cos_term = torch.sin(-angles_tensor), torch.cos(-angles_tensor)
        rotation_matrix = torch.stack(
            [
                torch.stack([cos_term, -sin_term], dim=-1),
                torch.stack([sin_term, cos_term], dim=-1),
            ],
            dim=-1,
        )

        return rotation_matrix

    # .................................................................................................................


class SinusoidalPE2D(nn.Module):
    """
    Implements position encodings similar to 'sine/cosine' approach originally
    used in paper: "Attention Is All You Need" (see page 5), but adapted
    to work with 'image-like' tokens.

    The math in this model is fairly non-sensical looking, it follows
    the formulas in the original paper:

        PE(pos, even-i) = sin(pos * f)
        PE(pos, odd-i)  = cos(pos * f)
        Where f = 1 / 10000 ^ (2*i / features-per-token)
        and 'i' is meant to be the feature index (so i/features-per-token is like a normalized index)

    This implementation uses the same 'f' values for the feature channels of all
    image tokens, but scales the values for each token based on the normalized
    x/y position of the token, prior to computing the sin(...) & cos(...) result.

    There are minor numerical differences between this version and the original,
    due to differences in caching and an 'division epsilon' that is omitted here.

    See original code:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/position_encoding.py#L10
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, temperature: float = 10000.0):

        # Inherit from parent
        super().__init__()

        # Sanity check. Need even number so we can make (x,y) pairings
        assert features_per_token % 2 == 0, "Need an even number of features for sinusoidal position encoding!"

        # Pre-compute frequencies with 'geometric scaling' (e.g. temperature ^ scaling)
        # -> These are of the form: '2πf' where 'f' is different for each feature index
        # -> There are C/2 'scale_factors' and they look like repeated-normalized values,
        #    for example, with features_per_token = 16, result is: [0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75]
        half_features = features_per_token // 2
        scale_factors = (torch.arange(half_features, dtype=torch.float32) // 2) * (2 / half_features)
        per_channel_period = temperature**scale_factors
        self.register_buffer("per_channel_frequency_factors", 2.0 * torch.pi / per_channel_period, persistent=False)

        # Also allocate storage for re-using encodings
        self.register_buffer("cached_posenc_bchw", torch.empty((1, 1, 1, 1)), persistent=False)

    # .................................................................................................................

    def forward(self, height: int, width: int) -> Tensor:

        # Re-generate cached result if needed
        cache_h, cache_w = self.cached_posenc_bchw.shape[-2:]
        if cache_h != height or cache_w != width in self.cache:

            # For convenience
            device, dtype = self.cached_posenc_bchw.device, self.cached_posenc_bchw.dtype

            # Make index sequence (e.g. sequences like: 1, 2, 3, 4, ..., height or width) that is then normalized
            # -> For a height of 5, result would be: [0.2, 0.4, 0.6, 0.8, 1.0]
            y_idx_norm = torch.arange(1, 1 + height, dtype=torch.float32, device=device).unsqueeze(-1) / height
            x_idx_norm = torch.arange(1, 1 + width, dtype=torch.float32, device=device).unsqueeze(-1) / width

            # Compute 'xy frequency' terms, these are of the form: α*2πf
            # where α is a normalized x-/y-index and f is pre-compute frequency value
            # -> Note the shapes here are: Hx(C/2) and Wx(C/2), where C is expected features-per-token
            y_frequencies = y_idx_norm * self.per_channel_frequency_factors
            x_frequencies = x_idx_norm * self.per_channel_frequency_factors

            # Calculate the sine of even index angles, cosine of odd index angles & stack together
            # -> Gives intermediate shapes: Hx(C/4)x2 & Wx(C/4)x2
            # -> Then flatten last 2 dimensions back to: Hx(C/2), Wx(C/2)
            y_sincos = torch.stack((y_frequencies[:, 0::2].sin(), y_frequencies[:, 1::2].cos()), dim=-1).flatten(-2)
            x_sincos = torch.stack((x_frequencies[:, 0::2].sin(), x_frequencies[:, 1::2].cos()), dim=-1).flatten(-2)

            # Repeat x/y components along h/w dimensions and stack to form a single BxCxHxW tensor
            x_sincos_hwc = x_sincos.unsqueeze(0).repeat(height, 1, 1)
            y_sincos_hwc = y_sincos.unsqueeze(1).repeat(1, width, 1)
            xy_stacked_bchw = torch.cat((y_sincos_hwc, x_sincos_hwc), dim=-1).permute(2, 0, 1).unsqueeze(0)
            self.cached_posenc_bchw = xy_stacked_bchw.to(dtype)

        return self.cached_posenc_bchw
