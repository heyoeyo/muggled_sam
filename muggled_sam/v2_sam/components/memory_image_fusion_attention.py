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


class RoPEAttention(nn.Module):
    """
    Slightly modified implementation of the 'RoPEAttention' model from:
        "SAM 2: Segment Anything in Images and Videos"
        By: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma,
        Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan,
        Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
        Piotr Dollár, Christoph Feichtenhofer
        @ https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

    This implementation slightly re-works the way the rotary position encodings
    are managed/applied, but is otherwise very similar to the original code, found here:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/sam/transformer.py#L289C7-L289C20

    For more information about RoPE, the original paper seem to be the following:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    By: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
    @ https://arxiv.org/abs/2104.09864
    """

    # .................................................................................................................

    def __init__(
        self,
        num_heads: int,
        features_per_token: int,
        features_per_kv_token: int | None = None,
        rope_theta=10000.0,
        use_complex_numbers=True,
    ):
        # Inherit from parent
        super().__init__()

        # Store config for re-use
        self.num_heads = num_heads
        self.features_per_head = features_per_token // num_heads
        features_per_kv_token = features_per_token if features_per_kv_token is None else features_per_kv_token

        # Mappings used to generate QKV vectors for attention calculations
        self.q_proj = nn.Linear(features_per_token, features_per_token)
        self.k_proj = nn.Linear(features_per_kv_token, features_per_token)
        self.v_proj = nn.Linear(features_per_kv_token, features_per_token)

        # Set up rotary position encoder
        PosEncoder = RPEComplexEncoder if use_complex_numbers else RPERealEncoder
        self.rotposenc = PosEncoder(self.features_per_head, rope_theta)

        # Output layer used to restore input feature count
        self.out_proj = nn.Linear(features_per_token, features_per_token)

    # .................................................................................................................

    def forward(
        self,
        q_tokens_hw: tuple[int, int],
        q: Tensor,
        k: Tensor,
        v: Tensor,
        num_final_k_to_exclude=0,
    ) -> Tensor:
        """
        Computes 'generic' attention between query/key/value tokens,
        while using rotary positional encodings. Can be used for either
        self or cross attention (self attention is when q, k, v are all the same).
        Inputs are expected to have shapes: BxNxF (i.e. 'rows of tokens' format)
        -> k & v must have the same N, q can be different

        Has support for excluding the final 'X' key tokens when applying
        positional encodings, which is meant to avoid assigning positioning
        info to non-positional tokens (e.g. object pointer tokens).
        Returns:
            encoded_query_tokens (same shape as q input)
        """

        # Compute QKV tokens & split into 'per-head' shape
        # -> Inputs have shape: BxNxF
        # -> Projection changes shape to: BxNxF' (F' matches the F for query tokens)
        # -> Reshape to get features per head: BxNxHxf (H is number of heads, f is features per head)
        # -> Transpose gives final shape: BxHxNxf
        batch_size_q, num_q = q.shape[0:2]
        batch_size_kv, num_k = k.shape[0:2]
        q = self.q_proj(q).reshape(batch_size_q, num_q, self.num_heads, self.features_per_head).transpose(1, 2)
        k = self.k_proj(k).reshape(batch_size_kv, num_k, self.num_heads, self.features_per_head).transpose(1, 2)
        v = self.v_proj(v).reshape(batch_size_kv, num_k, self.num_heads, self.features_per_head).transpose(1, 2)

        # Apply position encoding (shapes: BxHxNxF)
        num_k_keep = num_k - num_final_k_to_exclude
        q, k[:, :, :num_k_keep] = self.rotposenc(q_tokens_hw, q, k[:, :, :num_k_keep])

        # Attention
        # -> Original implementation has extra optimizations using: 'with torch.backends.cuda.sdp_kernel'
        attn = nn.functional.scaled_dot_product_attention(q, k, v)

        # Recombine per-head tokens and project back to input feature count
        # -> Tranpose converts shape: BxHxNqxf -> BxNqxHxf
        # -> Flatten merges per-head features to give shape: BxNqxF'
        # -> Output projection maps F' to F features giving output shape: BxNqxF
        enc_q_tokens = attn.transpose(1, 2).flatten(2)
        enc_q_tokens = self.out_proj(enc_q_tokens)

        return enc_q_tokens

    # .................................................................................................................


class RoPESelfAttention(nn.Module):
    """
    This module implements a self-attention model using RoPE attention
    along with a pre-norm layer and residual output
    """

    # .................................................................................................................

    def __init__(self, num_heads: int, features_per_token: int):
        super().__init__()
        self.attn = RoPEAttention(num_heads, features_per_token)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(self, a_tokens_hw: tuple[int, int], a_tokens: Tensor) -> Tensor:
        a_normed = self.norm(a_tokens)
        attn_result = self.attn(a_tokens_hw, a_normed, a_normed, a_normed)
        return a_tokens + attn_result

    # .................................................................................................................


class RoPECrossAttention(nn.Module):
    """
    This module implements a cross-attention model using RoPE attention
    It includes support for excluding some number of key tokens, which is
    needed by the fusion model to avoid adding position encodings to object pointers.

    This implementation is very specifically tailored to use within the
    memory fusion model, which does not use position encodings for
    the image (query) tokens!
    """

    # .................................................................................................................

    def __init__(self, num_heads: int, features_per_token: int, features_per_kv_token: int):
        super().__init__()
        self.attn = RoPEAttention(num_heads, features_per_token, features_per_kv_token)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(
        self,
        a_tokens_hw: tuple[int, int],
        a_tokens: Tensor,
        b_tokens: Tensor,
        b_posenc: Tensor,
        num_final_k_to_exclude=0,
    ) -> Tensor:
        a_normed = self.norm(a_tokens)
        b_embed = b_tokens + b_posenc
        attn_result = self.attn(a_tokens_hw, a_normed, b_embed, b_tokens, num_final_k_to_exclude)
        return a_tokens + attn_result

    # .................................................................................................................


class RPEComplexEncoder(nn.Module):
    """
    Positional encoder for RoPE, which encodes the position of tokens
    using rotation offsets. This implementation represents the rotations
    using complex numbers (as in the original implementation).

    This module doesn't exist in the original implementation, but
    is instead handled through 2 functions: 'compute_axial_cis'
    and 'apply_rotary_enc' (which themselves depend on other functions), see:
    https://github.com/facebookresearch/segment-anything-2/blob/main/sam2/modeling/position_encoding.py

    For some reason, the complex-number implementation is extraordinarily
    fast compared to the equivalent 2D rotation matrix implementation...

    Note: This module does not contain any learnable parameters!
    """

    # .................................................................................................................

    def __init__(self, features_per_token=256, theta=10000.0):

        # Inherit from parent
        super().__init__()

        # Pre-compute base angles used to construct rotation amounts
        powers = torch.linspace(0, 1, 1 + (features_per_token // 4))[:-1]
        base_angles = torch.pow(theta, -powers)
        self.register_buffer("base_angles", base_angles, persistent=False)

        # Allocate storage for caching results, so we don't re-compute position encodings repeatedly
        # (encodings don't change for a fixed sized input, e.g. video frames)
        self.register_buffer("rotvectors", torch.empty(1, 1, 1, 1), persistent=False)

    # .................................................................................................................

    def forward(self, q_tokens_hw: tuple[int, int], q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """
        Applies rotational position encodings to both query & key tokens.
        Every pair of feature values of each token is interpreted as a
        2D vector, 2D-rotated by some amount based on the (x,y) position
        associated with the token, and then converted back to the original
        non-paired representation.

        For example, if a token comes is as: [1,2,3,4,5,6,7,8],
        it first gets broken into xy pairs: [[1,2], [3,4], [5,6], [7,8]]
        Then each of these gets rotated based on it's position. For
        the sake of convenience, let's say the rotation amount for these
        4 'positions' are: [0, 90degrees, 180degrees, 270degrees]
        Then the rotation result would be:      [[1,2], [-4,3], [-5,-6], [8,-7]]
        The final output un-pairs these values: [1,2,-4,3,-5,-6,8,-7]

        Returns:
            q_out, k_out
        """

        # For clarity
        # -> Tokens assumed to have shape: Batch, Heads, NumTokens, Features/channels
        bq, hq, nq, cq = q.shape
        bk, hk, nk, ck = k.shape
        n_vec = self.rotvectors.shape[2]

        # Re-build rotation vectors if needed
        if n_vec != nq:
            self.rotvectors = self.get_rotation_vectors(q_tokens_hw)

        # Convert each consecutive feature value pair into 'xy' format and rotate
        q_as_xy = torch.view_as_complex(q.float().reshape(bq, hq, nq, cq // 2, 2))
        q_out = torch.view_as_real(q_as_xy * self.rotvectors)

        # Same for key tokens, but must handle potential mismatching number of q vs k tokens
        k_as_xy = torch.view_as_complex(k.float().reshape(bk, hk, nk, ck // 2, 2))
        k_rot_amt = self.rotvectors if nk == nq else self.rotvectors.repeat(1, 1, nk // nq, 1)
        k_out = torch.view_as_real(k_as_xy * k_rot_amt)

        # Restore to original (non-xy pairing) shape
        q_out = q_out.to(q.device, q.dtype).flatten(3)
        k_out = k_out.to(q.device, q.dtype).flatten(3)
        return q_out, k_out

    # .................................................................................................................

    def get_rotation_vectors(self, tokens_hw: tuple[int, int]) -> Tensor:
        """
        In the original implementation, this functionality is called 'compute_axial_cis', see:
        https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/position_encoding.py#L174

        This function calculates a rotation amount for all possible (x,y) positions,
        which is how RoPE encodes positioning of tokens. The rotation amounts are
        represented using 2D unit vectors stored as complex numbers.

        Returns:
            rotation_vectors (as complex numbers)
        """

        # For clarity
        h, w = tokens_hw
        device, dtype = self.base_angles.device, self.base_angles.dtype

        # Calculate multiples of base angles
        # -> Where sequence of multiples looks like: [0,1,2,0,1,2,0,1,2,0,1,2] (for w=3, h=4)
        x_mults = torch.arange(w, device=device, dtype=dtype).repeat(h)
        angles_x = torch.outer(x_mults, self.base_angles).float()
        rotvectors_x = torch.polar(torch.ones_like(angles_x), angles_x)

        # Calculate multiples of base angles
        # -> Where sequence of multiples looks like: [0,0,0,1,1,1,2,2,2,3,3,3] (for w=3, h=4)
        y_mults = torch.arange(h, device=device, dtype=dtype).repeat_interleave(w)
        angles_y = torch.outer(y_mults, self.base_angles).float()
        rotvectors_y = torch.polar(torch.ones_like(angles_y), angles_y)

        return torch.cat([rotvectors_x, rotvectors_y], dim=-1).unsqueeze(0).unsqueeze(0)

    # .................................................................................................................


class RPERealEncoder(nn.Module):
    """
    Alternate version of the RPEComplexEncoder, which doesn't use complex numbers.

    The basic idea is to perform complex multiplication using only real numbers.
    For example:
        Complex multiplication: (a + ib) * (c + id) = (a*c - b*d) + i(a*d + b*c)
    We can treat a,b,c,d as 4 real numbers and compute the result as:
        x_out = (a*c - b*d)
        y_out = (a*d + b*c)

    Compared to the 'complex' version, this implementation is very slightly
    slower, but should have better support across other run-times.

    It replaces an earlier implementation which used 2D rotation matrices
    but was significantly (>2x) slower, see:
    https://github.com/heyoeyo/muggled_sam/blob/f7ae0cdfa2de7e2432b4b0b4628a85e3103725aa/muggled_sam/v2_sam/components/memory_image_fusion_attention.py#L294

    Note: This module does not contain any learnable parameters!
    """

    # .................................................................................................................

    def __init__(self, features_per_token=256, theta=10000.0):

        # Inherit from parent
        super().__init__()

        # Pre-compute base angles used to construct rotation amounts
        # -> 'powers' are just evenly spaced values on interval [0,1), ex: [0.0, 0.25, 0.5, 0.75]
        quarter_features = features_per_token // 4
        powers = torch.arange(0, quarter_features) / quarter_features
        base_angles = 1.0 / torch.pow(theta, powers)  # WARNING: torch.pow(theta, -powers) is not the same numerically!
        self.register_buffer("base_angles", base_angles, persistent=False)

        # Allocate storage for caching results, so we don't re-compute position encodings repeatedly
        # (encodings don't change for a fixed sized input, e.g. video frames)
        self._cache_hw = (0, 0)
        self.register_buffer("x_rotors_bdnc", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer("y_rotors_bdnc", torch.empty(1, 1, 1, 1), persistent=False)

    # .................................................................................................................

    def forward(self, q_tokens_hw: tuple[int, int], q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
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
        # -> Tokens assumed to have shape: BxDxNxC (B is batch, D is num heads, N is number tokens, C is channels)
        bq, dq, nq, cq = q.shape
        bk, dk, nk, ck = k.shape

        # Re-build rotation vectors if needed
        if self._cache_hw != q_tokens_hw:
            self.x_rotors_bdnc, self.y_rotors_bdnc = self.get_real_rotors(q_tokens_hw)
            self._cache_hw = q_tokens_hw

        # Apply rotations to q tokens
        q_as_xy = q.reshape(bq, dq, nq, cq // 2, 2)
        q_x = q_as_xy[:, :, :, :, 0]
        q_y = q_as_xy[:, :, :, :, 1]
        q_out_x = (q_x * self.x_rotors_bdnc) - (q_y * self.y_rotors_bdnc)
        q_out_y = (q_x * self.y_rotors_bdnc) + (q_y * self.x_rotors_bdnc)
        q_out = torch.stack((q_out_x, q_out_y), dim=-1).flatten(3)

        # Apply rotations to k tokens
        k_as_xy = k.reshape(bk, dk, nk, ck // 2, 2)
        k_x = k_as_xy[:, :, :, :, 0]
        k_y = k_as_xy[:, :, :, :, 1]
        x_rotors = self.x_rotors_bdnc if nk == nq else self.x_rotors_bdnc.repeat(1, 1, nk // nq, 1)
        y_rotors = self.y_rotors_bdnc if nk == nq else self.y_rotors_bdnc.repeat(1, 1, nk // nq, 1)
        k_out_x = (k_x * x_rotors) - (k_y * y_rotors)
        k_out_y = (k_x * y_rotors) + (k_y * x_rotors)
        k_out = torch.stack((k_out_x, k_out_y), dim=-1).flatten(3)

        return q_out, k_out

    # .................................................................................................................

    def get_real_rotors(self, tokens_hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """
        See the RPEComplexEncoder 'get_rotation_vectors' for more info.
        In this implementation, rotors are held as separate x & y tensors.
        These correspond to the real & imaginary parts
        of the rotors from the original implementation.

        Returns:
            x_rotors_bdnc, y_rotors_bdnc
        """

        # For clarity
        h, w = tokens_hw
        device, dtype = self.base_angles.device, self.base_angles.dtype

        # Determine xy token indexing, in 1D, to use as angle multipliers
        # -> x_mults looks like: [0,1,2,0,1,2,0,1,2,0,1,2] (for w=3, h=4)
        # -> y_mults looks like: [0,0,0,1,1,1,2,2,2,3,3,3] (for w=3, h=4)
        x_mults = torch.arange(w, device=device, dtype=dtype).repeat(h)
        y_mults = torch.arange(h, device=device, dtype=dtype).repeat_interleave(w)

        # Calculate angles as multiples of (pre-computed) base angles
        # -> Base angles have length equal to 1/4 of the features per token/head (256/4 = 64 by default)
        # -> x/y_mults have length equal to h*w = number of tokens
        # -> So result has shape: Nx(f/4)
        x_angles = torch.outer(x_mults, self.base_angles)
        y_angles = torch.outer(y_mults, self.base_angles)

        # Form final 'rows-of-tokens' output with batch and 'heads' dimensions
        # -> Each rotor has shape: 1x1xNx(f/2)
        x_rotors = torch.cat((torch.cos(x_angles), torch.cos(y_angles)), dim=-1).unsqueeze(0).unsqueeze(0)
        y_rotors = torch.cat((torch.sin(x_angles), torch.sin(y_angles)), dim=-1).unsqueeze(0).unsqueeze(0)
        return x_rotors, y_rotors

    # .................................................................................................................
