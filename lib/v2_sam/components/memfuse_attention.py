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
        use_matrix_encoder=False,
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
        PosEncoder = RPERotmatEncoder if use_matrix_encoder else RPEComplexEncoder
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


class RPERotmatEncoder(nn.Module):
    """
    Alternate positional encoder for RoPE, which encodes the position of tokens
    using rotation offsets. This implementation represents the rotations
    using 2x2 rotation matrices (unlike the original implementation).

    While using rotation matrices is in many ways simpler, it is far slower than
    using complex numbers, I'm not sure why...

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
        self.register_buffer("rotmats", torch.empty(1, 1, 1, 1, 1, 1), persistent=False)

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
        bq, hq, nq, cq = q.shape
        bk, hk, nk, ck = k.shape
        n_mat = self.rotmats.shape[2]

        # Re-build rotation matrix if needed
        if n_mat != nq:
            self.rotmats = self.get_rotation_matrices(q_tokens_hw)

        # Convert each consecutive feature value pair into 'xy' format and rotate
        q_out = torch.matmul(self.rotmats, q.reshape(bq, hq, nq, cq // 2, 2, 1))

        # Convert key tokens and handle potential mismatched q vs. k sizing
        k_rotmat = self.rotmats if nk == nq else self.rotmats.repeat(1, 1, nk // nq, 1, 1, 1)
        k_out = torch.matmul(k_rotmat, k.reshape(bk, hk, nk, ck // 2, 2, 1))

        # Convert back to non-paired format for output
        q_out = q_out.flatten(3)
        k_out = k_out.flatten(3)
        return q_out, k_out

    # .................................................................................................................

    def get_rotation_matrices(self, tokens_hw: tuple[int, int]) -> Tensor:
        """
        This function is equivalent to a function called 'compute_axial_cis' from the original implementation:
        https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/position_encoding.py#L174

        Unlike the original function, here we compute rotation matrices rather than
        their complex number equivalents. Each possible (x,y) position is represented
        by a 2x2 rotation matrix, where the rotation amount 'IS' the position encoding.

        Returns:
            rotation_matrices (shape: 1x1xNxDx2x2)
        """

        # For clarity
        h, w = tokens_hw
        device, dtype = self.base_angles.device, self.base_angles.dtype

        # Calculate rotation matrix using multiples of base angles
        # -> Where sequence of multiples looks like: [0,1,2,0,1,2,0,1,2,0,1,2] (for w=3, h=4)
        x_mults = torch.arange(w, device=device, dtype=dtype).repeat(h)
        angles_x = torch.outer(x_mults, self.base_angles)
        rotmat_x = self.make_rotation_matrix(angles_x)

        # Calculate rotation matrix using multiples of base angles
        # -> Where sequence of multiples looks like: [0,0,0,1,1,1,2,2,2,3,3,3] (for w=3, h=4)
        y_mults = torch.arange(h, device=device, dtype=dtype).repeat_interleave(w)
        angles_y = torch.outer(y_mults, self.base_angles)
        rotmat_y = self.make_rotation_matrix(angles_y)

        # Combine x and y matrices into a single: 1x1xNxdx2x2 tensor
        # -> Extra leading 1's are for batch & head dimensions
        return torch.cat([rotmat_x, rotmat_y], dim=1).unsqueeze(0).unsqueeze(0)

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
