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


class DecomposedRelativePositionEncoder(nn.Module):
    """
    Simplified implementation of the additive 'decomposed relative position encodings' from:
        "Segment Anything"
        By: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
            Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
        @ https://arxiv.org/abs/2304.02643

    Arguably the most complicated component of SAM!
    This model is responsible for creating an additive position encoding which is meant
    to be added inside the attention calculation, just after the Q * K operation
    (see "Attention is all you need"). For the sake of optimization, the position encodings
    are separated into horizontal and vertical components which are added together. This is
    where the 'decomposed' name comes from. The idea itself seems to originate from the paper (see page 3):
        "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
        By: Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong,
            Jitendra Malik, Christoph Feichtenhofer
        @ https://arxiv.org/abs/2112.01526

    The original MViT2 code can be found here:
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py#L45

    The original SAM implementation does not have a dedicated module for managing these encodings,
    instead the learned encodings are stored in the 'Attention' module and the functionality is
    handled by a separate function. This implementation combines all related functionality/data
    into a single class and removes some of the (unused) flexibility of the original code,
    while adding extra documentation.

    Original SAM implementation:
    https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/image_encoder.py#L325
    """

    # .................................................................................................................

    def __init__(self, features_per_head, base_patch_grid_hw):

        # Inherit from parent
        super().__init__()

        # Figure out how many possible 'relative positions' in H & W we could have
        # -> For example, for a grid with a width of 4, the furthest possible points
        #    could be +/- 3 cells away from each other. Therefore the possible
        #    relative distances would be: -3, -2, -1, 0, 1, 2, 3,
        #    for a total of 7 (more generally: 2 * w - 1) possible indices
        base_h, base_w = base_patch_grid_hw
        num_h_idxs = (2 * base_h) - 1
        num_w_idxs = (2 * base_w) - 1

        # Set up storage for (learned) 'decomposed' relative position embeddings
        # -> The ordering of these embeddings assumes the 0th entry corresponds to the
        #    most 'negative' relative difference (in H or W), the last entry is the
        #    most 'positive' relative difference and the middle entry, which
        #    is always guaranteed by odd-length sizing, is the 'no difference'
        #    relative position (i.e. when token pairs occupy the same H/W position)
        self.relpos_h_1d = nn.Parameter(torch.zeros(num_h_idxs, features_per_head))
        self.relpos_w_1d = nn.Parameter(torch.zeros(num_w_idxs, features_per_head))

        # Store the base sizing, so we know what grid size was used for the learned embeddings
        self.base_h = base_h
        self.base_w = base_w

    # .................................................................................................................

    def forward(self, query_tokens: Tensor, patch_grid_hw: tuple[int, int]) -> Tensor:
        """
        Computes the relative position encoding matrix to be added to the attention tensor.
        Accounts for potential resizing, if image patch size is different from learned encodings.
        Returns the position encodings matching the row/column count of the attention tensor.
        """

        # For convenience
        b, n, c = query_tokens.shape
        h, w = patch_grid_hw
        device = query_tokens.device

        # Check if we need to re-size the relative position encodings
        rpos_h_1d = self.relpos_h_1d
        rpos_w_1d = self.relpos_w_1d
        if h != self.base_h:
            rpos_h_1d = self._scale_relpos_1d(self.relpos_h_1d, h)
        if w != self.base_w:
            rpos_w_1d = self._scale_relpos_1d(self.relpos_w_1d, w)

        # Generate list of all possible H & W 'positions' (or indicies)
        # -> For example, for a 2x4 tensor, the H indices are [0,1], W indices are [0,1,2,3]
        all_h_idx = torch.arange(h, dtype=torch.int64, device=device)
        all_w_idx = torch.arange(w, dtype=torch.int64, device=device)

        # Form 2D indexing 'deltas matrix', which stores an index corresponding to the relative
        # (horizontal or vertical) positioning of all possible pairs of tokens.
        # Simple example: For a tensor with a height of 4, the H deltas matrix is 4x4 and looks like:
        #         ┌              ┐                 ┌              ┐
        #         │  0 -1 -2 -3  │                 │  3  2  1  0  │
        #         │  1  0 -1 -2  │     Offset      │  4  3  2  1  │
        #         │  2  1  0 -1  │     ─────>      │  5  4  3  2  │
        #         │  3  2  1  0  │ (positive only) │  6  5  4  3  │
        #         └              ┘                 └              ┘
        # Each entry can be thought of as taking the row index and subtracting
        # the column index (see the left depiction above). These values are
        # meant for indexing, so they're offset to give only positive values.
        # Each index represents a relative positioning between query (row index)
        # and key (column index) tokens, but only considering the y/height dimension.
        # We repeat this for x/width as well.
        rpos_h_deltas_2d = (all_h_idx[:, None] - all_h_idx[None, :]) + (h - 1)
        rpos_w_deltas_2d = (all_w_idx[:, None] - all_w_idx[None, :]) + (w - 1)

        # Use deltas to index into the (learned) H & W relative position embeddings
        rpos_hhc = rpos_h_1d[rpos_h_deltas_2d]
        rpos_wwc = rpos_w_1d[rpos_w_deltas_2d]

        # Multiply query tokens into position embeddings, as per MViT2 implementation
        # -> This seems unusual to do and is very confusing to follow!
        # -> For every single query token, we take the dot product of the token with
        #    all relative position embeddings associated with the row of that token
        #    (as given by the H-deltas/embedding matrix above), which gives a total
        #    of 'H' scalar values. We do the same for position embeddings associated
        #    with the column of the token (using the W-deltas) to produce another
        #    'W' scalar values. So each query token produces (H + W) scalar values
        q_imglike = query_tokens.reshape(b, h, w, c)
        rpos_h_bnhw = torch.einsum("bhwc,hjc->bhwj", q_imglike, rpos_hhc).reshape(b, n, h, 1)
        rpos_w_bnhw = torch.einsum("bhwc,wic->bhwi", q_imglike, rpos_wwc).reshape(b, n, 1, w)

        # Create decomposed relative position encodings and reshape to match attention matrix
        # -> We only have (H + W) encodings, but there are (H * W = N) query-to-key results in the attention matrix
        # -> So the 'H' encodings are repeated for all columns, 'W' encodings are repeated for all rows,
        #    this is the idea behind 'decomposed' position encodings!
        add_relpos_bnhw = rpos_h_bnhw + rpos_w_bnhw
        return add_relpos_bnhw.reshape(b, n, n)

    # .................................................................................................................

    def _scale_relpos_1d(self, base_embedding_1d: Tensor, new_side_length: int) -> Tensor:
        """
        Helper used to make sure the position embeddings are scaled
        to match the input patch sizing, by linear interpolation.

        This is a bit like taking a list: [10, 20, 30]
        and interpolating to a different size: [10, 15, 20, 25, 30]
        Except that each element in the 'list' in this case is a full
        embedding vector, so that many values need to be interpolated
        between each item in the list.

        The interpolation does NOT use corner alignment, which is strange,
        but is implemented this way to match the original implementation,
        which can be found here:
        https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/image_encoder.py#L308
        """

        # Force embedding to float32 for computing interpolation
        # -> If we don't do this, we could get bad results/errors on lower precision dtypes
        orig_dtype = base_embedding_1d.dtype
        embed_f32 = base_embedding_1d.float()

        # Convert to shape needed by interpolation function and then convert back
        # -> Original shape is (num indexes, features per index)
        # -> Interpolation needs 3D shape: (1, features per index, num indexes)
        new_num_idxs = (2 * new_side_length) - 1
        resized_embedding = (
            nn.functional.interpolate(
                embed_f32.permute(1, 0).unsqueeze(0),
                size=new_num_idxs,
                mode="linear",
            )
            .squeeze(0)
            .permute(1, 0)
        )

        return resized_embedding.to(orig_dtype)
