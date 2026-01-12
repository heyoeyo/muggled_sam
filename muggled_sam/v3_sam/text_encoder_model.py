#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.text_tokenizer import TextTokenizer

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3TextEncoder(nn.Module):
    """
    Simplified implementation of the 'VETextEncoder' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This model is responsible for converting text, in the form of a string, into
    high-dimensional tokens for use in combination with sampling and image tokens.

    See original implementation:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/text_encoder_ve.py#L253
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_text_embedding: int = 1024,
        output_features_per_token: int = 256,
        num_layers: int = 24,
        num_heads: int = 16,
        tokenization_vocab_size: int = 49408,
        context_length: int = 32,
    ):
        # Inherit from parent
        super().__init__()

        # Set up model components
        self.tokenizer = TextTokenizer(tokenization_vocab_size, context_length)
        self.text_token_embeddings = nn.Embedding(tokenization_vocab_size, features_per_text_embedding)
        self.text_encoder = TextTransformer(features_per_text_embedding, num_layers, num_heads, context_length)
        self.text_proj = nn.Linear(features_per_text_embedding, output_features_per_token)

        # Store for padding (for numerical compatibility with original code)
        self._max_context_length = context_length

    # .................................................................................................................

    def forward(self, text: str, pad_to_context_length: bool = False) -> Tensor:
        """
        Converts a string of text into a token sequence.

        In the original implementation, text is padded to a maximum context
        length. This padding can cause minor numerical differences in the
        encoding (compared to not using it). In this implementation, the
        outputs are never padded, however the 'pad_to_context_length' flag
        can be used to enable padding internally to help maintain better
        numerical consistency with the original implementation. If enabled,
        the padding will be stripped on final output.

        Returns:
            encoded_text_tokens (Shape 1xNxC, N tokens, C channels)
            -> The batch size will always be 1, but is added to build batches easier
            -> N will typically be at least (2 + number of input words)
            -> Channel count is expected to match image tokens (256 by default)
        """

        # Convert text to sequence of vocabulary indices
        idx_tokens_bn = self.tokenizer.text_to_vocab_index(text)
        idx_b, idx_n = idx_tokens_bn.shape

        # Pad with 0-index entries if needed (used to get exact numerical match with original implementation)
        num_pad_tokens = 0
        if pad_to_context_length:
            num_pad_tokens = max(0, self._max_context_length - idx_n)
            padding_tokens_bn = torch.zeros((idx_b, num_pad_tokens)).to(idx_tokens_bn)
            idx_tokens_bn = torch.cat((idx_tokens_bn, padding_tokens_bn), dim=1)

        # Convert vocab index (integers) to embeddings (floats) and further encode
        encoded_text_bnc = self.text_token_embeddings(idx_tokens_bn)
        encoded_text_bnc = self.text_encoder(encoded_text_bnc)
        encoded_text_bnc = self.text_proj(encoded_text_bnc)

        # Strip padding from tokens if used
        # -> For some reason, this has to be done after projection or we can get numerical differences...
        if num_pad_tokens > 0:
            encoded_text_bnc = encoded_text_bnc[:, :idx_n, :]

        return encoded_text_bnc

    # .................................................................................................................

    def load_bpe_vocab(self, bpe_vocab_path: str):
        """Load BPE vocabulary, needed to perform tokenization of text inputs"""
        self.tokenizer.load_bpe_vocab(bpe_vocab_path)
        return self

    # .................................................................................................................


class TextTransformer(nn.Module):
    """
    Simplified implementation of the 'TextTransformer' from SAMv3.

    This model is responsible for further encoding initial text embeddings.
    Intuitively speaking, this model is responsible for making sense of text
    as a whole, rather than as a sequence of individual definitions.

    For example, the word 'ran' generally implies running (in past-tense),
    while the word 'out' implies something like 'outside rather than inside',
    however in the phrase 'ran out of time', these words mean something different.
    This model is responsible for updating the meaning of the text (tokens) within
    the context of the whole phrase.

    See original implementation:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/text_encoder_ve.py#L164
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_text_embedding: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        max_context_length: int = 32,
        mlp_ratio: float = 4.0,
    ):
        # Inherit from parent
        super().__init__()

        # Build model components
        self.posenc = nn.Parameter(torch.empty(max_context_length, features_per_text_embedding))
        self.blocks = nn.ModuleList(
            TransformerBlock(features_per_text_embedding, num_heads, mlp_ratio) for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(features_per_text_embedding)

    # .................................................................................................................

    def forward(self, text_embeddings_bnc: Tensor) -> Tensor:
        """Encodes text embeddings with self-attention. Returns: encoded_text_embeddings (same shape as input)"""

        # For convenience
        _, num_tokens, _ = text_embeddings_bnc.shape
        device, dtype = text_embeddings_bnc.device, text_embeddings_bnc.dtype

        # Handle position encoding, with support for extending to sequences longer than learned length
        additive_posenc = self.posenc[:num_tokens]
        if num_tokens > self.posenc.shape[0]:
            additive_posenc = nn.functional.interpolate(
                self.posenc.transpose(0, 1).unsqueeze(0),  # Convert from NxC -> 1xCxN (to scale N)
                size=num_tokens,
                mode="linear",
            )
            additive_posenc = additive_posenc.squeeze(0).transpose(0, 1)  # Convert back: 1xCxN -> NxC
        text_embedding = text_embeddings_bnc + additive_posenc

        # Run attention blocks with causal mask (prevents attention between early-to-later words)
        # For N = 3, mask looks like:
        #   [0, -inf, -inf]
        #   [0,    0, -inf]
        #   [0,    0,    0]
        casual_mask = torch.full((num_tokens, num_tokens), -torch.inf, device=device, dtype=dtype)
        casual_mask = torch.triu(casual_mask, diagonal=1)
        for block in self.blocks:
            text_embedding = block(text_embedding, casual_mask)

        return self.out_norm(text_embedding)

    # .................................................................................................................


class TransformerBlock(nn.Module):
    """A simple self-attention transformer block"""

    # .................................................................................................................

    def __init__(self, features_per_token: int = 1024, num_heads: int = 16, mlp_ratio: float = 4.0):

        # Inherit from parent
        super().__init__()

        # Attention components
        self.norm_preattn = nn.LayerNorm(features_per_token)
        self.attn = nn.MultiheadAttention(features_per_token, num_heads, batch_first=True)

        # MLP components
        num_hidden_features = int(round(features_per_token * mlp_ratio))
        self.norm_premlp = nn.LayerNorm(features_per_token)
        self.mlp = nn.Sequential(
            nn.Linear(features_per_token, num_hidden_features),
            nn.GELU(),
            nn.Linear(num_hidden_features, features_per_token),
        )

    def forward(self, tokens_bnc: Tensor, attn_mask: Tensor | None = None) -> Tensor:

        # Attention
        tokens_normed = self.norm_preattn(tokens_bnc)
        attn_result, _ = self.attn(tokens_normed, tokens_normed, tokens_normed, need_weights=False, attn_mask=attn_mask)
        attn_result = tokens_bnc + attn_result

        # MLP
        output = self.norm_premlp(attn_result)
        output = self.mlp(output)
        output = attn_result + output

        return output

    # .................................................................................................................
