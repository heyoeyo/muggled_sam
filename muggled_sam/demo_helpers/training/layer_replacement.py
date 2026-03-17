#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Loras


class LoraLinear(nn.Module):
    """
    Layer used for training linear layers. This is based on the paper:
        "LoRA: Low-Rank Adaptation of Large Language Models"
        By: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
        @ https://arxiv.org/abs/2106.09685

    The idea is to think of updating linear weights as:
        W' = W + ΔW
    Where we only update/train ΔW. If we construct ΔW from two much smaller matrices:
        ΔW = B*A
    Then we can dramatically reduce the number of parameters that need to be trained.
    For example, if W has size: 1000x2000, we can use a B of size 1000x1 and A of size 1x2000,
    then ΔW will have size 1000x2000 (2 million parameters!) but we only update B and A
    which have a total of 3000 parameters.
    """

    # .................................................................................................................

    def __init__(
        self,
        base_linear_layer: nn.Linear,
        rank: int = 1,
        force_no_grad: bool = True,
    ):

        # Inherit from parent
        super().__init__()

        # Make sure base layer isn't trainable
        if force_no_grad:
            base_linear_layer.requires_grad_(False)

        # Figure out config of input linear layer
        device, dtype = base_linear_layer.weight.device, base_linear_layer.weight.dtype
        out_features, in_features = base_linear_layer.weight.shape
        has_bias = base_linear_layer.bias is not None

        # Make lora matching base layer sizing
        lora_layers = self._make_lora(in_features, out_features, has_bias, rank, device, dtype)

        # Store components for forward calls
        self.base = base_linear_layer
        self.lora = lora_layers
        self.reset_weights()

    # .................................................................................................................

    @staticmethod
    def _make_lora(
        in_features: int,
        out_features: int,
        include_bias: bool,
        rank: int,
        device: str = "cpu",
        dtype: torch.device = torch.float32,
    ):
        lora_layers = nn.Sequential()
        lora_layers.add_module("A", nn.Linear(in_features, rank, bias=False))
        lora_layers.add_module("B", nn.Linear(rank, out_features, bias=include_bias))
        lora_layers.to(device=device, dtype=dtype)
        return lora_layers

    # .................................................................................................................

    def forward(self, tokens: Tensor) -> Tensor:
        return self.base(tokens) + self.lora(tokens)

    # .................................................................................................................

    def reset_weights(self, use_zeroed_state: bool = True, non_zero_scale: float = 0.1) -> None:
        has_bias = self.base.bias is not None
        nn.init.kaiming_uniform_(self.lora.A.weight, nonlinearity="linear")
        nn.init.zeros_(self.lora.B.weight)
        if has_bias:
            nn.init.zeros_(self.lora.B.bias)

        if not use_zeroed_state:
            nn.init.uniform_(self.lora.B.weight, -non_zero_scale, non_zero_scale)
            if has_bias:
                nn.init.uniform_(self.lora.B.bias, -non_zero_scale, non_zero_scale)
        return

    def record_weights(self, move_to_cpu: bool = True) -> dict[str, Tensor]:
        if move_to_cpu:
            return {name: weight.clone().float().cpu() for name, weight in self.lora.state_dict().items()}
        return {name: weight.clone() for name, weight in self.lora.state_dict().items()}

    def load_weights(self, state_dict) -> bool:

        # Wipe out any training gradients, since they shouldn't apply to loaded weights
        need_resize_layers = False
        self.lora.zero_grad()

        with torch.no_grad():

            try:
                # Try to load weights directly (this works if weights were originally saved from this module)
                self.lora.load_state_dict(state_dict)
            except RuntimeError:
                # -> Error happens if the loaded weights don't match the sizing (e.g. rank) of the existing module
                need_resize_layers = True

                # Try to read out the weights we expect to be loading
                a_weight = state_dict.get("A.weight", None)
                b_weight = state_dict.get("B.weight", None)
                b_bias = state_dict.get("B.bias", None)
                has_bias = b_bias is not None
                assert a_weight is not None, f"Unable to load target 'A.weight' (Got: {state_dict.keys()})"
                assert b_weight is not None, f"Unable to load target 'B.weight' (Got: {state_dict.keys()})"

                # Get loaded weight sizing and check that rank makes sense
                a_device, a_dtype = a_weight.device, a_weight.dtype
                a_rank, in_feats = a_weight.shape
                out_feats, b_rank = b_weight.shape
                assert a_rank == b_rank, f"A and B weights have mismatching rank ({a_rank} vs. {b_rank})"

                # Make sure loaded weights match the base weight sizing
                base_out_feats, base_in_feats = self.base.weight.shape
                assert in_feats == base_in_feats, f"In feature mismatch ({in_feats} vs. {base_in_feats})"
                assert out_feats == base_out_feats, f"Out feature mismatch ({out_feats} vs. {base_out_feats})"

                # Replace with lora sized to loaded weights
                del self.lora
                self.lora = self._make_lora(in_feats, out_feats, has_bias, a_rank, a_device, a_dtype)
                self.lora.load_state_dict(state_dict)

            # Make sure device/dtype of lora matches base layer
            device, dtype = self.base.weight.device, self.base.weight.dtype
            self.lora.to(device=device, dtype=dtype)

        return need_resize_layers

    @property
    def weight(self) -> Tensor:
        return self.base.weight + torch.matmul(self.lora.B.weight, self.lora.A.weight)

    @property
    def bias(self) -> Tensor or None:
        if self.base.bias is not None:
            return self.base.bias + self.lora.B.bias
        return self.base.bias

    # .................................................................................................................


class LoraConv2D(nn.Module):
    """
    Layer used for training 2D convolutional layers.
    Like the linear case, the idea is to think of updating conv weights as:
        W' = W + ΔW

    However, the way this works with convolution is to do a 'normal' 2D convolution,
    but with far fewer output channels (the number of channels is the lora 'rank'),
    and then follow this with a 1x1 convolution (which is the same as a linear layer)
    acting on the low-rank channels as input to reproduce the original output channel count,
    and then finally add this to the original convolution layer result to get the additive update effect.
    """

    # .................................................................................................................

    def __init__(
        self,
        base_conv_layer: nn.Conv2d,
        rank: int = 1,
        force_no_grad: bool = True,
    ):

        # Inherit from parent
        super().__init__()

        # Make sure base layer isn't trainable
        if force_no_grad:
            base_conv_layer.requires_grad_(False)

        # Store components for forward calls
        self.base = base_conv_layer
        self.lora = self._make_lora(base_conv_layer, rank)
        self.reset_weights()

    # .................................................................................................................

    @staticmethod
    def _make_lora(
        base_conv_layer: nn.Conv2d,
        rank: int,
    ):
        # Figure out config of input layer
        device, dtype = base_conv_layer.weight.device, base_conv_layer.weight.dtype
        out_features, in_features, _, _ = base_conv_layer.weight.shape
        conv_attrs_list = ["kernel_size", "stride", "padding", "dilation", "groups", "padding_mode"]
        conv_config = {attr: getattr(base_conv_layer, attr) for attr in conv_attrs_list}
        has_bias = base_conv_layer.bias is not None

        # Set up lora component
        lora_layers = nn.Sequential()
        lora_layers.add_module("A", nn.Conv2d(in_features, rank, bias=False, **conv_config))
        lora_layers.add_module("B", nn.Conv2d(rank, out_features, kernel_size=(1, 1), bias=has_bias))
        lora_layers.to(device=device, dtype=dtype)
        return lora_layers

    # .................................................................................................................

    def forward(self, imagelike_tokens: Tensor) -> Tensor:
        return self.base(imagelike_tokens) + self.lora(imagelike_tokens)

    # .................................................................................................................

    def reset_weights(self, use_zeroed_state: bool = True, non_zero_scale: float = 0.1) -> None:
        has_bias = self.base.bias is not None
        nn.init.kaiming_uniform_(self.lora.A.weight, nonlinearity="conv2d")
        nn.init.zeros_(self.lora.B.weight)
        if has_bias:
            nn.init.zeros_(self.lora.B.bias)

        if not use_zeroed_state:
            nn.init.uniform_(self.lora.B.weight, -non_zero_scale, non_zero_scale)
            if has_bias:
                nn.init.uniform_(self.lora.B.bias, -non_zero_scale, non_zero_scale)
        return

    def load_weights(self, state_dict) -> bool:

        # Wipe out any training gradients, since they shouldn't apply to loaded weights
        need_resize_layers = False
        self.lora.zero_grad()

        with torch.no_grad():

            try:
                # Try to load weights directly (this works if weights were originally saved from this module)
                self.lora.load_state_dict(state_dict)
            except RuntimeError:
                # -> Error happens if the loaded weights don't match the sizing (e.g. rank) of the existing module
                need_resize_layers = True

                # Try to read out the weights we expect to be loading
                a_weight = state_dict.get("A.weight", None)
                b_weight = state_dict.get("B.weight", None)
                assert a_weight is not None, f"Unable to load target 'A.weight' (Got: {state_dict.keys()})"
                assert b_weight is not None, f"Unable to load target 'B.weight' (Got: {state_dict.keys()})"

                # Sanity checks. Try to make sure weights are sized correctly
                a_rank, a_in_feats, kh, kw = a_weight.shape
                b_out_feats, b_rank, _, _ = b_weight.shape
                base_out_feats, base_in_feats, base_kh, base_kw = self.base.weight.shape
                assert a_rank == b_rank, f"Bad A/B rank ({a_rank} vs. {b_rank})"
                assert a_in_feats == base_in_feats, f"Bad in-feature counts ({a_in_feats} vs. {base_in_feats})"
                assert b_out_feats == base_out_feats, f"Bad out-feature counts ({b_out_feats} vs. {base_out_feats})"
                assert (kh == base_kh) and (kw == base_kw), f"Bad kernel sizing ({kh},{kw}) vs. ({base_kh},{base_kw})"

                # Replace with lora sized to loaded rank
                del self.lora
                self.lora = self._make_lora(self.base, a_rank)
                self.lora.load_state_dict(state_dict)

            # Make sure device/dtype of lora matches base layer
            device, dtype = self.base.weight.device, self.base.weight.dtype
            self.lora.to(device=device, dtype=dtype)

        return need_resize_layers

    def record_weights(self, move_to_cpu: bool = True) -> dict[str, Tensor]:
        if move_to_cpu:
            return {name: weight.clone().float().cpu() for name, weight in self.lora.state_dict().items()}
        return {name: weight.clone() for name, weight in self.lora.state_dict().items()}

    # .................................................................................................................

    @property
    def weight(self):
        additive_weight = torch.einsum("OrHW,rIHW->OIHW", self.lora.B.weight, self.lora.A.weight)
        return self.base.weight + additive_weight

    @property
    def bias(self):
        if self.base.bias is not None:
            return self.base.bias + self.lora.B.bias
        return self.base.bias

    # .................................................................................................................


class LoraConvTranspose2D(nn.Module):
    """
    Layer used for training 2D convolution-transpose layers.

    ConvTranspose2d is meant to be the 'reverse' of a Conv2d, so the idea
    here is similar to the conv2D lora implementation.
    A 'low-rank' conv-transpose is done first, followed by a 1x1 conv2D (not transpose)
    to get back to the original channel count.
    This result is added to the output from running the original layer.
    """

    # .................................................................................................................

    def __init__(
        self,
        base_convtranspose_layer: nn.ConvTranspose2d,
        rank: int = 1,
        force_no_grad: bool = True,
    ):

        # Inherit from parent
        super().__init__()

        # Make sure base layer isn't trainable
        if force_no_grad:
            base_convtranspose_layer.requires_grad_(False)

        # Store components for forward calls
        self.base = base_convtranspose_layer
        self.lora = self._make_lora(base_convtranspose_layer, rank)
        self.reset_weights()

    # .................................................................................................................

    @staticmethod
    def _make_lora(
        base_convtranspose_layer: nn.ConvTranspose2d,
        rank: int,
    ):
        # Figure out config of input layer
        device, dtype = base_convtranspose_layer.weight.device, base_convtranspose_layer.weight.dtype
        in_features, out_features, _, _ = base_convtranspose_layer.weight.shape
        ct_attrs_list = ["kernel_size", "stride", "padding", "output_padding", "groups", "dilation", "padding_mode"]
        ct_config = {attr: getattr(base_convtranspose_layer, attr) for attr in ct_attrs_list}
        has_bias = base_convtranspose_layer.bias is not None

        # Set up lora component
        lora_layers = nn.Sequential()
        lora_layers.add_module("A", nn.ConvTranspose2d(in_features, rank, bias=False, **ct_config))
        lora_layers.add_module("B", nn.Conv2d(rank, out_features, kernel_size=(1, 1), bias=has_bias))
        lora_layers.to(device=device, dtype=dtype)
        return lora_layers

    # .................................................................................................................

    def forward(self, imagelike_tokens: Tensor) -> Tensor:
        return self.base(imagelike_tokens) + self.lora(imagelike_tokens)

    # .................................................................................................................

    def reset_weights(self, use_zeroed_state: bool = True, non_zero_scale: float = 0.1) -> None:
        has_bias = self.base.bias is not None
        nn.init.kaiming_uniform_(self.lora.A.weight, nonlinearity="conv_transpose2d")
        nn.init.zeros_(self.lora.B.weight)
        if has_bias:
            nn.init.zeros_(self.lora.B.bias)

        if not use_zeroed_state:
            nn.init.uniform_(self.lora.B.weight, -non_zero_scale, non_zero_scale)
            if has_bias:
                nn.init.uniform_(self.lora.B.bias, -non_zero_scale, non_zero_scale)
        return

    def load_weights(self, state_dict) -> bool:

        # Wipe out any training gradients, since they shouldn't apply to loaded weights
        need_resize_layers = False
        self.lora.zero_grad()

        with torch.no_grad():

            try:
                # Try to load weights directly (this works if weights were originally saved from this module)
                self.lora.load_state_dict(state_dict)
            except RuntimeError:
                # -> Error happens if the loaded weights don't match the sizing (e.g. rank) of the existing module
                need_resize_layers = True

                # Try to read out the weights we expect to be loading
                a_weight = state_dict.get("A.weight", None)
                b_weight = state_dict.get("B.weight", None)
                assert a_weight is not None, f"Unable to load target 'A.weight' (Got: {state_dict.keys()})"
                assert b_weight is not None, f"Unable to load target 'B.weight' (Got: {state_dict.keys()})"

                # Sanity checks. Try to make sure weights are sized correctly
                a_in_feats, a_rank, kh, kw = a_weight.shape
                b_out_feats, b_rank, _, _ = b_weight.shape
                base_in_feats, base_out_feats, base_kh, base_kw = self.base.weight.shape
                assert a_rank == b_rank, f"Bad A/B rank ({a_rank} vs. {b_rank})"
                assert a_in_feats == base_in_feats, f"Bad in-feature counts ({a_in_feats} vs. {base_in_feats})"
                assert b_out_feats == base_out_feats, f"Bad out-feature counts ({b_out_feats} vs. {base_out_feats})"
                assert (kh == base_kh) and (kw == base_kw), f"Bad kernel sizing ({kh},{kw}) vs. ({base_kh},{base_kw})"

                # Replace with lora sized to loaded rank
                del self.lora
                self.lora = self._make_lora(self.base, a_rank)
                self.lora.load_state_dict(state_dict)

            # Make sure device/dtype of lora matches base layer
            device, dtype = self.base.weight.device, self.base.weight.dtype
            self.lora.to(device=device, dtype=dtype)

        return need_resize_layers

    def record_weights(self, move_to_cpu: bool = True) -> dict[str, Tensor]:
        if move_to_cpu:
            return {name: weight.clone().float().cpu() for name, weight in self.lora.state_dict().items()}
        return {name: weight.clone() for name, weight in self.lora.state_dict().items()}

    # .................................................................................................................

    @property
    def weight(self):
        additive_weight = torch.einsum("Oryx,IrHW->IOHW", self.lora.B.weight, self.lora.A.weight)
        return self.base.weight + additive_weight

    @property
    def bias(self):
        if self.base.bias is not None:
            return self.base.bias + self.lora.B.bias
        return self.base.bias

    # .................................................................................................................


class LoraEmbedding(nn.Module):
    """
    Layer used to train embeddings.
    Works by 'rotating & offsetting' all embeddings
    by the same (low-rank) rotation matrix:
        original embedding 'E' is an NxF 'matrix',
        -> Where N in the number of embeddings, F is feature count

    Lora does:
        E * (identity + A * B) + bias
        -> Where A, B are low-rank matrices (FxR and RxF, R is rank), bias is a vector of size F
    """

    # .................................................................................................................

    def __init__(
        self,
        base_embedding: nn.Embedding,
        rank: int = 1,
        force_no_grad: bool = True,
    ):

        # Inherit from parent
        super().__init__()

        # Make sure base layer isn't trainable
        if force_no_grad:
            base_embedding.requires_grad_(False)

        # Figure out config of input layer
        device, dtype = base_embedding.weight.device, base_embedding.weight.dtype
        num_embeddings, num_features = base_embedding.weight.shape

        # Set up lora components
        lora_layers, lora_eye = self._make_lora(num_features, rank, device, dtype)
        self.register_buffer("lora_eye", lora_eye, persistent=False)

        # Store components for forward calls
        self.base = base_embedding
        self.lora = lora_layers
        self.reset_weights()

    @staticmethod
    def _make_lora(
        num_features: int,
        rank: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):

        lora_layers = nn.ParameterDict()
        lora_layers["A"] = nn.Parameter(torch.zeros(num_features, rank))
        lora_layers["B"] = nn.Parameter(torch.zeros(rank, num_features))
        lora_layers["bias"] = nn.Parameter(torch.zeros(num_features))
        lora_layers.to(device=device, dtype=dtype)

        lora_eye = torch.eye(num_features, device=device, dtype=dtype)

        return lora_layers, lora_eye

    # .................................................................................................................

    def forward(self, index_tensor: Tensor) -> Tensor:
        base = self.base(index_tensor)
        rotate = self.lora_eye + torch.matmul(self.lora.A, self.lora.B)
        return torch.matmul(base, rotate) + self.lora.bias

    # .................................................................................................................

    def reset_weights(self, use_zeroed_state: bool = True, non_zero_scale: float = 0.1) -> None:
        nn.init.kaiming_uniform_(self.lora.A, nonlinearity="linear")
        nn.init.zeros_(self.lora.B)
        nn.init.zeros_(self.lora.bias)

        if not use_zeroed_state:
            nn.init.uniform_(self.lora.B.weight, -non_zero_scale, non_zero_scale)
            nn.init.uniform_(self.lora.B.bias, -non_zero_scale, non_zero_scale)
        return

    def record_weights(self, move_to_cpu: bool = True) -> dict[str, Tensor]:
        if move_to_cpu:
            return {name: weight.clone().float().cpu() for name, weight in self.lora.state_dict().items()}
        return {name: weight.clone() for name, weight in self.lora.state_dict().items()}

    def load_weights(self, state_dict) -> bool:

        # Wipe out any training gradients, since they shouldn't apply to loaded weights
        need_resize_layers = False
        self.lora.zero_grad()

        with torch.no_grad():

            try:
                # Try to load weights directly (this works if weights were originally saved from this module)
                self.lora.load_state_dict(state_dict)

            except RuntimeError:
                # -> Error happens if the loaded weights don't match the sizing (e.g. rank) of the existing module
                need_resize_layers = True

                # Try to read out the weights we expect to be loading
                a_weight = state_dict.get("A", None)
                b_weight = state_dict.get("B", None)
                bias_weight = state_dict.get("bias", None)
                assert (
                    any(weight is None for weight in (a_weight, b_weight, bias_weight)) is not None
                ), f"Unable to load target weights ('A', 'B' and 'bias') (Got: {state_dict.keys()})"

                # Get loaded weight sizing and check that rank makes sense
                a_device, a_dtype = a_weight.device, a_weight.dtype
                num_a_feats, a_rank = a_weight.shape
                b_rank, num_b_feats = b_weight.shape
                num_bias = bias_weight.shape[0]
                assert a_rank == b_rank, f"A and B weights have mismatching rank ({a_rank} vs. {b_rank})"
                assert num_a_feats == num_b_feats, f"A/B feature count mismatch ({num_a_feats} vs. {num_b_feats})"
                assert num_a_feats == num_bias, f"A/bias feature count mismatch ({num_a_feats} vs. {num_bias})"

                # Make sure loaded weights match the base weight sizing
                _, num_base_feats = self.base.weight.shape
                assert num_a_feats == num_base_feats, f"Base feature mismatch ({num_a_feats} vs. {num_base_feats})"

                # Replace with lora sized to loaded weights
                new_lora, lora_eye = self._make_lora(num_a_feats, a_rank, a_device, a_dtype)
                del self.lora
                self.lora = new_lora
                self.lora.load_state_dict(state_dict)
                self.lora_eye = lora_eye

            # Make sure device/dtype of lora matches base layer
            device, dtype = self.base.weight.device, self.base.weight.dtype
            self.lora.to(device=device, dtype=dtype)
            self.lora_eye = self.lora_eye.to(device=device, dtype=dtype)

        return need_resize_layers

    # .................................................................................................................

    @property
    def weight(self) -> Tensor:
        base_weight = self.base.weight
        lora_linear = torch.matmul(self.lora.A, self.lora.B)
        perturb = self.lora_eye + lora_linear
        return torch.matmul(base_weight, perturb) + self.lora.bias

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Offsets


class OffsetLayernorm(nn.Module):
    """
    Layer used to train layernorms without directly altering the original values,
    this allows the trained weights to be reset without losing the original weights.

    Idea is as follows:
            layernorm(T) = normalize(T) * W + B
      OffsetLayernorm(T) = layernorm(T) * (W + w_offset) + (B + b_offset)
    """

    # .................................................................................................................

    def __init__(
        self,
        base_layernorm_layer: nn.LayerNorm,
        include_bias: bool = True,
        force_no_grad: bool = True,
    ):

        # Inherit from parent
        super().__init__()

        # Make sure base layer isn't trainable
        if force_no_grad:
            base_layernorm_layer.requires_grad_(False)

        # Figure out config of input layer
        device, dtype = base_layernorm_layer.weight.device, base_layernorm_layer.weight.dtype
        num_features = base_layernorm_layer.weight.shape[0]
        has_bias = base_layernorm_layer.bias is not None
        include_bias = has_bias and include_bias

        # Set up offsets
        offset_layers = self._make_offset(num_features, include_bias, device, dtype)

        # Store components for forward calls
        self._has_bias = include_bias
        self.base = base_layernorm_layer
        self.offset = offset_layers
        self.reset_weights()

    @staticmethod
    def _make_offset(
        num_features: int,
        include_bias: bool,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:

        offset_layers = nn.ParameterDict()
        offset_layers["weight"] = nn.Parameter(torch.empty(num_features))
        if include_bias:
            offset_layers["bias"] = nn.Parameter(torch.zeros(num_features))
        offset_layers.to(device=device, dtype=dtype)

        return offset_layers

    # .................................................................................................................

    def forward(self, tokens: Tensor) -> Tensor:
        zeroed_mean = tokens - tokens.mean(dim=-1, keepdim=True)
        channel_stdev = torch.sqrt(zeroed_mean.square().mean(dim=-1, keepdim=True) + self.base.eps)
        offset_weight = self.base.weight + self.offset.weight
        result = offset_weight * (zeroed_mean / channel_stdev)
        if self._has_bias:
            result += self.base.bias + self.offset.bias
        return result

    # .................................................................................................................

    def reset_weights(self, use_zeroed_state: bool = True, non_zero_scale: float = 0.1) -> None:
        nn.init.zeros_(self.offset.weight)
        if self._has_bias:
            nn.init.zeros_(self.offset.bias)

        if not use_zeroed_state:
            nn.init.uniform_(self.offset.weight, -non_zero_scale, non_zero_scale)
            if self._has_bias:
                nn.init.uniform_(self.offset.bias, -non_zero_scale, non_zero_scale)
        return

    def record_weights(self, move_to_cpu: bool = True) -> dict[str, Tensor]:
        if move_to_cpu:
            return {name: weight.clone().float().cpu() for name, weight in self.offset.state_dict().items()}
        return {name: weight.clone() for name, weight in self.offset.state_dict().items()}

    def load_weights(self, state_dict) -> bool:

        # Wipe out any training gradients, since they shouldn't apply to loaded weights
        need_resize_layers = False
        self.offset.zero_grad()

        with torch.no_grad():

            try:
                self.offset.load_state_dict(state_dict)

            except RuntimeError:
                # -> Error happens if the loaded weights don't match the config of the existing module
                need_resize_layers = True

                # Try to read out the weights we expect to be loading
                weight_offset = state_dict.get("weight", None)
                bias_offset = state_dict.get("bias", None)
                has_bias = bias_offset is not None
                assert weight_offset is not None, "Unable to load 'weight' offset"

                # Get loaded weight sizing
                w_device, w_dtype = weight_offset.device, weight_offset.dtype
                num_w_feats = weight_offset.shape[0]
                _, num_base_feats = self.base.weight.shape
                assert num_w_feats == num_base_feats, f"Base feature mismatch ({num_w_feats} vs. {num_base_feats})"

                # Replace with offset sized to loaded weights
                new_lora = self._make_offset(num_w_feats, has_bias, w_device, w_dtype)
                del self.offset
                self.offset = new_lora
                self.offset.load_state_dict(state_dict)
                self._has_bias = has_bias

            # Make sure device/dtype of lora matches base layer
            device, dtype = self.base.weight.device, self.base.weight.dtype
            self.offset.to(device=device, dtype=dtype)

        return need_resize_layers

    # .................................................................................................................

    @property
    def weight(self):
        return self.base.weight + self.offset.weight

    @property
    def bias(self):
        if self._has_bias:
            return self.base.bias + self.offset.bias
        return self.base.bias

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Checkpointing


class CheckpointModule(nn.Module):
    """Helper used to 'checkpoint' a model component, which helps reduce memory use during training"""

    def __init__(self, original_module: nn.Module):
        super().__init__()
        self.original_module = original_module

    def forward(self, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(self.original_module, *args, use_reentrant=False, **kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def replace_submodule(
    model: nn.Module,
    submodule_str: str,
    replacement_module: nn.Module,
) -> nn.Module:
    """
    Function used to (dynamically) do the equivalent of:
        model.component.layer = replacement_module(model.component.layer)

    Notice that the 'replacement_module' is expected to take in the module
    it replaces as it's only argument. The replacement should also have
    a .forward(...) function that behaves like the module it replaces,
    or else it will likely break the model.

    Also note that the 'component.layer' part is specified using a string
    which allows this to be done in a loop across the entire model,
    without needing to write out the full 'model.A.B.C...' assignment.
    The function will automatically handle indexed layers as well,
    for example 'component.0.layer' is equivalent to replacing:
        model.component[0].layer = replacement_module(model.component[0].layer)

    Returns the instantiated replacement module
    """

    # Sanity check
    assert len(submodule_str) > 0, "Must provide a submodule string!"

    # Check submodule components (expecting things like: 'layer.1.proj')
    component_str_list = submodule_str.split(".")
    is_next_numeric = component_str_list[0].isnumeric()
    is_leaf = len(component_str_list) == 1
    component_idx = int(component_str_list[0]) if is_next_numeric else None

    # Handle last-most submodule differently, so we properly replace it
    if is_leaf:
        base_layer = model[component_idx] if is_next_numeric else model.get_submodule(submodule_str)
        new_lora_layer = replacement_module(base_layer)
        if is_next_numeric:
            model[component_idx] = new_lora_layer
        else:
            model.add_module(submodule_str, new_lora_layer)
        return new_lora_layer

    # Run function recursively to find leaf modules
    sub_model = model[component_idx] if is_next_numeric else model.get_submodule(component_str_list[0])
    remaining_submod_str = ".".join(component_str_list[1:])
    return replace_submodule(sub_model, remaining_submod_str, replacement_module)


# .....................................................................................................................


def replace_target_modules(
    model: nn.Module,
    submodule_prefix: str | None,
    module_to_replace: nn.Module,
    replacement_module: nn.Module,
    exclude_names_func: callable | None = None,
) -> tuple[dict[str, nn.Module], int]:
    """
    Helper used to replace all instances of a type of module (e.g. nn.Linear)
    with a replacement_module. The replacement_module is expected to take in
    the module it replaces as it's only argument.
    See the 'replace_submodule' function for more details.

    A 'submodule_prefix' can be given which allows for specifying replacement of
    a specific submodule of the model. For example, submodule_prefix='image_encoder'
    would mean that only modules inside the image encoder component are replaced.
    This is equivalent to calling the function using:
        replace_target_modules(model.image_encoder, ....)
    However, by using a submodule string instead of directly providing the submodule,
    the returned weights will have the proper submodule pathing/weight keys!

    Returns:
        new_modules_dict, trainable_parameter_count
        -> The new_modules_dict has keys which represent the submodule string of every
           module that has been replaced and values which are the corresponding replacement modules
    """

    # Index into the model if we're given a prefix
    # (useful to limit replacement to be within certain submodules)
    if submodule_prefix is not None:
        model = model.get_submodule(submodule_prefix)

    # Record full submodule names of target layers to replace (e.g. model.layer.component.mlp.0...)
    submod_strs_to_replace = []
    for module_name, module in model.named_modules():
        if isinstance(module, module_to_replace):
            if exclude_names_func is not None:
                if exclude_names_func(module_name):
                    continue
            submod_strs_to_replace.append(module_name)

    # Replace each of the layers found above & record submodule name->lora instance
    # -> This has to be done after-the-fact to avoid changing .named_modules() sizing inside of the loop
    new_modules_dict = {
        mod_str: replace_submodule(model, mod_str, replacement_module) for mod_str in submod_strs_to_replace
    }

    # Re-add the submodule string to the key for storage (so that it's correct relative to the original input model)
    if submodule_prefix is not None:
        new_modules_dict = {f"{submodule_prefix}.{key}": val for key, val in new_modules_dict.items()}

    # Count number of new trainable parameters
    param_counts_list = []
    for name, module_ref in new_modules_dict.items():
        param_counts_list.append(sum(p.numel() for p in module_ref.parameters() if p.requires_grad))
    param_count = sum(param_counts_list)

    return new_modules_dict, param_count


# .....................................................................................................................


def checkpoint_image_encoder_stages(model: nn.Module):
    """Helper used to apply activation checkpointing to the SAM image encoder stages"""

    # Sanity check
    is_trainable_model = any(param.requires_grad for param in model.image_encoder.parameters())
    if not is_trainable_model:
        raise AttributeError("Checkpointing failed! Model does not contain trainable parameters...")

    # Apply checkpointing to each stage of the image encoder
    num_stages = len(model.image_encoder.stages)
    for idx in range(num_stages):
        model.image_encoder.stages[idx] = CheckpointModule(model.image_encoder.stages[idx])

    return
