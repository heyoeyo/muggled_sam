#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch.nn as nn


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class ModelOutputCapture:
    """
    Helper used to store results after target modules/layers within a given model are executed.
    Relies on pytorch module 'hooks' functionality:
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook

    Example usage:

        import torch

        # Load model
        model = ...

        # Capture target module output (linear layers)
        target_module = torch.nn.Linear
        captures = ModelOutputCapture(model, target_module)

        # Run model (results will be captured during processing)
        model(input_data)

        # Check out the results
        for result in captures:
            print(result.shape)
            # do something with results...
    """

    def __init__(self, model_ref: nn.Module, target_modules: nn.Module | list[nn.Module]):
        self.results = []
        self._hook_model_outputs(model_ref, target_modules)

    def __call__(self, module, module_in, module_out):
        self.results.append(module_out)

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        yield from self.results

    def __getitem__(self, index):
        return self.results[index]

    def _hook_model_outputs(self, model: nn.Module, target_modules: nn.Module | list[nn.Module]):

        for module in model.modules():
            if isinstance(module, target_modules):
                module.register_forward_hook(self)
            pass

        return
