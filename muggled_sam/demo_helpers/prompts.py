#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def check_have_prompts(box_xy1xy2_norm_list: list, fg_xy_norm_list: list, bg_xy_norm_list: list) -> bool:
    """Helper used to check if there are any prompts (i.e. checks for at least one non-empty list)"""
    return any((len(items) > 0 for items in (box_xy1xy2_norm_list, fg_xy_norm_list, bg_xy_norm_list)))
