#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2

from .base import BaseCallback, BaseImageCallback
from .helpers.images import blank_image


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class ExpandingImage(BaseImageCallback):

    # .................................................................................................................

    def __init__(self, image, min_side_length=32, max_side_length=4096):

        # Store sizing constraints
        self._min_side_length = min_side_length
        self._max_side_length = max_side_length

        # Inherit from parent, then override sizing using own internal implementation!
        super().__init__(blank_image(1, 1), expand_h=True, expand_w=True)
        self.set_image(image)

    # .................................................................................................................

    def get_render_hw(self, *, return_as_wh=False):
        return tuple(reversed(self._render_image.shape[0:2])) if return_as_wh else self._render_image.shape[0:2]

    # .................................................................................................................

    def set_image(self, image):

        # Force grayscale images to have 3 channels for display
        self._full_image = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self._render_image = self._full_image.copy()

        # Update sizing limits
        # -> Choose min h/w so that the smallest side matches the min_side_length config setting
        # -> Choose max h/w so that the largest side matches the max_side_length config setting
        img_hw = image.shape[0:2]
        min_h, min_w = (round(val * self._min_side_length / min(img_hw)) for val in img_hw)
        max_h, max_w = (round(val * self._max_side_length / max(img_hw)) for val in img_hw)
        self._rdr.limits.update(min_h=min_h, min_w=min_w, max_h=max_h, max_w=max_w)

        # Reset cache settings to force a re-render
        self._targ_h = -1
        self._targ_w = -1

        return self

    # .................................................................................................................


class StretchImage(BaseImageCallback):

    def __init__(self, image, min_side_length=32, max_side_length=4096):
        raise NotImplementedError()


class FixedImage(BaseCallback):

    def __init__(self, image, min_side_length=32, max_side_length=4096):
        raise NotImplementedError()
