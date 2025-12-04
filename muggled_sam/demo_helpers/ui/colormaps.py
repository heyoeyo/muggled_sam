#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .base import BaseCallback
from .helpers.images import draw_box_outline

from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class HColormapsBar(BaseCallback):
    """
    UI element used to render a horizontal arrangement of colormap 'buttons'
    Similar to using:
        HStack(ToggleImage(colormap_image_1), ToggleImage(...), ... etc)

    However this implementation is more efficient and is better behaved when
    scaling display sizes.
    """

    # .................................................................................................................

    def __init__(self, *colormap_codes_or_luts: int | ndarray | None, bar_height=40, minimum_width=128):
        """
        Colormap inputs should be provided as either:
            1) integer opencv colormap codes, e.g. cv2.COLORMAP_VIRIDIS
            2) LUTs, which are uint8 numpy arrays of shape: 1x256x3, holding BGR mappings
            3) None, which is interpretted to mean a grayscale colormap
        """

        # Store basic state
        self._is_changed = False
        self._cmap_idx = 0

        # Store sizing settings
        self._height = bar_height
        self._min_width = minimum_width
        self._need_rerender = True
        self._rendered_hw = (0, 0)
        self._cached_image = np.zeros((1, 1, 3), dtype=np.uint8)

        # Check & store each provided colormap and interpret 'None' as grayscale colormap
        cmap_luts_list = []
        for cmap in colormap_codes_or_luts:
            if cmap is None:
                cmap = make_gray_colormap()
            elif isinstance(cmap, int):
                gray_img = make_gray_colormap()
                cmap = cv2.applyColorMap(gray_img, cmap)

            if not isinstance(cmap, ndarray):
                raise TypeError("Unrecognized colormap type! Must be a cv2 colormap code, None or a 1x256x3 array")
            assert cmap.shape == (1, 256, 3), "Bad colormap shape, must be: 1x256x3"
            cmap_luts_list.append(cmap)
        self._cmap_luts_list = cmap_luts_list
        self._num_cmaps = len(self._cmap_luts_list)

        # Sanity check. Make sure we have at least 1 colormap
        if self._num_cmaps == 0:
            self._cmap_luts_list = [make_gray_colormap()]
            self._num_cmaps = 1

        # Inherit from parent
        super().__init__(bar_height, minimum_width, max_h=bar_height, expand_w=True)

    # .................................................................................................................

    def read(self) -> tuple[bool, int, ndarray]:
        """
        Returns:
            is_changed, selected_colormap_index, selected_colormap_lut

        -> This function is not needed, unless needing to detect the change
           state or if there is a need for the selected index.
        -> If only colormapping is needed, use the .apply_colormap(image_1ch) function!
        """

        is_changed = self._is_changed
        self._is_changed = False

        cmap_idx = self._cmap_idx
        cmap_lut = self._cmap_luts_list[cmap_idx]

        return is_changed, cmap_idx, cmap_lut

    # .................................................................................................................

    def on_left_click(self, cbxy, cbflags) -> None:

        # Map normalized x-coord to button boundaries
        x_norm = cbxy.xy_norm[0]
        new_cmap_idx = int(x_norm * self._num_cmaps)
        new_cmap_idx = int(np.clip(new_cmap_idx, 0, self._num_cmaps - 1))

        # Store updated selection
        self._is_changed |= new_cmap_idx != self._cmap_idx
        self._cmap_idx = new_cmap_idx
        self._need_rerender = self._is_changed

        return

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        # Only render if sizing changes or we force a re-render
        # (we don't expect frequent changes to the colormap display!)
        img_h, img_w = self._cached_image.shape[0:2]
        if img_h != h or img_w != w or self._need_rerender:

            ideal_w_per_btn = w // self._num_cmaps
            num_gap_pixels = w - (ideal_w_per_btn * self._num_cmaps)
            imgs_list = []
            for btn_idx, lut in enumerate(self._cmap_luts_list):

                btn_w = ideal_w_per_btn + (1 if btn_idx < num_gap_pixels else 0)
                img = cv2.resize(lut, dsize=(btn_w, h), interpolation=cv2.INTER_NEAREST_EXACT)
                img = draw_box_outline(img, (0, 0, 0), 1)
                if btn_idx == self._cmap_idx:
                    img = draw_box_outline(img, (0, 0, 0), 2)
                    img = draw_box_outline(img, (255, 255, 255), 1)

                imgs_list.append(img)

            # Cache final image and clear re-render flag
            self._cached_image = np.hstack(imgs_list)
            self._need_rerender = False

        return self._cached_image

    # .................................................................................................................

    def _get_height_given_width(self, w) -> int:
        return self._height

    # .................................................................................................................

    def _get_height_and_width_without_hint(self) -> [int, int]:
        return self._height, self._min_width

    # .................................................................................................................

    def _get_width_given_height(self, h) -> int:
        return self._min_width

    # .................................................................................................................

    def apply_colormap(self, image_uint8_1ch) -> ndarray:
        """Apply the currently selected colormap to the provided (1-channel) image"""
        return self.apply_given_colormap(image_uint8_1ch, self._cmap_luts_list[self._cmap_idx])

    # .................................................................................................................

    @staticmethod
    def apply_given_colormap(image_uint8_1ch, colormap_code_or_lut) -> ndarray:
        """
        Converts a uint8 image (numpy array) into a bgr color image using opencv colormaps
        or using LUTs (numpy arrays of shape 1x256x3).
        Colormap code should be from opencv, which are accessed with: cv2.COLORMAP_{name}
        LUTs should be numpy arrays of shape 1x256x3, where each of the 256 entries
        encodes a bgr value which maps on to a 0-255 range.

        Expects an image of shape: HxWxC (with 1 or no channels, i.e. HxW only)
        """

        if isinstance(colormap_code_or_lut, int):
            # Handle maps provided as opencv colormap codes (e.g. cv2.COLORMAP_VIRIDIS)
            return cv2.applyColorMap(image_uint8_1ch, colormap_code_or_lut)

        elif isinstance(colormap_code_or_lut, ndarray):
            # Handle maps provided as LUTs (e.g. 1x256x3 numpy arrays)
            image_ch3 = cv2.cvtColor(image_uint8_1ch, cv2.COLOR_GRAY2BGR)
            return cv2.LUT(image_ch3, colormap_code_or_lut)

        elif colormap_code_or_lut is None:
            # Return grayscale image if no mapping is provided
            return cv2.cvtColor(image_uint8_1ch, cv2.COLOR_GRAY2BGR)

        # Error if we didn't deal with the colormap above
        raise TypeError(f"Error applying colormap, unrecognized colormap type: {type(colormap_code_or_lut)}")

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions

# .................................................................................................................


def make_gray_colormap(num_samples=256):
    """Makes a colormap in opencv LUT format, for grayscale output using cv2.LUT function"""
    return make_colormap_from_keypoints(np.float32([(0, 0, 0), (1, 1, 1)]))


# .................................................................................................................


def make_spectral_colormap(num_samples=256):
    """
    Creates a colormap which is a variation on the built-in opencv 'TURBO' colormap,
    but with muted colors and overall less harsh contrast. The colormap originates from
    the matplotlib library, where it is the reversed version of a colormap called
    'Spectral'. It is being generated this way to avoid requiring the full matplotlib dependency!

    The original colormap definition can be found here:
    https://github.com/matplotlib/matplotlib/blob/30f803b2e9b5e237c5c31df57f657ae69bec240d/lib/matplotlib/_cm.py#L793
    -> The version here uses a slightly truncated copy of the values
    -> This version is also pre-reversed compared to the original
    -> Color keypoints are in bgr order (the original uses rgb ordering, opencv needs bgr)

    Returns a colormap which can be used with opencv, for example:

        spectral_colormap = make_spectral_colormap()
        gray_image_3ch = cv2.cvtColor(gray_image_1ch, cv2.COLOR_GRAY2BGR)
        colormapped_image = cv2.LUT(gray_image_3ch, spectral_colormap)

    The result has a shape of: 1xNx3, where N is number of samples (256 by default and required for cv2.LUT usage)
    """

    # Colormap keypoints from matplotlib. The colormap is produced by linear-interpolation of these points
    spectral_rev_bgr = np.float32(
        (
            (0.635, 0.310, 0.369),
            (0.741, 0.533, 0.196),
            (0.647, 0.761, 0.400),
            (0.643, 0.867, 0.671),
            (0.596, 0.961, 0.902),
            (0.749, 1.000, 1.000),
            (0.545, 0.878, 0.996),
            (0.380, 0.682, 0.992),
            (0.263, 0.427, 0.957),
            (0.310, 0.243, 0.835),
            (0.259, 0.004, 0.620),
        )
    )

    return make_colormap_from_keypoints(spectral_rev_bgr)


# .................................................................................................................


def make_colormap_from_keypoints(bgr_norm_keypoints: ndarray, num_samples=256) -> ndarray:
    """
    Helper used to construct colormaps from a set of keypoint values.
    Uses linear interpolation to compute intermediate values needed to
    reach the given 'num_samples'.
    The input keypoints should be a float32 numpy array
    of bgr-ordered colors in 0-to-1 normalized format:

        bgr_norm_keypoints = np.float32(
            (
                (0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0),
            )
        )

    Returns a a uint8 numpy array which can be used as a
    colormap LUT for use with opencv (assuming num_samples=256).
    The LUT will have a shape of 1xNx3 (N = num_samples)
    """

    # Build out indexing into the keypoint array vs. colormap sample indexes
    norm_idx = np.linspace(0, 1, num_samples)
    keypoint_idx = norm_idx * (len(bgr_norm_keypoints) - 1)

    # Get the start/end indexes for linear interpolation at each colormap sample
    a_idx = np.int32(np.floor(keypoint_idx))
    b_idx = np.int32(np.ceil(keypoint_idx))
    t_val = keypoint_idx - a_idx

    # Compute colormap as a linear interpolation between keypoints
    bias = bgr_norm_keypoints[a_idx]
    slope = bgr_norm_keypoints[b_idx] - bgr_norm_keypoints[a_idx]
    cmap_bgr_values = bias + slope * np.expand_dims(t_val, 1)
    cmap_bgr_values = np.round(cmap_bgr_values * 255).astype(np.uint8)

    return np.expand_dims(cmap_bgr_values, 0)
