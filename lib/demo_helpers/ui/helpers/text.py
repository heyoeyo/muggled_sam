#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from typing import TypeAlias


# ---------------------------------------------------------------------------------------------------------------------
# %% Data types

XYPX: TypeAlias = tuple[int, int]
XYNORM: TypeAlias = tuple[float, float]
COLOR: TypeAlias = tuple[int, int, int]


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class TextDrawer:
    """
    Helper used to handle text-drawing onto images
    If a background color is given, text will be drawn with a thicker background for better contrast
    """

    # .................................................................................................................

    def __init__(
        self,
        scale: float = 0.5,
        thickness: int = 1,
        color: COLOR = (255, 255, 255),
        bg_color: COLOR | None = None,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        line_type=cv2.LINE_AA,
    ):
        self._fg_color = color
        self._fg_thick = thickness
        self._bg_color = bg_color
        self._font = font
        self._scale = scale
        self._ltype = cv2.LINE_AA

    # .................................................................................................................

    @classmethod
    def from_existing(cls, other_text_drawer):
        assert isinstance(other_text_drawer, cls), "Must be created from another text drawer instance!"
        o = other_text_drawer
        return cls(o._scale, o._fg_thick, o._fg_color, o._bg_color, o._font, o._ltype)

    # .................................................................................................................

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(scale={self._scale}, thickness={self._fg_thick}, color={self._fg_color})"

    # .................................................................................................................

    def style(
        self,
        scale: float | None = None,
        thickness: int | None = None,
        color: COLOR | None = None,
        bg_color: COLOR | None = None,
    ):
        """
        Update text styling. Any settings given as None will remain unchanged
        To clear an existing background color (which normally requires setting it to None), use -1
        """
        if scale is not None:
            self._scale = scale
        if thickness is not None:
            self._fg_thick = thickness
        if color is not None:
            self._fg_color = color
        elif bg_color is not None:
            self._bg_color = bg_color if bg_color != -1 else None
        return self

    # .................................................................................................................

    def xy_px(
        self,
        image,
        text: str,
        xy_px: XYPX,
        scale: float | None = None,
        thickness: int | None = None,
        color: COLOR | None = None,
    ):
        """Helper used to draw text at a give location using pre-configured settings"""

        # Fill in defaults
        if scale is None:
            scale = self._scale
        if color is None:
            color = self._fg_color
        if thickness is None:
            thickness = self._fg_thick

        if self._bg_color is not None:
            bg_thick = min(thickness + 3, thickness * 3)
            image = cv2.putText(image, text, xy_px, self._font, scale, self._bg_color, bg_thick, self._ltype)

        return cv2.putText(image, text, xy_px, self._font, scale, color, thickness, self._ltype)

    # .................................................................................................................

    def xy_norm(
        self,
        image,
        text: str,
        xy_norm: XYNORM,
        anchor_xy_norm: XYNORM | None = None,
        offset_xy_px: XYPX = (0, 0),
        scale: float | None = None,
        thickness: int | None = None,
        color: COLOR | None = None,
    ):
        """
        Helper used to draw text given normalized (0-to-1) xy coordinates
        An anchor point can be provided to change where the text is drawn, relative
        to the given xy_norm position.
        If an anchor point isn't given, then it will match the xy_norm value itself,
        which will lead to text always being drawn within the image, as long as 0-to-1 coordinates are given.

        For example, an anchor of (0.5, 0.5) means that the text will be centered on the given xy_norm position.
        - To draw text in the top-left corner, use xy_norm = (0,0) and anchor = (0,0)
        - To draw text in the bottom-right corner, use xy_norm = (1,1) and anchor = (1,1)
        - To draw text at the bottom-center, use xy_norm = (0.5, 1) and anchor = (0.5, 1)
        """

        # Figure out pixel coords for the given normalized position
        txt_w, txt_h, txt_base = self.get_text_size(text, scale, thickness)
        img_h, img_w = image.shape[0:2]
        x_norm, y_norm = xy_norm

        # If no anchor is given, match to positioning, which has a 'bounding' effect of text position
        if anchor_xy_norm is None:
            anchor_xy_norm = xy_norm

        # Figure out text positioning on image, in pixel coords
        anchor_x_norm, anchor_y_norm = anchor_xy_norm
        txt_x_px = x_norm * (img_w - 1) - txt_w * anchor_x_norm
        txt_y_px = y_norm * (img_h - 1) + txt_h * (1 - anchor_y_norm)

        # Apply offset before final drawing
        offset_x_px, offset_y_px = offset_xy_px
        txt_xy_px = (round(txt_x_px + offset_x_px), round(txt_y_px + offset_y_px))
        return self.xy_px(image, text, txt_xy_px, scale, thickness, color)

    # .................................................................................................................

    def xy_centered(
        self,
        image,
        text: str,
        scale: float | None = None,
        thickness: int | None = None,
        color: COLOR | None = None,
    ):
        """Helper used to draw x/y centered text"""
        xy_norm, anchor_xy_norm, offset_xy_px = (0.5, 0.5), (0.5, 0.5), (0, 0)
        return self.xy_norm(image, text, xy_norm, anchor_xy_norm, offset_xy_px, scale, thickness, color)

    # .................................................................................................................

    def draw_to_box_norm(
        self,
        image,
        text,
        xy1_norm=(0.0, 0.0),
        xy2_norm=(1.0, 1.0),
        margin_xy_px=(0, 0),
        scale_step_size=0.05,
    ):
        """
        Function used to draw text in order to 'fill' a given box region in the image.

        The scale of the text will be chosen so that the text fits into the
        box given by the top-left/bottom-right coords. (xy1, xy2), minus any
        margin specified and with a scaling limited to multiples of the
        given scale step size (rendering can be cleaner with certain multiples).
        """

        # Figure out how large of a drawing area we have
        img_h, img_w = image.shape[0:2]
        (x1, y1), (x2, y2) = xy1_norm, xy2_norm
        target_h = max(1, (abs(y2 - y1) * img_h) - margin_xy_px[1])
        target_w = max(1, (abs(x2 - x1) * img_w) - margin_xy_px[0])

        # Figure out how much to adjust scale to fit target size
        base_scale = 1
        txt_w, txt_h, _ = self.get_text_size(text, base_scale)
        h_scale, w_scale = target_h / txt_h, target_w / txt_w
        scale_adjust = min(h_scale, w_scale)

        # Draw text to new scale (with step size limiting)
        xy_mid = tuple((a + b) / 2 for a, b in zip(xy1_norm, xy2_norm))
        new_scale = np.floor((base_scale * scale_adjust) / scale_step_size) * scale_step_size
        return self.xy_norm(image, text, xy_mid, anchor_xy_norm=(0.5, 0.5), scale=new_scale)

    # .................................................................................................................

    def check_will_fit_width(self, text: str, target_width: int, shrink_factor=0.9) -> bool:
        """Helper used to check if text could be written into given container width"""
        txt_w, _, _ = self.get_text_size(text)
        return txt_w < int(target_width * shrink_factor)

    # .................................................................................................................

    def check_will_fit_height(self, text: str, target_height: int, shrink_factor=0.9) -> bool:
        """Helper used to check if text could be written into given container height"""
        txt_w, _, _ = self.get_text_size(text)
        return txt_w < int(target_height * shrink_factor)

    # .................................................................................................................

    def get_text_size(self, text, scale: float | None = None, thickness: int | None = None) -> [int, int, int]:
        """
        Helper used to check how big a piece of text will be when drawn
        Returns:
            text_width_px, text_height_px, text_baseline
        """

        if scale is None:
            scale = self._scale
        if thickness is None:
            thickness = self._fg_thick

        (txt_w, txt_h), txt_base = cv2.getTextSize(text, self._font, scale, thickness)
        return txt_w, txt_h, txt_base

    # .................................................................................................................


if __name__ == "__main__":

    txt1 = TextDrawer()
    txt2 = TextDrawer(2, 2, color=(0, 0, 255))

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image = txt2.xy_norm(image, "X=0.25", (0.25, 1))
    image = txt2.xy_norm(image, "X=0.75", (0.75, 1), offset_xy_px=(0, -5), scale=0.75, thickness=1, color=(0, 255, 0))
    image = txt2.xy_centered(image, "**CENTERED**", color=(0, 255, 255))
    image = txt1.xy_norm(image, "LEFT-ANCHORED", (0.5, 0.20), (0, 0.5))
    image = txt1.xy_norm(image, "RIGHT-ANCHORED", (0.5, 0.25), (1, 0.5))

    cv2.imshow("Example", image)
    keypress = cv2.waitKey(0)
    cv2.destroyAllWindows()
