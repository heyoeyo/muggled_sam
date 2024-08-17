#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2

from .base import BaseCallback
from .helpers.text import TextDrawer
from .helpers.images import blank_image, convert_color, draw_box_outline, get_image_hw_to_fill


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class TitledTextBlock(BaseCallback):

    # .................................................................................................................

    def __init__(self, title: str, default_value: str = "", block_height=80, bg_color=(64, 53, 52), text_scale=0.5):

        # Set up text drawing config
        self._text_title = title
        self._text_value = default_value
        self._title_txtdraw = TextDrawer(text_scale * 1.15, font=cv2.FONT_HERSHEY_DUPLEX)
        self._value_txtdraw = TextDrawer(text_scale, thickness=1)

        # Store visual settings
        self._bg_color = bg_color
        self._base_image = blank_image(1, 1, bg_color)
        self._image = self._base_image.copy()
        self._outline_color = (0, 0, 0)

        # Set up sizing limits
        ref_txt = f"  {title}  "
        txt_w, title_txt_h, txt_baseline = self._title_txtdraw.get_text_size(ref_txt)
        _, value_txt_h, _ = self._value_txtdraw.get_text_size(ref_txt)
        min_h = max(block_height, int(1.5 * title_txt_h + value_txt_h))
        super().__init__(min_h, txt_w, expand_h=False, expand_w=False)
        self._title_h = title_txt_h
        self._value_h = value_txt_h

    # .................................................................................................................

    def set_text(self, text):
        self._text_value = str(text)
        return self

    def set_title(self, title):
        if title != self._text_title:
            self._base_image = blank_image(1, 1, self._bg_color)
        self._text_title = title
        return self

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        img_h, img_w = self._image.shape[0:2]
        if img_h != h or img_w != w:
            new_base_img = blank_image(h, w, self._bg_color)
            self._title_txtdraw.xy_norm(new_base_img, self._text_title, (0.5, 0.5), offset_xy_px=(0, -self._title_h))
            self._base_image = new_base_img

        # Re-draw base image when sizing has changed
        disp_img = self._base_image.copy()
        self._value_txtdraw.xy_norm(disp_img, self._text_value, (0.5, 0.5), offset_xy_px=(0, self._value_h))

        # Draw bounding box
        disp_img = draw_box_outline(disp_img, color=self._outline_color)
        return disp_img

    # .................................................................................................................


class TextBlock(BaseCallback):

    def __init__(self, text: str = "", block_height=40, bg_color=(30, 25, 25), text_scale=0.5):

        # Set up text drawing config
        self._text_value = text
        self._value_txtdraw = TextDrawer(text_scale)

        # Store visual settings
        self._bg_color = bg_color
        self._base_image = blank_image(1, 1, bg_color)
        self._image = self._base_image.copy()
        self._outline_color = (0, 0, 0)

        # Set up sizing limits
        ref_txt = "  abcdefg  " if len(text) == 0 else f"  {text}  "
        txt_w, txt_h, _ = self._value_txtdraw.get_text_size(ref_txt)
        min_h = max(block_height, int(1.5 * txt_h))
        super().__init__(min_h, txt_w, expand_h=False, expand_w=False)

    # .................................................................................................................

    def set_text(self, text):
        self._text_value = str(text)
        return self

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        img_h, img_w = self._image.shape[0:2]
        if img_h != h or img_w != w:
            new_base_img = blank_image(h, w, self._bg_color)
            self._base_image = new_base_img

        # Re-draw text, assumnig it may have changed
        disp_img = self._base_image.copy()
        self._value_txtdraw.xy_norm(disp_img, self._text_value, (0.5, 0.5))

        # Draw bounding box
        disp_img = draw_box_outline(disp_img, color=self._outline_color)
        return disp_img

    # .................................................................................................................
