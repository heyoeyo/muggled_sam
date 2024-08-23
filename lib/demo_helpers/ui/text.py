#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2

from .base import BaseCallback
from .helpers.text import TextDrawer
from .helpers.images import blank_image, draw_box_outline


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

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}({self._text_title}: {self._text_value})"

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

    def __init__(self, text: str = "", block_height=40, bg_color=(30, 25, 25), text_scale=0.35, max_characters=6):

        # Set up text drawing config
        text_str = str(text)
        self._text_value = text_str
        self._value_txtdraw = TextDrawer(text_scale)

        # Store visual settings
        self._bg_color = bg_color
        self._base_image = blank_image(1, 1, bg_color)
        self._image = self._base_image.copy()
        self._outline_color = (0, 0, 0)

        # Set up sizing limits
        max_characters_with_spacing = max(max_characters + 2, len(" {text_str} "))
        ref_txt = " " * max_characters_with_spacing
        txt_w, txt_h, _ = self._value_txtdraw.get_text_size(ref_txt)
        min_h = max(block_height, int(1.5 * txt_h))
        super().__init__(min_h, txt_w, expand_h=False, expand_w=False)

    # .................................................................................................................

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}({self._text_value})"

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


class ValueBlock(TextBlock):

    def __init__(
        self,
        prefix="Value",
        initial_value="-",
        suffix="",
        block_height=40,
        bg_color=(30, 25, 25),
        text_scale=0.35,
        max_characters=4,
    ):

        # Set up text drawing config
        self._prefix = str(prefix)
        self._suffix = str(suffix)
        self._text_value = ""
        self.set_value(initial_value)

        # Store visual settings
        self._bg_color = bg_color
        self._base_image = blank_image(1, 1, bg_color)
        self._image = self._base_image.copy()
        self._outline_color = (0, 0, 0)

        # Inherit from parent
        max_all_characters = len(self._prefix) + len(self._suffix) + max_characters
        super().__init__(self._text_value, block_height, bg_color, text_scale, max_all_characters)

    # .................................................................................................................

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}({self._text_value})"

    # .................................................................................................................

    def set_prefix_suffix(self, new_prefix=None, new_suffix=None):

        if new_prefix is not None:
            self._prefix = str(new_prefix)

        if new_suffix is not None:
            self._suffix = str(new_suffix)

        return self

    def set_value(self, value):
        self._text_value = f"{self._prefix}{value}{self._suffix}"
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
