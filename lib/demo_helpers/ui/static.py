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


class StaticImage(BaseCallback):

    # .................................................................................................................

    def __init__(self, image, min_scale_factor=0.05, max_scale_factor=None):

        # Store image for re-use when rendering
        image_3ch = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self._image = image_3ch
        self._render_image = image_3ch.copy()
        self._targ_h = 0
        self._targ_w = 0

        # Set up sizing limits
        img_w, img_h = image.shape[0:2]
        min_h = int(img_h * min_scale_factor)
        min_w = int(img_w * min_scale_factor)
        super().__init__(min_h, min_w, expand_h=True, expand_w=True)

        # Add max size limit, if needed
        if max_scale_factor is not None:
            max_h = int(img_h * max_scale_factor)
            max_w = int(img_w * max_scale_factor)
            self._rdr.update(max_h=max_h, max_w=max_w)

        pass

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        h, w = min(h, self._rdr.limits.max_h), min(w, self._rdr.limits.max_w)
        img_h, img_w = self._render_image.shape[0:2]
        if self._targ_h != h or self._targ_w != w:
            fill_h, fill_w = get_image_hw_to_fill(self._image, (h, w))
            self._render_image = cv2.resize(self._image, dsize=(fill_w, fill_h))
            self._targ_h = h
            self._targ_w = w

        return self._render_image

    # .................................................................................................................

    def _get_width_given_height(self, h):
        h = min(h, self._rdr.limits.max_h)
        img_h, img_w = self._image.shape[0:2]
        scaled_w = round(img_w * h / img_h)
        return scaled_w

    def _get_height_given_width(self, w):
        w = min(w, self._rdr.limits.max_w)
        img_h, img_w = self._image.shape[0:2]
        scaled_h = round(img_h * w / img_w)
        return scaled_h

    def _get_height_and_width_without_hint(self):
        img_h, img_w = self._image.shape[0:2]
        return img_h, img_w

    # .................................................................................................................


class StaticMessageBar(BaseCallback):

    # .................................................................................................................

    def __init__(self, *messages, bar_height=40, bar_bg_color=(64, 53, 52), text_scale=0.5, space_equally=False):

        # Store messages with front/back padding for nicer spacing on display (and skip 'None' entries)
        self._msgs_list = [f" {msg}  " for msg in messages if msg is not None]

        # Store visual settings
        self._base_image = blank_image(1, 1, bar_bg_color)
        self._image = self._base_image.copy()
        self._txtdraw = TextDrawer(scale=text_scale)
        c_hue, c_sat, c_val = convert_color(bar_bg_color, cv2.COLOR_BGR2HSV_FULL)
        self._outline_color = convert_color((c_hue, c_sat * 0.75, c_val * 0.75), cv2.COLOR_HSV2BGR_FULL)

        # Make sure our text sizing fits in the given bar height
        _, txt_h, _ = self._txtdraw.get_text_size("".join(self._msgs_list))
        if txt_h > bar_height:
            new_scale = text_scale * (bar_height / txt_h) * 0.8
            self._txtdraw.style(scale=new_scale)

        # Record message widths, used to assign space when minimum drawing size
        msg_widths = [self._txtdraw.get_text_size(m)[0] for m in self._msgs_list]
        total_msg_w = sum(msg_widths)

        # Pre-compute the relative x-positioning of each message for display
        cumulative_w = [sum(msg_widths[:k]) for k in range(len(self._msgs_list))]
        self._msg_x_norms = [(cum_w + 0.5 * msg_w) / total_msg_w for cum_w, msg_w in zip(cumulative_w, msg_widths)]
        if space_equally:
            num_msgs = len(msg_widths)
            self._msg_x_norms = [(k + 0.5) / num_msgs for k in range(num_msgs)]
            total_msg_w = max(msg_widths) * num_msgs
        self._space_equal = space_equally

        # Inherit from parent & render initial image to cache results
        super().__init__(bar_height, total_msg_w)
        self.render(bar_height, total_msg_w)

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        # Re-use stored image, if sizing doesn't change
        img_h, img_w = self._image.shape[0:2]
        if img_h == h and img_w == w:
            return self._image

        # Re-draw base image when sizing has changed
        disp_img = cv2.resize(self._base_image, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        for msg_str, x_norm in zip(self._msgs_list, self._msg_x_norms):
            disp_img = self._txtdraw.xy_norm(disp_img, msg_str, (x_norm, 0.5), anchor_xy_norm=(0.5, 0.5))

        # Draw bounding box and store final result for re-use
        disp_img = draw_box_outline(disp_img, color=self._outline_color)
        self._image = disp_img

        return disp_img

    # .................................................................................................................


class HSeparator(BaseCallback):

    # .................................................................................................................

    def __init__(self, width=2, color=(20, 20, 20)):
        self._color = color
        self._image = blank_image(1, width, color)
        super().__init__(1, width)

    # .................................................................................................................

    @classmethod
    def many(cls, num_separators, width=2, color=(20, 20, 20)):
        return [cls(width, color) for _ in range(num_separators)]

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        # Re-use stored image, if sizing doesn't change, otherwise re-draw it
        img_h, img_w = self._image.shape[0:2]
        if img_h != h or img_w != w:
            self._image = blank_image(h, w, self._color)

        return self._image

    # .................................................................................................................


class VSeparator(BaseCallback):

    # .................................................................................................................

    def __init__(self, height=2, color=(20, 20, 20)):
        self._color = color
        self._image = blank_image(height, 1, color)
        super().__init__(height, 1)

    # .................................................................................................................

    @classmethod
    def many(cls, num_separators, height=2, color=(20, 20, 20)):
        return [cls(height, color) for _ in range(num_separators)]

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        # Re-use stored image, if sizing doesn't change, otherwise re-draw it
        img_h, img_w = self._image.shape[0:2]
        if img_h != h or img_w != w:
            self._image = blank_image(h, w, self._color)

        return self._image

    # .................................................................................................................
