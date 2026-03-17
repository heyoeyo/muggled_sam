#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .base import BaseCallback
from .helpers.text import TextDrawer
from .helpers.images import blank_image, draw_box_outline

# Typing
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class LossPlot(BaseCallback):
    """UI element used to display a running (training) loss plot"""

    # .................................................................................................................

    def __init__(
        self,
        title: str = "Loss",
        bg_color=(50, 45, 40),
        line_color=(75, 185, 225),
        line_width: int = 1,
        axis_color=(255, 255, 255),
        axis_width: int = 2,
        xy_margin=(8, 10),
        text_scale: float = 0.5,
        min_side_length: int = 256,
    ):

        # Storage for slider value
        self._title = title

        # Storage for slider state
        self._plot_data_norm = np.float32((0, 0))
        self._plot_x_norm = np.float32((0, 1))
        self._is_plot_data_changed = False
        self._plot_min = 0
        self._plot_max = 1

        # Display config
        self._bg_color = bg_color
        self._line_color = line_color
        self._axis_color = axis_color
        self._line_thickness = line_width
        self._axis_thickness = axis_width
        self._base_image = blank_image(1, 1)
        self._curr_image = blank_image(1, 1)
        self._txtdraw = TextDrawer(scale=text_scale, color=axis_color)
        self._line_type = cv2.LINE_AA

        # Positioning config
        self._title_y_margin = 8
        self._xy_margin = xy_margin
        self._plot_xy1 = (0, 0)
        self._plot_xy2 = (1, 1)

        # Inherit from parent
        super().__init__(min_side_length, min_side_length, expand_h=True, expand_w=True)

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-render label & marker lines onto new blank background if render size changes
        base_h, base_w = self._base_image.shape[0:2]
        is_sized_changed = base_h != h or base_w != w
        if is_sized_changed:
            new_base = blank_image(h, w, self._bg_color)

            x_margin, y_margin = self._xy_margin
            self._txtdraw.xy_norm(new_base, self._title, (0.5, 0), offset_xy_px=(0, self._title_y_margin))
            txt_w, txt_h, txt_base = self._txtdraw.get_text_size(self._title)

            # Update margins
            plot_x1, plot_x2 = x_margin, w - x_margin
            plot_y1, plot_y2 = max(self._title_y_margin + txt_h + txt_base, y_margin), h - y_margin
            self._plot_xy1 = (plot_x1, plot_y1)
            self._plot_xy2 = (plot_x2, plot_y2)
            self._base_image = new_base

        # Update plot graphics
        img = None
        if self._is_plot_data_changed or is_sized_changed:
            x1, y1 = self._plot_xy1
            x2, y2 = self._plot_xy2
            plot_w, plot_h = (x2 - x1), (y2 - y1)

            xy_norm = np.column_stack((self._plot_x_norm, 1.0 - self._plot_data_norm))
            xy_px = np.round(np.float32((plot_w - 1, plot_h - 1)) * xy_norm).astype(np.int32)
            xy_px += np.int32((x1, y1))

            # Draw plot line
            img = self._base_image.copy()
            cv2.polylines(img, [xy_px], False, self._line_color, self._line_thickness, self._line_type)

            # Draw max value indicator
            max_txt = f"{self._plot_max:.1e}"
            _, _, txt_base = self._txtdraw.get_text_size(max_txt)
            self._txtdraw.xy_px(img, max_txt, (x1, y1 - txt_base), color=self._line_color)

            # Draw x/y axis
            cv2.line(img, (x1, y1), (x1, y2), self._axis_color, self._axis_thickness)
            cv2.line(img, (x1, y2), (x2, y2), self._axis_color, self._axis_thickness)

            self._curr_image = img.copy()
            self._is_plot_data_changed = False
        else:
            img = self._curr_image.copy()

        return draw_box_outline(img, (0, 0, 0))

    # .................................................................................................................

    def set_plot_data(self, plot_data, plot_min: float | None = 0, plot_max: float | None = None):

        # Fill in 'blank' data so we have something to plot
        if len(plot_data) == 0:
            plot_data = (0, 0)

        plot_data = np.float32(plot_data)
        self._plot_min = np.min(plot_data) if plot_min is None else float(plot_min)
        self._plot_max = np.max(plot_data) if plot_max is None else float(plot_max)
        if self._plot_max <= self._plot_min:
            self._plot_max = self._plot_min + 1.0
        self._plot_data_norm = (plot_data - self._plot_min) / (self._plot_max - self._plot_min)
        self._plot_x_norm = np.linspace(0, 1, len(self._plot_data_norm), dtype=np.float32)
        self._is_plot_data_changed = True

        return self

    # .................................................................................................................

    def _get_width_given_height(self, h):
        return h

    def _get_height_given_width(self, w):
        return w

    def _get_height_and_width_without_hint(self):
        return self._rdr.limits.min_h, self._rdr.limits.min_w

    # .................................................................................................................


class ScoresPlot(BaseCallback):
    """UI element used to display model score predictions"""

    # .................................................................................................................

    def __init__(
        self,
        title: str = "Scores",
        bg_color=(50, 45, 40),
        test_color=(80, 255, 0),
        true_color=(250, 10, 110),
        axis_color=(255, 255, 255),
        axis_width: int = 2,
        bar_width_pct: tuple[int, int] = (100, 100),
        xy_margin=(8, 10),
        text_scale: float = 0.5,
        use_log_scale: bool = False,
        min_side_length: int = 256,
    ):

        # Storage for slider value
        self._title = title

        # Storage for slider state
        self._true_data = np.float32([1])
        self._true_xbounds_norm = np.float32((0, 1))
        self._test_data = np.float32([0])
        self._test_xbounds_norm = np.float32((0, 1))
        self._is_true_data_changed = False
        self._is_test_data_changed = False
        self._use_log_scale = use_log_scale

        # Display config
        self._bg_color = bg_color
        self._test_data_color = test_color
        self._true_data_color = true_color
        self._axis_color = axis_color
        self._axis_thickness = axis_width
        self._base_image = blank_image(1, 1)
        self._curr_image = blank_image(1, 1)
        self._txtdraw = TextDrawer(scale=text_scale, color=axis_color)

        # Positioning config
        true_w_pct, test_w_pct = bar_width_pct
        self._true_w_norm = true_w_pct / 100
        self._test_w_norm = test_w_pct / 100
        self._title_y_margin = 8
        self._xy_margin = xy_margin
        self._plot_xy1 = (0, 0)
        self._plot_xy2 = (1, 1)

        # Inherit from parent
        super().__init__(min_side_length, min_side_length, expand_h=True, expand_w=True)

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-render label & marker lines onto new blank background if render size changes
        base_h, base_w = self._base_image.shape[0:2]
        is_sized_changed = base_h != h or base_w != w
        if self._is_true_data_changed or is_sized_changed:
            new_base = blank_image(h, w, self._bg_color)

            x_margin, y_margin = self._xy_margin
            self._txtdraw.xy_norm(new_base, self._title, (0.5, 0), offset_xy_px=(0, self._title_y_margin))
            txt_w, txt_h, txt_base = self._txtdraw.get_text_size(self._title)

            # Update margins
            plot_x1, plot_x2 = x_margin, w - x_margin
            plot_y1, plot_y2 = max(self._title_y_margin + txt_h + txt_base, y_margin), h - y_margin
            self._plot_xy1 = (plot_x1, plot_y1)
            self._plot_xy2 = (plot_x2, plot_y2)
            plot_w, plot_h = (plot_x2 - plot_x1), (plot_y2 - plot_y1)

            x_px_f32 = plot_x1 + (plot_w - 1.0) * self._true_xbounds_norm
            y_px = plot_y1 + plot_h - np.round((plot_h - 1.0) * self._true_data).astype(np.int32)

            # Average x-bounds towards center point to shrink bar width
            x1x2_px = np.column_stack((x_px_f32[:-1], x_px_f32[1:]))
            if self._test_w_norm < 1:
                x1x2_px = (x1x2_px * self._true_w_norm) + x1x2_px.mean(-1, keepdims=True) * (1.0 - self._true_w_norm)

            # Draw each score bar
            x1x2_px_list = np.round(x1x2_px).astype(np.int32).tolist()
            for y_idx, (x1_px, x2_px) in enumerate(x1x2_px_list):  # zip(x_px_f32[:-1], x_px_f32[1:])):
                xy1 = (x1_px, y_px[y_idx])
                xy2 = (x2_px, plot_y2)
                cv2.rectangle(new_base, xy1, xy2, self._true_data_color, -1, cv2.LINE_4)

            self._base_image = new_base
            self._is_true_data_changed = False

        # Update plot graphics
        img = None
        if self._is_test_data_changed or is_sized_changed:
            x1, y1 = self._plot_xy1
            x2, y2 = self._plot_xy2
            plot_w, plot_h = (x2 - x1), (y2 - y1)

            x_px_f32 = x1 + (plot_w - 1.0) * self._test_xbounds_norm
            y_px = y1 + plot_h - np.round((plot_h - 1.0) * self._test_data).astype(np.int32)

            # Average x-bounds towards center point to shrink bar width
            x1x2_px = np.column_stack((x_px_f32[:-1], x_px_f32[1:]))
            if self._test_w_norm < 1:
                x1x2_px = (x1x2_px * self._test_w_norm) + x1x2_px.mean(-1, keepdims=True) * (1.0 - self._test_w_norm)

            # Draw each score bar
            x1x2_px_list = np.round(x1x2_px).astype(np.int32).tolist()
            img = self._base_image.copy()
            for y_idx, (x1_px, x2_px) in enumerate(x1x2_px_list):  # zip(x_px_f32[:-1], x_px_f32[1:])):
                xy1 = (x1_px, y_px[y_idx])
                xy2 = (x2_px, y2)
                cv2.rectangle(img, xy1, xy2, self._test_data_color, -1, cv2.LINE_4)

            # Draw x/y axis
            cv2.line(img, (x1, y2), (x2, y2), self._axis_color, self._axis_thickness)
            self._curr_image = img.copy()
            self._is_test_data_changed = False
        else:
            img = self._curr_image.copy()

        return draw_box_outline(img, (0, 0, 0))

    # .................................................................................................................

    def set_true_data(self, true_scores_norm):

        # Fill in 'blank' data so we have something to plot
        if len(true_scores_norm) == 0:
            true_scores_norm = [0]
        self._true_data = np.float32(true_scores_norm)
        if self._use_log_scale:
            self._true_data = np.log1p(self._true_data * 100) / np.log1p(100.0)
        self._true_xbounds_norm = np.linspace(0, 1, len(self._true_data) + 1, dtype=np.float32)
        self._is_true_data_changed = True

        # Sanity check, if there is a nan value, discard the data
        if np.isnan(self._true_data[0]):
            self._true_data = np.zeros_like(self._true_data)
            print("", "WARNING: True score data contains NaN!", sep="\n", flush=True)
            return

        return self

    # .................................................................................................................

    def set_test_data(self, test_scores_norm):

        # Fill in 'blank' data so we have something to plot
        if len(test_scores_norm) == 0:
            test_scores_norm = [0]
        self._test_data = np.float32(test_scores_norm)
        if self._use_log_scale:
            self._test_data = np.log1p(self._test_data * 100) / np.log1p(100.0)
        self._test_xbounds_norm = np.linspace(0, 1, len(self._test_data) + 1, dtype=np.float32)
        self._is_test_data_changed = True

        # Sanity check, if there is a nan value, discard the data
        if np.isnan(self._test_data[0]):
            self._test_data = np.zeros_like(self._test_data)
            return

        return self

    # .................................................................................................................

    def _get_width_given_height(self, h):
        return h

    def _get_height_given_width(self, w):
        return w

    def _get_height_and_width_without_hint(self):
        return self._rdr.limits.min_h, self._rdr.limits.min_w

    # .................................................................................................................
