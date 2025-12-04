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


class HSlider(BaseCallback):

    # .................................................................................................................

    def __init__(
        self,
        label: str,
        initial_value: float = 0.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        step_size: float = 0.05,
        bar_height: int = 40,
        bar_bg_color=(40, 40, 40),
        indicator_line_width: int = 1,
        text_scale: float = 0.5,
        marker_steps: None | int | float = None,
        enable_value_display: bool = True,
    ):

        # Make sure the given values make sense
        min_value, max_value = sorted((min_value, max_value))
        initial_value = min(max_value, max(min_value, initial_value))

        # Storage for slider value
        self._label = label
        self._initial_value = initial_value
        self._slider_value = initial_value
        self._slider_min = min_value
        self._slider_max = max_value
        self._slider_step = step_size
        self._slider_delta = max(self._slider_max - self._slider_min, 1e-9)

        # Storage for slider state
        self._x_px = 0
        self._is_changed = False

        # Display config
        self._enable_value_display = enable_value_display
        self._bar_bg_color = bar_bg_color
        self._indicator_thickness = indicator_line_width
        self._base_image = blank_image(1, 1)
        self._txtbright = TextDrawer(scale=text_scale, color=(255, 255, 255))

        # Set up value display string formatting
        val_width = max(len(str(val)) for val in [min_value, max_value])
        dec_width = 0 if "." not in str(step_size) else len(str(step_size).split(".")[-1])
        self._max_precision = val_width + dec_width

        # Pre-compute marker drawing locations
        marker_x_norm = []
        if marker_steps is not None:
            marker_step_size = marker_steps * step_size
            marker_min = marker_step_size * (self._slider_min // marker_step_size)
            marker_max = marker_step_size * (2 + self._slider_max // marker_step_size)
            marker_pts = np.arange(marker_min, marker_max, marker_step_size)
            marker_x_norm = (marker_pts - self._slider_min) / self._slider_delta
        self._marker_x_norm = marker_x_norm
        self._marker_color = (70, 70, 70)

        # Make sure our text sizing fits in the given bar height
        _, txt_h, _ = self._txtbright.get_text_size(self._label)
        if txt_h > bar_height:
            new_scale = text_scale * (bar_height / txt_h) * 0.8
            self._txtbright.style(scale=new_scale)
        label_w, _, _ = self._txtbright.get_text_size(self._label)
        self._txtdim = TextDrawer.from_existing(self._txtbright).style(color=(120, 120, 120))

        # Inherit from parent & set default helper name for debugging
        super().__init__(bar_height, label_w, expand_w=True)
        self.set_debug_name(f"{label} (HSlider)")

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-render label & marker lines onto new blank background if render size changes
        base_h, base_w = self._base_image.shape[0:2]
        if base_h != h or base_w != w:
            new_base = blank_image(h, w, self._bar_bg_color)
            self._txtdim.xy_norm(new_base, self._label, (0, 0.5), offset_xy_px=(5, 0))
            for x_norm in self._marker_x_norm:
                x_px = round(w * x_norm)
                cv2.line(new_base, (x_px, 6), (x_px, h - 7), self._marker_color, 1, cv2.LINE_4)
            self._base_image = new_base

        # Draw indicator line
        img = self._base_image.copy()
        slider_norm = (self._slider_value - self._slider_min) / self._slider_delta
        line_x_px = round(slider_norm * (w - 1))
        img = cv2.line(img, (line_x_px, -1), (line_x_px, h + 1), (255, 255, 255), self._indicator_thickness)

        # Draw text beside indicator line to show current value if needed
        if self._enable_value_display:
            value_str = f"{float(self._slider_value):.{self._max_precision}g}"
            txt_w, txt_h, txt_baseline = self._txtbright.get_text_size(value_str)

            # Draw the text to the left or right of the indicator line, depending on where the image border is
            is_near_right_edge = line_x_px + txt_w + 10 > w
            anchor_xy_norm = (1, 0.5) if is_near_right_edge else (0, 0.5)
            offset_xy_px = (-5, 0) if is_near_right_edge else (5, 0)
            self._txtbright.xy_norm(img, value_str, (slider_norm, 0.5), anchor_xy_norm, offset_xy_px)

        return draw_box_outline(img, (0, 0, 0))

    # .................................................................................................................

    def read(self) -> tuple[bool, float | int]:
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._slider_value

    def set(self, slider_value, use_as_default_value=True):
        new_value = max(self._slider_min, min(self._slider_max, slider_value))
        if use_as_default_value:
            self._initial_value = new_value
        self._is_changed |= new_value != self._slider_value
        self._slider_value = new_value
        return self

    def increment(self, num_increments=1):
        return self.set(self._slider_value + self._slider_step * num_increments, use_as_default_value=False)

    def decrement(self, num_decrements=1):
        return self.set(self._slider_value - self._slider_step * num_decrements, use_as_default_value=False)

    def set_is_changed(self, is_changed=True):
        """Helper used to artificially toggle is_changed flag, useful for forcing read updates (e.g. on startup)"""
        self._is_changed = is_changed
        return self

    # .................................................................................................................

    def on_left_down(self, cbxy, cbflags) -> None:

        # Ignore clicks outside of the slider
        if not cbxy.is_in_region:
            return

        # Record changes
        x_px = cbxy.xy_px[0]
        self._is_changed |= x_px != self._x_px
        if self._is_changed:
            self._x_px = x_px
            self._mouse_x_norm_to_slider_value(cbxy.xy_norm[0])

        return

    def on_drag(self, cbxy, cbflags) -> None:

        # Update slider value while dragging
        x_px = cbxy.xy_px[0]
        self._is_changed |= x_px != self._x_px
        if self._is_changed:
            self._x_px = x_px
            self._mouse_x_norm_to_slider_value(cbxy.xy_norm[0])

        return

    def on_right_click(self, cbxy, cbflags) -> None:

        # Reset slider position on right click
        old_value = self._slider_value
        self._is_changed = old_value != self._initial_value
        if self._is_changed:
            self._slider_value = self._initial_value
            self._x_px = int(round(cbxy.xy_norm[0] * (cbxy.hw_px[1] - 1)))

        return

    # .................................................................................................................

    def _mouse_x_norm_to_slider_value(self, x_norm: float) -> float | int:
        """Helper used to convert normalized mouse position into slider values"""

        # Map normalized x position to slider range, snapped to step increments
        slider_x = (x_norm * self._slider_delta) + self._slider_min
        slider_x = round(slider_x / self._slider_step) * self._slider_step

        # Finally, make sure the slider value doesn't go out of range
        self._slider_value = max(self._slider_min, min(self._slider_max, slider_x))

        return self._slider_value

    # .................................................................................................................


class HMultiSlider(BaseCallback):

    # .................................................................................................................

    def __init__(
        self,
        label: str,
        initial_values: int | float | list[int | float],
        min_value: float = 0.0,
        max_value: float = 1.0,
        step_size: float = 0.05,
        bar_height: int = 40,
        bar_bg_color=(40, 40, 40),
        indicator_line_width: int = 1,
        text_scale: float = 0.5,
        marker_steps: None | int | float = None,
        enable_value_display: bool = True,
        fill_between_points=False,
        fill_between_color=(240, 200, 120),
    ):

        # Force to list-type, so we can handle single values as if they are multi-values
        if isinstance(initial_values, (int, float)):
            initial_values = [initial_values]

        # Make sure the given values make sense
        min_value, max_value = sorted((min_value, max_value))
        initial_values = np.clip(np.array(sorted(initial_values)), min_value, max_value)

        # Storage for slider value
        self._label = label
        self._initial_values = initial_values.copy()
        self._slider_values = initial_values.copy()
        self._slider_min = min_value
        self._slider_max = max_value
        self._slider_step = step_size
        self._slider_delta = max(self._slider_max - self._slider_min, 1e-9)

        # Storage for slider state
        self._drag_idx = 0
        self._last_x_px = 0
        self._is_changed = True

        # Display config
        self._enable_value_display = enable_value_display
        self._bar_bg_color = bar_bg_color
        self._indicator_thickness = indicator_line_width
        self._base_image = blank_image(1, 1)
        self._txtbright = TextDrawer(scale=text_scale, color=(255, 255, 255))
        self._fill_between_points = fill_between_points
        self._fill_between_color = tuple(val // 2 for val in fill_between_color)

        # Set up value display string formatting
        val_width = max(len(str(val)) for val in [min_value, max_value])
        dec_width = 0 if "." not in str(step_size) else len(str(step_size).split(".")[-1])
        self._max_precision = val_width + dec_width

        # Pre-compute marker drawing locations
        marker_x_norm = []
        if marker_steps is not None:
            marker_step_size = marker_steps * step_size
            marker_min = marker_step_size * (self._slider_min // marker_step_size)
            marker_max = marker_step_size * (2 + self._slider_max // marker_step_size)
            marker_pts = np.arange(marker_min, marker_max, marker_step_size)
            marker_x_norm = (marker_pts - self._slider_min) / self._slider_delta
        self._marker_x_norm = marker_x_norm
        self._marker_color = (70, 70, 70)

        # Make sure our text sizing fits in the given bar height
        _, txt_h, _ = self._txtbright.get_text_size(self._label)
        if txt_h > bar_height:
            new_scale = text_scale * (bar_height / txt_h) * 0.8
            self._txtbright.style(scale=new_scale)
        label_w, _, _ = self._txtbright.get_text_size(self._label)
        self._txtdim = TextDrawer.from_existing(self._txtbright).style(color=(120, 120, 120))

        # Inherit from parent & set default helper name for debugging
        super().__init__(bar_height, label_w, expand_w=True)
        self.set_debug_name(f"{label} (HMultiSlider)")

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-render label & marker lines onto new blank background if render size changes
        base_h, base_w = self._base_image.shape[0:2]
        if base_h != h or base_w != w:
            new_base = blank_image(h, w, self._bar_bg_color)
            self._txtdim.xy_norm(new_base, self._label, (0, 0.5), offset_xy_px=(5, 0))
            for x_norm in self._marker_x_norm:
                x_px = round(w * x_norm)
                cv2.line(new_base, (x_px, 6), (x_px, h - 7), self._marker_color, 1, cv2.LINE_4)
            self._base_image = new_base

        # Draw inner region if needed
        img = self._base_image.copy()
        if self._fill_between_points:

            # Figure out highlight region bounds
            x1_norm = (np.min(self._slider_values) - self._slider_min) / self._slider_delta
            x2_norm = (np.max(self._slider_values) - self._slider_min) / self._slider_delta
            x1_px = round(x1_norm * (w - 1))
            x2_px = round(x2_norm * (w - 1))

            # Mix in another color to indicator highlighted region
            fill_region = np.full((h, (x2_px - x1_px), 3), self._fill_between_color, dtype=np.uint8)
            img[:, x1_px:x2_px, :] = fill_region + (img[:, x1_px:x2_px, :] // 2)

        # Draw indicator lines
        for value in self._slider_values:

            # Draw indicator lines
            value_norm = (value - self._slider_min) / self._slider_delta
            line_x_px = round(value_norm * (w - 1))
            img = cv2.line(img, (line_x_px, -1), (line_x_px, h + 1), (255, 255, 255), self._indicator_thickness)

            # Draw text beside indicator line to show current value if needed
            if self._enable_value_display:
                value_str = f"{float(value):.{self._max_precision}g}"
                txt_w, txt_h, txt_baseline = self._txtbright.get_text_size(value_str)

                # Draw the text to the left or right of the indicator line, depending on where the image border is
                is_near_right_edge = line_x_px + txt_w + 10 > w
                anchor_xy_norm = (1, 0.5) if is_near_right_edge else (0, 0.5)
                offset_xy_px = (-5, 0) if is_near_right_edge else (5, 0)
                self._txtbright.xy_norm(img, value_str, (value_norm, 0.5), anchor_xy_norm, offset_xy_px)

        return draw_box_outline(img, (0, 0, 0))

    # .................................................................................................................

    def read(self) -> tuple[bool, float | int]:
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, np.sort(self._slider_values).tolist()

    def set(self, new_values, use_as_default_values=True):

        # Force new values to array-like data, within slider range
        if isinstance(new_values, (int, float)):
            new_values = [new_values]
        new_values = np.clip(np.array(sorted(new_values)), self._slider_min, self._slider_max)

        # Use new values as default if needed
        if use_as_default_values:
            self._initial_values = new_values.copy()

        # Check if new values are actually different and store
        self._is_changed |= np.allclose(new_values, self._slider_values)
        self._slider_values = new_values

        return self

    def set_is_changed(self, is_changed=True):
        """Helper used to artificially toggle is_changed flag, useful for forcing read updates"""
        self._is_changed = is_changed
        return self

    # .................................................................................................................

    def on_left_down(self, cbxy, cbflags) -> None:
        """Update closest slider point on click and record index for dragging"""

        # Ignore clicks outside of the slider
        if not cbxy.is_in_region:
            return

        # Update closest click, redardless of whether we actually change values
        # (this is important for dragging to work properly)
        new_slider_value = self._mouse_x_norm_to_slider_value(cbxy.xy_norm[0])
        closest_idx = np.argmin((np.abs(self._slider_values - new_slider_value)))
        self._drag_idx = closest_idx

        # Record changes
        x_px = cbxy.xy_px[0]
        self._is_changed |= x_px != self._last_x_px
        if self._is_changed:
            self._last_x_px = x_px
            self._slider_values[closest_idx] = new_slider_value

        return

    def on_drag(self, cbxy, cbflags) -> None:
        """Update a single slider point on drag (determined by closest on left click)"""

        # Update slider value while dragging
        x_px = cbxy.xy_px[0]
        self._is_changed |= x_px != self._last_x_px
        if self._is_changed:
            self._last_x_px = x_px
            new_slider_value = self._mouse_x_norm_to_slider_value(cbxy.xy_norm[0])
            self._slider_values[self._drag_idx] = new_slider_value

        return

    def on_left_up(self, cbxy, cbflags) -> None:
        """For slight efficiency gain, sort values after modifications are complete"""
        self._slider_values = np.sort(self._slider_values)
        return

    def on_right_click(self, cbxy, cbflags) -> None:
        """Reset slider position on right click"""

        self._is_changed = not np.allclose(self._slider_values, self._initial_values)
        if self._is_changed:
            self._slider_values = self._initial_values.copy()
            self._last_x_px = int(round(cbxy.xy_norm[0] * (cbxy.hw_px[1] - 1)))

        return

    # .................................................................................................................

    def _mouse_x_norm_to_slider_value(self, x_norm: float) -> float | int:
        """Helper used to convert normalized mouse position into slider values"""

        # Map normalized x position to slider range, snapped to step increments
        slider_x = (x_norm * self._slider_delta) + self._slider_min
        slider_x = round(slider_x / self._slider_step) * self._slider_step
        slider_x = max(self._slider_min, min(self._slider_max, slider_x))

        return slider_x

    # .................................................................................................................
