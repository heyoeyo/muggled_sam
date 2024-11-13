#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2

from .base import BaseCallback, force_same_min_width
from .images import ExpandingImage
from .helpers.text import TextDrawer
from .helpers.images import blank_image, draw_box_outline, convert_color


# ---------------------------------------------------------------------------------------------------------------------
# %% Base classes


class Toggleable(BaseCallback):

    # .................................................................................................................

    def __init__(self, min_h, min_w, default_state: bool):

        # Inherit from parent
        super().__init__(min_h, min_w)

        # Storage for toggle state
        self._is_changed = False
        self._is_on = default_state

        # Storage for on-change callbacks
        self._on_change_callbacks = []

    # .................................................................................................................

    def toggle(self, new_state: bool | None = None, *, flag_if_changed=True) -> tuple[bool, bool]:
        old_is_on = self._is_on
        self._is_on = not self._is_on if new_state is None else new_state

        # Check for change state if needed and trigger on-change callbacks
        if flag_if_changed:
            self._is_changed |= self._is_on != old_is_on
            if self._is_changed:
                for on_change_cb in self._on_change_callbacks:
                    on_change_cb(self._is_on)

        return self._is_on

    # .................................................................................................................

    def set_is_changed(self, is_changed=True):
        """Helper used to artificially toggle is_changed flag, useful for forcing read updates (e.g. on startup)"""
        self._is_changed = is_changed
        return self

    # .................................................................................................................

    def read(self) -> tuple[bool, bool]:
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._is_on

    # .................................................................................................................

    def add_on_change_listeners(self, *callbacks: callable, call_on_add=True):
        """
        Add callbacks which will run whenever the button state changes
        The callback is expected to have one input argument for receiving
        the toggle button state, and no outputs!
        Example:
            btn = ToggleButton("Example")
            example_callback = lambda is_on: print("Button is on:", is_on)
            btn.add_on_change_listeners(example_callback)
        """

        self._on_change_callbacks.extend(callbacks)
        if call_on_add:
            for cb in callbacks:
                cb(self._is_on)

        return self

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class ToggleButton(Toggleable):

    # .................................................................................................................

    def __init__(
        self,
        label: str,
        default_state=False,
        on_color=(80, 80, 80),
        off_color=None,
        button_height=40,
        text_scale=0.75,
        text_on_color=(255, 255, 255),
        text_off_color=(120, 120, 120),
    ):

        # Figure out off color, if needed
        if off_color is None:
            h_on, s_on, v_on = convert_color(on_color, cv2.COLOR_BGR2HSV_FULL)
            hsv_off_color = (h_on, s_on * 0.4, v_on * 0.5)
            off_color = convert_color(hsv_off_color, cv2.COLOR_HSV2BGR_FULL)

        # Store visual settings
        self._label = f" {label} "
        self._color_on = on_color
        self._color_off = off_color
        self._txt_bright = TextDrawer(scale=text_scale).style(color=text_on_color)

        # Make sure our text sizing fits in the given bar height
        _, txt_h, _ = self._txt_bright.get_text_size(self._label)
        if txt_h > button_height:
            new_scale = text_scale * (button_height / txt_h) * 0.8
            self._txt_bright.style(scale=new_scale)
        btn_w, _, _ = self._txt_bright.get_text_size(self._label)
        self._txt_dim = TextDrawer.from_existing(self._txt_bright).style(color=text_off_color)

        # Inherit from parent & set default helper name for debugging
        super().__init__(button_height, btn_w, default_state)
        self.set_debug_name(f"{label} (ToggleBtn)")

    # .................................................................................................................

    def style(self, on_color=None, off_color=None, text_scale=None, text_on_color=None, text_off_color=None):

        if on_color is not None:
            self._color_on = on_color
        if off_color is not None:
            self._color_off = off_color
        if text_scale is not None:
            self._txt_bright.style(scale=text_scale)
            self._txt_dim.style(scale=text_scale)
        if text_on_color is not None:
            self._txt_bright.style(color=text_on_color)
        if text_off_color is not None:
            self._txt_dim.style(color=text_off_color)

        return self

    # .................................................................................................................

    @classmethod
    def many(
        cls,
        *labels: list[str],
        default_state=False,
        on_color=(80, 80, 80),
        off_color=None,
        button_height=40,
        text_scale=0.75,
        all_same_width=True,
    ):
        """Helper used to create multiple toggle buttons of the same style, all at once"""

        # Make sure labels iterable is held as a list of strings
        labels = [labels] if isinstance(labels, str) else [str(label) for label in labels]

        # Create toggle buttons, but force buttons to have same width (matched to max size)
        btns = [cls(label, default_state, on_color, off_color, button_height, text_scale) for label in labels]
        if all_same_width:
            force_same_min_width(*btns)

        return btns

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        # Create blank background in on/off color to represent button state
        btn_color = self._color_on if self._is_on else self._color_off
        image = blank_image(h, w, btn_color)

        # Draw button label
        is_hovered = self.is_hovered()
        txtdraw = self._txt_bright if self._is_on or is_hovered else self._txt_dim
        image = txtdraw.xy_centered(image, self._label)
        box_color = (255, 255, 255) if (self._is_on and is_hovered) else (0, 0, 0)
        return draw_box_outline(image, box_color)

    # .................................................................................................................

    def on_left_click(self, cbxy, cbflags) -> None:
        self.toggle()

    # .................................................................................................................


class ToggleImage(ExpandingImage):

    # .................................................................................................................

    def __init__(self, image=None, default_state=False, highlight_color=(255, 255, 255)):

        self._highlight_color = highlight_color

        # Pick black/white as background color, depending on highlight brightness
        col_hue, col_sat, col_val = convert_color(highlight_color, cv2.COLOR_BGR2HSV_FULL)
        bg_hsv = (0, 0, 0 if col_val > 127 else 255)
        self._highlight_bg_color = convert_color(bg_hsv, cv2.COLOR_HSV2BGR_FULL)

        # Use a dimmer, desaturated version of the highlight when hovering
        hover_hsv = (col_hue, round(col_sat * 0.75), round(col_val * 0.75))
        self._hover_color = convert_color(hover_hsv, cv2.COLOR_HSV2BGR_FULL)

        # Fill in default blank image if we aren't given anything
        if image is None:
            image = blank_image(128, 128, (127, 127, 127))

        # Storage for overlaying text onto the image, if configured
        self._text = None
        self._txt_xy_norm = (0.5, 0.5)
        self._txt_anchor_xy_norm = None
        self._txt_offset_xy_px = (0, 0)
        self._txtdraw = TextDrawer()

        # Storage for toggle state
        self._toggable = Toggleable(1, 1, default_state)

        # Inherit from parent
        super().__init__(image)

    # .................................................................................................................

    @classmethod
    def many(cls, *images, default_state=False, highlight_color=(255, 255, 255)):
        """Helper used to create multiple toggle images of the same style, all at once"""
        return [cls(img, default_state, highlight_color) for img in images]

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        disp_image = super()._render_up_to_size(h, w).copy()

        # Draw a box to indicate on/hovered status
        is_on = self._toggable._is_on
        is_hovered = self.is_hovered()
        needs_box = is_hovered or is_on
        if needs_box:
            box_color = self._highlight_color if is_on else self._hover_color
            if is_on:
                disp_image = draw_box_outline(disp_image, self._highlight_bg_color, thickness=2)
            disp_image = draw_box_outline(disp_image, color=box_color)

        # Draw text overlay
        if self._text is not None:
            self._txtdraw.xy_norm(
                disp_image, self._text, self._txt_xy_norm, self._txt_anchor_xy_norm, self._txt_offset_xy_px
            )

        return disp_image

    # .................................................................................................................

    # 'Inherit' methods from togglable class
    # -> Doing it this way because we already need to inherit from the image class
    # -> Need to reconsider how functionality is structured/inherited
    #   -> Makes sense to think of rendering/callbacks/sizing as separate things?
    #   -> Each 'element' could take in instantiated classes which only handle visual/non-visual capabilities?

    def toggle(self, new_state: bool | None = None, *, flag_if_changed=True) -> tuple[bool, bool]:
        return self._toggable.toggle(new_state, flag_if_changed=flag_if_changed)

    def read(self) -> tuple[bool, bool]:
        return self._toggable.read()

    def add_on_change_listeners(self, *callbacks: callable, call_on_add=True):
        return self._toggable.add_on_change_listeners(*callbacks, call_on_add=call_on_add)

    # .................................................................................................................

    def on_left_click(self, cbxy, cbflags) -> None:
        self._toggable.toggle()

    # .................................................................................................................

    def set_text(
        self,
        text=None,
        scale=None,
        thickness=None,
        color=None,
        bg_color=None,
        xy_norm=None,
        anchor_xy_norm=None,
        offset_xy_px=None,
    ):
        """Set overlay text to display on top of toggle image"""

        # Update text & drawing style
        self._text = text
        self._txtdraw.style(scale, thickness, color, bg_color)

        # Update text positioning
        if xy_norm is not None:
            self._txt_xy_norm = xy_norm
        if anchor_xy_norm is not None:
            self._txt_anchor_xy_norm = anchor_xy_norm if anchor_xy_norm != -1 else None
        if offset_xy_px is not None:
            self._txt_offset_xy_px = offset_xy_px

        return self

    # .................................................................................................................


class ImmediateButton(BaseCallback):

    # .................................................................................................................

    def __init__(self, label, color=(70, 120, 140), button_height=40, text_scale=0.75):

        # Storage for button state
        self._is_changed = False

        # Store visual settings
        self._label = f" {label} "
        self._color = color
        self._hover_color = (255, 255, 255)
        self._txtdraw = TextDrawer(scale=text_scale)

        # Make sure our text sizing fits in the given bar height
        _, txt_h, _ = self._txtdraw.get_text_size(self._label)
        if txt_h > button_height:
            new_scale = text_scale * (button_height / txt_h) * 0.8
            self._txtdraw.style(scale=new_scale)
        btn_w, _, _ = self._txtdraw.get_text_size(self._label)

        # Inherit from parent & set default helper name for debugging
        super().__init__(button_height, btn_w)
        self.set_debug_name(f"{label} (ImmediateBtn)")

    # .................................................................................................................

    def style(self, color=None, text_scale=None):

        if color is not None:
            self._color = color
        if text_scale is not None:
            self._txtdraw.style(scale=text_scale)

        return self

    # .................................................................................................................

    @classmethod
    def many(cls, *labels: list[str], color=(70, 120, 140), button_height=40, text_scale=0.75, all_same_width=True):
        """Helper used to create multiple immediate buttons of the same style, all at once"""

        # Make sure labels iterable is held as a list of strings
        labels = [labels] if isinstance(labels, str) else [str(label) for label in labels]

        # Create buttons, but force them to have same width (matched to largest button)
        btns = [cls(label, color, button_height, text_scale) for label in labels]
        if all_same_width:
            force_same_min_width(*btns)

        return btns

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        # Draw button with label
        image = blank_image(h, w, self._color)
        image = self._txtdraw.xy_centered(image, self._label)
        box_color = self._hover_color if self.is_hovered() else (0, 0, 0)
        return draw_box_outline(image, box_color)

    # .................................................................................................................

    def read(self) -> bool:
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed

    def click(self):
        self._is_changed = True
        return self

    # .................................................................................................................

    def on_left_click(self, cbxy, cbflags) -> None:
        self.click()

    # .................................................................................................................

    def set_is_changed(self, is_changed=True):
        """Helper used to artificially toggle is_changed flag, useful for forcing read updates (e.g. on startup)"""
        self._is_changed = is_changed
        return self

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Constraints


class RadioConstraint:

    # .................................................................................................................

    def __init__(self, *items, initial_selected_index=0):

        # Storage for radio state
        self._is_changed = True
        self._select_idx = None

        # Skip items that do not have the require toggling methods
        self._items = []
        required_methods = ["add_on_change_listeners", "read", "toggle"]
        for item in items:

            # Skip items that do not have the required methods
            has_methods = all(hasattr(item, method) for method in required_methods)
            if not has_methods:
                print(
                    f"WARNING: Cannot use radio constraint on item: {item}",
                    "Missing required method(s):",
                    *required_methods,
                    sep="\n",
                )
                continue

            self._items.append(item)

        # Turn off all but 1 item on startup
        self._select_idx = initial_selected_index % len(self._items)
        for item_idx, item in enumerate(self._items):
            item.toggle(item_idx == self._select_idx, flag_if_changed=False)

        # Add on-change callback to enforce radio constraint when items change state
        for item in self._items:
            item.add_on_change_listeners(self._enforce_constraint)

        pass

    # .................................................................................................................

    def _enforce_constraint(self, item_is_on=None) -> None:

        # Check if there has been a change in the state of any item, going from off to on
        idx_to_set = self._select_idx
        for item_idx, item in enumerate(self._items):
            _, item_is_on = item.read()
            new_toggled_on = item_is_on and (item_idx != self._select_idx)
            if new_toggled_on:
                idx_to_set = item_idx

        # Record changes to overal radio state and index of the 'on-item'
        self._is_changed |= idx_to_set != self._select_idx
        self._select_idx = idx_to_set

        # Force all but one item to be on
        for item_idx, item in enumerate(self._items):
            item.toggle(item_idx == self._select_idx, flag_if_changed=True)

        return

    # .................................................................................................................

    def read(self) -> tuple[bool, int, Toggleable]:
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._select_idx, self._items[self._select_idx]

    # .................................................................................................................

    def change_to(self, item_index_or_instance):

        if isinstance(item_index_or_instance, int):
            item_index = item_index_or_instance
        else:
            # Assume we got an instance of one of the items
            try:
                item_index = self._items.index(item_index_or_instance)
            except ValueError:
                # Occurs if the item isn't in the items list
                item_index = self._select_idx
                print(
                    "",
                    "WARNING:",
                    "  Couldn't change radio constraint selection",
                    f"  Item part of constraint: {item_index_or_instance}",
                    sep="\n",
                )

        old_idx = self._select_idx
        self._select_idx = item_index % len(self._items)
        is_changed = old_idx != self._select_idx
        if is_changed:
            # self._items[item_index].toggle(True)
            self._items[old_idx].toggle(False)
            self._enforce_constraint()
            self._is_changed = True

        return self._items[self._select_idx]

    # .................................................................................................................

    def next(self, allow_wrap_around=True):

        curr_idx = self._select_idx
        next_idx = curr_idx + 1

        num_items = len(self._items)
        next_idx = next_idx % num_items if allow_wrap_around else min(next_idx, num_items - 1)

        self._is_changed |= curr_idx != next_idx
        if self._is_changed:
            self.change_to(next_idx)

        return self

    # .................................................................................................................

    def previous(self, allow_wrap_around=True):

        curr_idx = self._select_idx
        prev_idx = curr_idx - 1

        num_items = len(self._items)
        prev_idx = prev_idx % num_items if allow_wrap_around else max(prev_idx, 0)

        self._is_changed |= curr_idx != prev_idx
        if self._is_changed:
            self.change_to(prev_idx)

        return self

    # .................................................................................................................
