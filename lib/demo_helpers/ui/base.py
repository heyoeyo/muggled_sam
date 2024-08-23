#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
# %% Data types


@dataclass(frozen=True)
class CBEventFlags:

    ctrl_key: bool
    shift_key: bool
    alt_key: bool

    @classmethod
    def create(cls, cv2_flags):
        return cls(
            cv2_flags & cv2.EVENT_FLAG_CTRLKEY,
            cv2_flags & cv2.EVENT_FLAG_SHIFTKEY,
            cv2_flags & cv2.EVENT_FLAG_ALTKEY,
        )


@dataclass(frozen=True)
class CBEventXY:

    global_xy_px: [int, int]
    xy_px: [int, int]
    xy_norm: [float, float]
    hw_px: [int, int]
    is_in_region: bool


@dataclass(frozen=False)
class CBRegion:

    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    w: int = 1
    h: int = 1

    def update(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = max(1, self.x2 - self.x1)
        self.h = max(1, self.y2 - self.y1)
        return self

    def make_cbeventxy(self, global_x_px, global_y_px):
        """Convenience function which does multiple internal conversions (all together for efficiency)"""

        x_px, y_px = global_x_px - self.x1, global_y_px - self.y1
        x_norm = (global_x_px - self.x1) / self.w  # max(1, (self.x2 - self.x1))
        y_norm = (global_y_px - self.y1) / self.h  # max(1, (self.y2 - self.y1))
        is_in_region = (self.x1 <= global_x_px < self.x2) and (self.y1 <= global_y_px < self.y2)

        return CBEventXY((global_x_px, global_y_px), (x_px, y_px), (x_norm, y_norm), (self.h, self.w), is_in_region)


@dataclass(frozen=False)
class CBState:
    """Container for basic callback state"""

    disabled: bool = False
    hovered: bool = False
    left_pressed: bool = False
    middle_pressed: bool = False
    right_pressed: bool = False


@dataclass(frozen=False)
class RenderLimits:

    min_h: int = 0
    min_w: int = 0
    expand_h: bool = False
    expand_w: bool = False
    max_h: int = 4096
    max_w: int = 4096

    def update(self, min_h=None, min_w=None, expand_h=None, expand_w=None, max_h=None, max_w=None):

        if min_h is not None:
            self.min_h = min_h
        if min_w is not None:
            self.min_w = min_w
        if expand_h is not None:
            self.expand_h = expand_h
        if expand_w is not None:
            self.expand_w = expand_w
        if max_h is not None:
            self.max_h = max_h
        if max_w is not None:
            self.max_w = max_w

        return self

    def match_to(self, other_render_limits):
        self.min_h = other_render_limits.min_h
        self.min_w = other_render_limits.min_w
        self.expand_h = other_render_limits.expand_h
        self.expand_w = other_render_limits.expand_w
        self.max_h = other_render_limits.max_h
        self.max_w = other_render_limits.max_w
        return self


@dataclass(frozen=False, kw_only=True)
class RenderTargetSize:

    h: int = 0
    w: int = 0

    def is_match(self, frame_shape):
        frame_h, frame_w = frame_shape[0:2]
        return (frame_h == self.h) and (frame_w == self.w)


@dataclass(frozen=False)
class RenderPaddingStyle:

    color: [int, int, int] = (0, 0, 0)
    style: int = cv2.BORDER_CONSTANT


class BaseRenderable:

    def __init__(self, min_h, min_w, expand_h=False, expand_w=False, max_h=None, max_w=None):
        self.limits = RenderLimits(min_h, min_w, expand_h, expand_w).update(max_h=max_h, max_w=max_w)
        self.size = RenderTargetSize()
        self.pad = RenderPaddingStyle()

    def set_render_size(self, h, w) -> RenderTargetSize:
        self.size.h = h
        self.size.w = w
        return self.size

    def __repr__(self):
        return f"{self.limits}, {self.size}, {self.pad}"


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class BaseCallback:

    def __init__(self, min_h, min_w, expand_h=False, expand_w=False, max_h=None, max_w=None):

        # Storage for rendering info
        self._rdr = BaseRenderable(min_h, min_w, expand_h, expand_w, max_h, max_w)

        # Storage for placement and state of callback
        self._cb_region = CBRegion()
        self._cb_state = CBState()

        # Storage for all child callback items
        self._cb_parent_list: BaseCallback = []
        self._cb_child_list: BaseCallback = []

        # To help with debugging/printouts
        self._debug_name = self.__class__.__name__

    def __repr__(self):
        return self._debug_name

    def enable(self, enable_callback=True):
        self._cb_state.disabled = not enable_callback

    def set_debug_name(self, name):
        self._debug_name = str(name) if name is not None else self.__class__.__name__
        return self

    def append_children(self, *child_items):
        for child in child_items:
            # Skip None entries, which allows for 'disabling' when building UIs
            if child is None:
                continue
            # Assume numpy arrays are meant to be static images
            if isinstance(child, np.ndarray):
                child = BaseImageCallback(child)
            assert isinstance(child, BaseCallback), f"Children must be inherit from: BaseCallback, got: {type(child)}"
            self._cb_child_list.append(child)
            child._cb_parent_list.append(self)
        return self

    def __len__(self):
        return len(self._cb_child_list)

    def __iter__(self):
        return iter(self._cb_child_list)

    def __getitem__(self, index):
        return self._cb_child_list[index]

    def is_hovered(self) -> bool:
        return self._cb_state.hovered

    def on_move(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_drag(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_left_click(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_left_down(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_left_up(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_left_double(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_right_click(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_right_down(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_right_up(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_right_double(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_middle_click(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_middle_down(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_middle_up(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_middle_double(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def on_mouse_wheel(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def __call__(self, event, x, y, flags, params) -> None:

        # Disable callback handling if needed
        if self._cb_state.disabled:
            return

        # Precompute flag states for easier handling
        cbflags = CBEventFlags.create(flags)

        # Big ugly if-else to handle all possible events
        if event == cv2.EVENT_MOUSEMOVE:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.hovered = cbxy.is_in_region
                if cbitem._cb_state.left_pressed:
                    cbitem.on_drag(cbxy, cbflags)
                cbitem.on_move(cbxy, cbflags)

        elif event == cv2.EVENT_LBUTTONDOWN:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.left_pressed = cbxy.is_in_region
                cbitem.on_left_down(cbxy, cbflags)

        elif event == cv2.EVENT_RBUTTONDOWN:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.right_pressed = cbxy.is_in_region
                cbitem.on_right_down(cbxy, cbflags)

        elif event == cv2.EVENT_MBUTTONDOWN:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.middle_pressed = cbxy.is_in_region
                cbitem.on_middle_down(cbxy, cbflags)

        elif event == cv2.EVENT_LBUTTONUP:
            for cbitem, cbxy in self._cb_iter(x, y):
                if cbitem._cb_state.left_pressed:
                    cbitem._cb_state.left_pressed = False
                    if cbxy.is_in_region:
                        cbitem.on_left_click(cbxy, cbflags)
                cbitem.on_left_up(cbxy, cbflags)

        elif event == cv2.EVENT_RBUTTONUP:
            for cbitem, cbxy in self._cb_iter(x, y):
                if cbitem._cb_state.right_pressed:
                    cbitem._cb_state.right_pressed = False
                    if cbxy.is_in_region:
                        cbitem.on_right_click(cbxy, cbflags)
                cbitem.on_right_up(cbxy, cbflags)

        elif event == cv2.EVENT_MBUTTONUP:
            for cbitem, cbxy in self._cb_iter(x, y):
                if cbitem._cb_state.middle_pressed:
                    cbitem._cb_state.middle_pressed = False
                    if cbxy.is_in_region:
                        cbitem.on_middle_click(cbxy, cbflags)
                cbitem.on_middle_up(cbxy, cbflags)

        if event == cv2.EVENT_LBUTTONDBLCLK:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.left_pressed = False
                cbitem.on_left_double(cbxy, cbflags)

        elif event == cv2.EVENT_RBUTTONDBLCLK:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.right_pressed = False
                cbitem.on_right_double(cbxy, cbflags)

        elif event == cv2.EVENT_MBUTTONDBLCLK:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.middle_pressed = False
                cbitem.on_middle_double(cbxy, cbflags)

        elif event == cv2.EVENT_MOUSEWHEEL:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem.on_mouse_wheel(cbxy, cbflags)

        return

    def _cb_iter(self, global_x_px, global_y_px):
        """Helper used to run callbacks on all self + children"""

        # Return our own event data
        cbxy = self._cb_region.make_cbeventxy(global_x_px, global_y_px)
        if not self._cb_state.disabled:
            yield self, cbxy

        # Recursively call iterator on all children and children-of-children etc. to call all nested callbacks
        for child in self._cb_child_list:
            if not child._cb_state.disabled:
                yield from child._cb_iter(global_x_px, global_y_px)
            # child_cbxy = child._cb_region.make_cbeventxy(global_x_px, global_y_px)
            # yield child, child_cbxy

        return

    def render(self, h=None, w=None):
        render_size = self._update_render_sizing(h, w)
        frame = self._render_up_to_size(render_size.h, render_size.w)

        # Sanity check that render target did what it was told to do...
        if len(self._cb_parent_list) > 0:
            assert render_size.is_match(
                frame.shape
            ), f"Bad render size: {frame.shape} vs {render_size} ({self._debug_name})"

        return frame

    def rerender(self):
        """Helper used to re-render using last render size"""
        render_h = self._rdr.size.h
        return self.render(h=render_h)

    def _render_up_to_size(self, h, w):
        class_name = self.__class__.__name__
        raise NotImplementedError(f"Must implement '_render_up_to_size' function ({class_name})")

    def _update_render_sizing(self, h=None, w=None) -> RenderTargetSize:

        if h is not None and w is not None:
            # If both sizes are given, treat them as 'max height & width' settings
            # -> Compute outcomes for both cases independently, and pick the one that ends up smaller overall
            alt_w = self._get_width_given_height(h)
            alt_h = self._get_height_given_width(w)
            fewer_pixels_with_alt_w = (alt_w * h) < (alt_h * w)
            h, w = (h, alt_w) if fewer_pixels_with_alt_w else (alt_h, w)

        elif h is None and w is None:
            h, w = self._get_height_and_width_without_hint()
        elif h is None:
            h = self._get_height_given_width(w)
        elif w is None:
            w = self._get_width_given_height(h)

        return self._rdr.set_render_size(h, w)

    def _get_width_given_height(self, h: int) -> int:
        """Function used to communicate how wide an element will be, if asked to render to a given height"""
        return self._rdr.limits.min_w

    def _get_height_given_width(self, w: int) -> int:
        """Function used to communicate how tall an element will be, if asked to render to a given width"""
        return self._rdr.limits.min_h

    def _get_height_and_width_without_hint(self) -> [int, int]:
        """Function used to communicate how wide & tall an element will be, if no size is specified"""
        return self._rdr.limits.min_h, self._rdr.limits.min_w


class BaseImageCallback(BaseCallback):

    # .................................................................................................................

    def __init__(self, image, expand_h=False, expand_w=False):

        # Store image for re-use when rendering
        image_3ch = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self._full_image = image_3ch
        self._render_image = image_3ch
        self._targ_h = -1
        self._targ_w = -1

        # Set up sizing limits
        img_hw = image.shape[0:2]
        min_h, min_w = (max(1, val // 4) for val in img_hw)
        max_h, max_w = (max(1, round(val * 2)) for val in img_hw)
        super().__init__(min_h, min_w, expand_h=expand_h, expand_w=expand_w, max_h=max_h, max_w=max_w)

    # .................................................................................................................

    def __repr__(self):
        img_h, img_w, img_ch = self._full_image.shape
        return f"Image ({img_h}x{img_w}x{img_ch})"

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        # Re-use rendered image if possible, otherwise re-render to target size
        h, w = min(h, self._rdr.limits.max_h), min(w, self._rdr.limits.max_w)
        if self._targ_h != h or self._targ_w != w:
            img_h, img_w = self._full_image.shape[0:2]
            scale = min(h / img_h, w / img_w)
            fill_wh = (round(scale * img_w), round(scale * img_h))
            self._render_image = cv2.resize(self._full_image, dsize=fill_wh)
            self._targ_h = h
            self._targ_w = w

        return self._render_image

    # .................................................................................................................

    def _get_width_given_height(self, h):
        h = min(h, self._rdr.limits.max_h)
        img_h, img_w = self._full_image.shape[0:2]
        scaled_w = round(img_w * h / img_h)
        return scaled_w

    def _get_height_given_width(self, w):
        w = min(w, self._rdr.limits.max_w)
        img_h, img_w = self._full_image.shape[0:2]
        scaled_h = round(img_h * w / img_w)
        return scaled_h

    def _get_height_and_width_without_hint(self):
        img_h, img_w = self._full_image.shape[0:2]
        return img_h, img_w

    # .................................................................................................................


class BaseOverlay(BaseCallback):

    # .................................................................................................................

    def __init__(self):
        super().__init__(1, 1)

    def _render_overlay(self, frame):
        return frame

    # .................................................................................................................


# %% Functions


def force_same_min_width(*items, min_w=None) -> int:
    """
    Helper used to force all items to the same minimum width.
    If a target width isn't given, then the largest minimum width
    of the given items will be used.
    Returns the min width setting
    """

    if min_w is None:
        min_w = max(item._rdr.limits.min_w for item in items)

    for item in items:
        item._rdr.limits.update(min_w=min_w)

    return min_w


def force_same_max_width(*items, max_w=None) -> int:
    """
    Helper used to force all items to the same maximum width.
    If a width isn't given, then the largest maximum width
    of the given items will be used.
    Returns the max width setting
    """

    if max_w is None:
        max_w = max(item._rdr.limits.max_w for item in items)

    for item in items:
        item._rdr.limits.update(max_w=max_w)

    return max_w


def force_same_min_height(*items, min_h=None) -> int:
    """
    Helper used to force all items to the same minimum height.
    If a height isn't given, then the largest minimum height
    of the given items will be used
    Returns the min height setting
    """

    if min_h is None:
        min_h = max(item._rdr.limits.min_h for item in items)

    for item in items:
        item._rdr.limits.update(min_h=min_h)

    return min_h


def force_same_max_height(*items, max_h=None) -> int:
    """
    Helper used to force all items to the same maximum height.
    If a height isn't given, then the largest maximum height
    of the given items will be used
    Returns the max height setting
    """

    if max_h is None:
        max_h = max(item._rdr.limits.max_h for item in items)

    for item in items:
        item._rdr.limits.update(max_h=max_h)

    return max_h
