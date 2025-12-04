#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .base import BaseOverlay, CBEventXY
from .helpers.images import draw_normalized_polygons
from .helpers.text import TextDrawer

# Typing
from numpy import ndarray

# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class DrawPolygonsOverlay(BaseOverlay):
    """Simple overlay which draws polygons over top of base images"""

    # .................................................................................................................

    def __init__(
        self,
        color=(0, 255, 255),
        bg_color=None,
        thickness=2,
        line_type=cv2.LINE_AA,
        is_closed=True,
    ):

        self._poly_xy_norm_list = []
        self._fg_color = color
        self._bg_color = bg_color
        self._fg_thick = thickness
        self._ltype = line_type
        self._is_closed = is_closed

        super().__init__()

    # .................................................................................................................

    def style(self, color=None, bg_color=None, thickness=None):
        """Update polygon styling. Any settings given as None will remain unchanged"""

        if color is not None:
            self._fg_color = color
        if thickness is not None:
            self._fg_thick = thickness
        if bg_color is not None:
            self._bg_color = bg_color if bg_color != -1 else None

        return self

    # .................................................................................................................

    def clear(self):
        self._poly_xy_norm_list = []
        return self

    # .................................................................................................................

    def set_polygons(self, polygon_xy_norm_list: list[ndarray] | ndarray):
        """
        Set or update polygons. Polygons should be provided as a list of numpy arrays,
        where each array hold polygon points in normalized xy format.
        For example:
            polygon = np.float32([(0.25,0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
            set_polygons([polygon])

        If only a single polygon is provided, then the outer list will be added automatically
        """

        if isinstance(polygon_xy_norm_list, ndarray):
            polygon_xy_norm_list = [polygon_xy_norm_list]
        self._poly_xy_norm_list = polygon_xy_norm_list

        return self

    # .................................................................................................................

    def _render_overlay(self, frame):

        if self._poly_xy_norm_list is None:
            return frame

        return draw_normalized_polygons(
            frame, self._poly_xy_norm_list, self._fg_color, self._fg_thick, self._bg_color, self._ltype, self._is_closed
        )

    # .................................................................................................................


class TextOverlay(BaseOverlay):

    # .................................................................................................................

    def __init__(
        self,
        xy_norm=(0.5, 0.5),
        scale=0.5,
        thickness=1,
        color=(255, 255, 255),
        bg_color=(0, 0, 0),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        line_type=cv2.LINE_AA,
        anchor_xy_norm=None,
        offset_xy_px=(0, 0),
    ):
        super().__init__()
        self._text = None
        self._xy_norm = xy_norm
        self._anchor_xy_norm = anchor_xy_norm
        self._offset_xy_px = offset_xy_px
        self._txtdraw = TextDrawer(scale, thickness, color, bg_color, font, line_type)

    # .................................................................................................................

    def style(self, scale=None, thickness=None, color=None, bg_color=None):
        """
        Update text styling. Any settings given as None will remain unchanged
        To clear an existing background color (which normally requires setting it to None), use -1
        """

        self._txtdraw.style(scale, thickness, color, bg_color)

        return self

    # .................................................................................................................

    def set_text(self, text: str | None, xy_norm: None, anchor_xy_norm: None, offset_xy_px: None):

        self._text = text

        if xy_norm is not None:
            self._xy_norm = xy_norm
        if anchor_xy_norm is not None:
            self._anchor_xy_norm = anchor_xy_norm
        if offset_xy_px is not None:
            self._offset_xy_px = offset_xy_px

        return self

    # .................................................................................................................

    def _render_overlay(self, frame):

        if self._text is None:
            return frame

        return self._txtdraw.xy_norm(frame, self._text, self._xy_norm, self._anchor_xy_norm, self._offset_xy_px)

    # .................................................................................................................


class HoverOverlay(BaseOverlay):

    # .................................................................................................................

    def __init__(self):
        super().__init__()
        self._event_xy = CBEventXY((0, 0), (0, 0), (0, 0), (1, 1), False)
        self._is_valid = False
        self._is_clicked = False
        self._is_changed = False
        self._prev_is_in_region = None

    # .................................................................................................................

    def clear(self, flag_is_changed=True):
        """Reset overlay state"""
        self._event_xy = CBEventXY((0, 0), (0, 0), (0, 0), (1, 1), False)
        self._is_valid = False
        self._is_clicked = False
        self._is_changed = flag_is_changed
        self._prev_is_in_region = None
        return self

    # .................................................................................................................

    def read(self) -> tuple[bool, bool, CBEventXY]:
        """Returns:  is_changed, is_click, event_xy"""
        is_changed = self._is_changed
        self._is_changed = False
        is_clicked = self._is_clicked
        self._is_clicked = False
        return is_changed, is_clicked, self._event_xy

    # .................................................................................................................

    def on_move(self, cbxy, cbflags) -> None:
        is_in_region = cbxy.is_in_region
        is_in_region_changed = is_in_region != self._prev_is_in_region
        self._is_changed |= is_in_region or is_in_region_changed
        self._prev_is_in_region = is_in_region
        self._event_xy = cbxy
        return

    def on_left_click(self, cbxy, cbflags) -> None:
        self._is_changed = True
        self._is_clicked = True
        self._event_xy = cbxy
        return

    # .................................................................................................................

    def _render_overlay(self, frame):
        return frame

    # .................................................................................................................


class PointSelectOverlay(BaseOverlay):

    # .................................................................................................................

    def __init__(self, color=(0, 255, 255), point_radius=4, bg_color=(0, 0, 0), thickness=-1, line_type=cv2.LINE_AA):
        # Inherit from parent
        super().__init__()

        # Store point state
        self._xy_norm_list: list[tuple[float, float]] = []
        self._is_changed = False

        # Store display config
        self._fg_color = color
        self._bg_color = bg_color
        self._fg_radius = point_radius
        self._bg_radius = point_radius if thickness > 0 else point_radius + 1
        self._fg_thick = thickness
        self._bg_thick = max(1 + thickness, 2 * thickness) if thickness > 0 else thickness
        self._ltype = line_type

    # .................................................................................................................

    def style(self, color=None, radius=None, thickness=None, bg_color=None, bg_radius=None, bg_thickness=None):
        """Update point styling. Any settings given as None will remain unchanged"""

        if color is not None:
            self._fg_color = color
        if radius is not None:
            self._fg_radius = radius
        if thickness is not None:
            self._fg_t = thickness
        if bg_color is not None:
            self._bg_color = bg_color if bg_color != -1 else None
        if bg_radius is not None:
            self._bg_radius = bg_radius
        if bg_thickness is not None:
            self._bg_thick = bg_thickness

        return self

    # .................................................................................................................

    def clear(self, flag_is_changed=True):
        self._is_changed = (len(self._xy_norm_list) > 0) and flag_is_changed
        self._xy_norm_list = []
        return self

    # .................................................................................................................

    def read(self) -> tuple[bool, tuple]:
        """Returns: is_changed, xy_norm_list"""
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, tuple(self._xy_norm_list)

    # .................................................................................................................

    def on_left_click(self, cbxy, cbflags):

        # Add point if shift clicked or update point otherwise
        new_xy_norm = cbxy.xy_norm
        if cbflags.shift_key:
            self.add_points(new_xy_norm)
        else:
            if len(self._xy_norm_list) == 0:
                self._xy_norm_list = [new_xy_norm]
            else:
                self._xy_norm_list[-1] = new_xy_norm

        self._is_changed = True

        return

    def on_right_click(self, cbxy, cbflags):

        self.remove_closest(cbxy.xy_norm, cbxy.hw_px)

        return

    # .................................................................................................................

    def _render_overlay(self, frame):

        # Convert points to pixel coords for drawing
        frame_h, frame_w = frame.shape[0:2]
        norm_to_px_scale = np.float32((frame_w - 1, frame_h - 1))
        xy_px_list = [np.int32(xy_norm * norm_to_px_scale) for xy_norm in self._xy_norm_list]

        # Draw each point as a circle with a background if needed
        if self._bg_color is not None:
            for xy_px in xy_px_list:
                cv2.circle(frame, xy_px, self._bg_radius, self._bg_color, self._bg_thick, self._ltype)
        for xy_px in xy_px_list:
            cv2.circle(frame, xy_px, self._fg_radius, self._fg_color, self._fg_thick, self._ltype)

        return frame

    # .................................................................................................................

    def add_points(self, *xy_norm_points):

        if len(xy_norm_points) == 0:
            return self

        self._xy_norm_list.extend(xy_norm_points)
        self._is_changed = True

        return self

    # .................................................................................................................

    def remove_closest(self, xy_norm, frame_hw=None) -> None | tuple[float, float]:

        # Can't remove points if there aren't any!
        if len(self._xy_norm_list) == 0:
            return None

        # Default to 'fake' pixel count if not given (so we can re-use the same calculations)
        if frame_hw is None:
            frame_hw = (10, 10)
        frame_h, frame_w = frame_hw
        norm_to_px_scale = np.float32((frame_w - 1, frame_h - 1))

        # Find the point closest to the given (x,y) for removal
        xy_px_array = np.int32([np.int32(xy_norm * norm_to_px_scale) for xy_norm in self._xy_norm_list])
        input_array = np.int32(xy_norm * norm_to_px_scale)
        dist_to_pts = np.linalg.norm(xy_px_array - input_array, ord=2, axis=1)
        closest_pt_idx = np.argmin(dist_to_pts)

        # Remove the point closest to the click and finish
        closest_xy_norm = self._xy_norm_list.pop(closest_pt_idx)
        self._is_changed = True

        return closest_xy_norm

    # .................................................................................................................


class BoxSelectOverlay(BaseOverlay):

    # .................................................................................................................

    def __init__(self, color=(0, 255, 255), thickness=1, bg_color=(0, 0, 0)):
        super().__init__()
        self._tlbr_norm_list: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._tlbr_norm_inprog = None
        self._is_changed = False

        # Store display config
        self._fg_color = color
        self._bg_color = bg_color
        self._fg_thick = thickness
        self._bg_thick = thickness + 1
        self._ltype = cv2.LINE_4

    # .................................................................................................................

    def style(self, color=None, thickness=None, bg_color=None, bg_thickness=None):
        """Update box styling. Any settings given as None will remain unchanged"""

        if color is not None:
            self._fg_color = color
        if thickness is not None:
            self._fg_thick = thickness
        if bg_color is not None:
            self._bg_color = bg_color if bg_color != -1 else None
        if bg_thickness is not None:
            self._bg_thick = bg_thickness

        return self

    # .................................................................................................................

    def clear(self, flag_is_changed=True):
        had_boxes = (len(self._tlbr_norm_list) > 0) or (self._tlbr_norm_inprog is not None)
        self._is_changed = had_boxes and flag_is_changed
        self._tlbr_norm_list = []
        self._tlbr_norm_inprog = None
        return self

    # .................................................................................................................

    def read(self, include_in_progress_box=True) -> tuple[bool, tuple]:
        """Returns: is_changed, box_tlbr_list"""

        # Toggle change state, if needed
        is_changed = self._is_changed
        self._is_changed = False

        # Get list of boxes including in-progress box if needed
        out_list = self._tlbr_norm_list
        if include_in_progress_box:
            is_valid, extra_tlbr = self._make_inprog_tlbr()
            extra_tlbr_list = [extra_tlbr] if is_valid else []
            out_list = self._tlbr_norm_list + extra_tlbr_list

        return is_changed, tuple(out_list)

    # .................................................................................................................

    def on_left_down(self, cbxy, cbflags):

        # Ignore clicks outside of region
        if not cbxy.is_in_region:
            return

        # Begin new 'in-progress' box
        self._tlbr_norm_inprog = [cbxy.xy_norm, cbxy.xy_norm]

        # Remove newest box if we're not shift-clicking
        if not cbflags.shift_key:
            if len(self._tlbr_norm_list) > 0:
                self._tlbr_norm_list.pop()

        self._is_changed = True

        return

    def on_drag(self, cbxy, cbflags):

        # Update second in-progress box point
        if self._tlbr_norm_inprog is not None:
            new_xy = np.clip(cbxy.xy_norm, 0.0, 1.0)
            self._tlbr_norm_inprog[1] = tuple(new_xy)
            self._is_changed = True

        return

    def on_left_up(self, cbxy, cbflags):

        is_valid, new_tlbr = self._make_inprog_tlbr()
        if is_valid:
            self._tlbr_norm_list.append(new_tlbr)
            self._is_changed = True
        self._tlbr_norm_inprog = None

        return

    def on_right_click(self, cbxy, cbflags):
        self.remove_closest(cbxy.xy_norm, cbxy.hw_px)
        return

    # .................................................................................................................

    def _render_overlay(self, frame):

        # Check if we need to draw an in-progress box
        is_valid, new_tlbr = self._make_inprog_tlbr()
        extra_tlbr = [new_tlbr] if is_valid else []
        boxes_to_draw = self._tlbr_norm_list + extra_tlbr

        frame_h, frame_w = frame.shape[0:2]
        norm_to_px_scale = np.float32((frame_w - 1, frame_h - 1))
        box_px_list = []
        for box in boxes_to_draw:
            box = np.int32([xy_norm * norm_to_px_scale for xy_norm in box])
            box_px_list.append(box)

        if self._bg_color is not None:
            for xy1_px, xy2_px in box_px_list:
                cv2.rectangle(frame, xy1_px, xy2_px, self._bg_color, self._bg_thick, self._ltype)
        for xy1_px, xy2_px in box_px_list:
            cv2.rectangle(frame, xy1_px, xy2_px, self._fg_color, self._fg_thick, self._ltype)

        return frame

    # .................................................................................................................

    def add_boxes(self, *tlbr_norm_list):

        if len(tlbr_norm_list) == 0:
            return self

        self._tlbr_norm_list.extend(tlbr_norm_list)
        self._is_changed = True

        return self

    # .................................................................................................................

    def remove_closest(self, xy_norm, frame_hw=None) -> None:

        # Can't remove boxes if there aren't any!
        if len(self._tlbr_norm_list) == 0:
            return None

        # Default to 'fake' pixel count if not given (so we can re-use the same calculations)
        if frame_hw is None:
            frame_hw = (10, 10)
        frame_h, frame_w = frame_hw
        norm_to_px_scale = np.float32((frame_w - 1, frame_h - 1))

        # For each box, find the distance to the closest corner
        input_array = np.int32(xy_norm * norm_to_px_scale)
        closest_dist_list = []
        for (x1, y1), (x2, y2) in self._tlbr_norm_list:
            xy_px_array = np.float32([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]) * norm_to_px_scale
            dist_to_pts = np.linalg.norm(xy_px_array - input_array, ord=2, axis=1)
            closest_dist_list.append(min(dist_to_pts))

        # Among all boxes, remove the one with the closest corner to the given click
        closest_pt_idx = np.argmin(closest_dist_list)
        closest_tlbr_norm = self._tlbr_norm_list.pop(closest_pt_idx)
        self._is_changed = True

        return closest_tlbr_norm

    # .................................................................................................................

    def _make_inprog_tlbr(self):
        """
        Helper used to make a 'final' box out of in-progress data
        Includes re-arranging points to be in proper top-left/bottom-right order
        as well as discarding boxes that are 'too small'
        """

        new_tlbr = None
        is_valid = self._tlbr_norm_inprog is not None
        if is_valid:

            # Re-arrange points to make sure first xy is top-left, second is bottom-right
            xy1_xy2 = np.float32(self._tlbr_norm_inprog)
            tl_xy_norm = xy1_xy2.min(0)
            br_xy_norm = xy1_xy2.max(0)

            # Make sure the box is not infinitesimally small
            xy_diff = br_xy_norm - tl_xy_norm
            is_valid = np.all(xy_diff > 1e-4)
            if is_valid:
                new_tlbr = (tl_xy_norm.tolist(), br_xy_norm.tolist())

        return is_valid, new_tlbr

    # .................................................................................................................


class EditBoxOverlay(BaseOverlay):
    """
    Overlay used to provide a 'crop-box' or similar UI
    The idea being to have a single box that can be modified
    by clicking and dragging the corners or sides, or otherwise
    fully re-drawn by clicking far enough away from the box.
    It is always assumed that there is 1 box!

    This differs from the regular 'box select overlay' which
    re-draws boxes on every click and supports multiple boxes
    """

    # .................................................................................................................

    def __init__(
        self,
        frame_shape=None,
        color=(0, 255, 255),
        thickness=1,
        bg_color=(0, 0, 0),
        indicator_base_radius=6,
        interaction_distance_px=100,
        minimum_box_area_norm=5e-5,
    ):
        # Inherit from parent
        super().__init__()

        # Store box points in format that supports 'mid points'
        self._x_norms = np.float32([0.0, 0.5, 1.0])
        self._y_norms = np.float32([0.0, 0.5, 1.0])
        self._prev_xy_norms = (self._x_norms, self._y_norms)
        self._is_changed = False

        # Store indexing used to specify which of the box points is being modified, if any
        self._is_modifying = False
        self._xy_modify_idx = (2, 2)
        self._mouse_xy_norm = (0.0, 0.0)

        # Store sizing of frame being cropped, only use when 'nudging' the crop box
        self._full_frame_hw = frame_shape[0:2] if frame_shape is not None else (100, 100)

        # Store thresholding settings
        self._minimum_area_norm = minimum_box_area_norm
        self._interact_dist_px_threshold = interaction_distance_px

        # Store display config
        self._fg_color = color
        self._bg_color = bg_color
        self._fg_thick = thickness
        self._bg_thick = thickness + 1
        self._ltype = cv2.LINE_4
        self._ind_base_radius = indicator_base_radius
        self._ind_fg_radius = self._ind_base_radius + self._fg_thick
        self._ind_bg_radius = self._ind_fg_radius + (self._bg_thick - self._fg_thick)
        self._ind_ltype = cv2.LINE_AA

    # .................................................................................................................

    def style(self, color=None, thickness=None, bg_color=None, bg_thickness=None):
        """Update box styling. Any settings given as None will remain unchanged"""

        if color is not None:
            self._fg_color = color
        if thickness is not None:
            self._fg_thick = thickness
            self._ind_fg_radius = self._ind_base_radius + self._fg_thick
        if bg_color is not None:
            self._bg_color = bg_color if bg_color != -1 else None
        if bg_thickness is not None:
            self._bg_thick = bg_thickness
            self._ind_bg_radius = self._ind_fg_radius + (self._bg_thick - self._fg_thick)

        return self

    # .................................................................................................................

    def clear(self):
        """Reset box back to entire frame size"""
        self._x_norms = np.float32([0.0, 0.5, 1.0])
        self._y_norms = np.float32([0.0, 0.5, 1.0])
        self._is_changed = True
        self._is_modifying = False
        return self

    # .................................................................................................................

    def read(self) -> tuple[bool, bool, tuple[tuple[float, float], tuple[float, float]]]:
        """
        Read current box state
        Returns:
            is_changed, is_valid, box_tlbr_norm
            -> 'is_box_valid' is based on the minimum box area setting
            -> box_tlbr_norm is in format: ((x1, y1), (x2, y2))
        """

        # Toggle change state, if needed
        is_changed = self._is_changed
        self._is_changed = False

        # Get top-left/bottom-right output if it exists
        x1, _, x2 = sorted(self._x_norms.tolist())
        y1, _, y2 = sorted(self._y_norms.tolist())
        box_tlbr_norm = ((x1, y1), (x2, y2))
        is_valid = ((x2 - x1) * abs(y2 - y1)) > self._minimum_area_norm

        return is_changed, is_valid, box_tlbr_norm

    # .................................................................................................................

    def set_box(self, tlbr_norm: tuple[tuple[float, float], tuple[float, float]]):
        """
        Update box coordinates. Input is expected in top-left/bottom-right format:
            ((x1, y1), (x2, y2))
        """

        (x1, y1), (x2, y2) = tlbr_norm
        x_mid, y_mid = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        self._x_norms = np.float32((x1, x_mid, x2))
        self._y_norms = np.float32((y1, y_mid, y2))
        self._is_changed = True
        self._is_modifying = False

        return self

    # .................................................................................................................

    def on_move(self, cbxy, cbflags):

        # Record mouse position for rendering 'closest point' indicator on hover
        self._mouse_xy_norm = cbxy.xy_norm

        return

    def on_left_down(self, cbxy, cbflags):
        """Create a new box or modify exist box based on left-click position"""

        # Ignore clicks outside of region
        if not cbxy.is_in_region:
            return

        # Record 'previous' box, in case we need to reset (happens if user draws invalid box)
        self._prev_xy_norms = (self._x_norms, self._y_norms)

        # Figure out if we're 'modifying' the box or drawing a new one
        xy_idx, _, is_interactive_dist = self._check_xy_interaction(cbxy.xy_norm, cbxy.hw_px)
        is_new_click = not is_interactive_dist or cbflags.shift_key

        # Either modify an existing point or reset/re-draw the box if clicking away from existing points
        self._xy_modify_idx = xy_idx
        if is_new_click:
            # We modify the 'last' xy coord on new boxes, by convention
            self._xy_modify_idx = (2, 2)
            new_x, new_y = cbxy.xy_norm
            self._x_norms = np.float32((new_x, new_x, new_x))
            self._y_norms = np.float32((new_y, new_y, new_y))

        self._is_modifying = True
        self._is_changed = True

        return

    def on_drag(self, cbxy, cbflags):
        """Modify box corner or midpoint when dragging"""

        # Bail if no points are being modified (shouldn't happen...?)
        if not self._is_modifying:
            return

        # Don't allow dragging out-of-bounds!
        new_x, new_y = np.clip(cbxy.xy_norm, 0.0, 1.0)

        # Update corner points (if they're the ones being modified) and re-compute mid-points
        x_mod_idx, y_mod_idx = self._xy_modify_idx
        if x_mod_idx != 1:
            self._x_norms[x_mod_idx] = new_x
            self._x_norms[1] = (self._x_norms[0] + self._x_norms[2]) * 0.5
        if y_mod_idx != 1:
            self._y_norms[y_mod_idx] = new_y
            self._y_norms[1] = (self._y_norms[0] + self._y_norms[2]) * 0.5

        # Assume box is changed by dragging update
        self._is_changed = True

        return

    def on_left_up(self, cbxy, cbflags):
        """Stop modifying box on left up"""

        # Reset modifier indexing
        self._is_modifying = False

        # Reset if the resulting box is too small
        h_px, w_px = cbxy.hw_px
        box_w = int(np.abs(self._x_norms[0] - self._x_norms[2]) * (h_px - 1))
        box_h = int(np.abs(self._y_norms[0] - self._y_norms[1]) * (w_px - 1))
        box_area_norm = (box_h * box_w) / (h_px * w_px)
        if box_area_norm < self._minimum_area_norm:
            self._x_norms, self._y_norms = self._prev_xy_norms
            self._is_changed = True

        return

    def on_right_click(self, cbxy, cbflags):
        self.clear()
        return

    # .................................................................................................................

    def _check_xy_interaction(self, target_xy_norm, frame_hw=None) -> tuple[tuple[int, int], tuple[float, float], bool]:
        """
        Helper used to check which of the box points (corners or midpoints)
        are closest to given target xy coordinate, and what the x/y distance
        ('manhattan distance') is to the closest point. Used to determine
        which points may be interacted with for dragging/modifying the box.

        Returns:
            closest_xy_index, closest_xy_distance_px, is_interactive_distance
            -> Indexing is with respect to self._x_norms & self._y_norms
        """

        # Default to 'fake' pixel count if not given (so we can re-use the same calculations)
        if frame_hw is None:
            frame_hw = (2.0, 2.0)
        h_scale, w_scale = tuple(np.float32(size - 1.0) for size in frame_hw)
        target_x, target_y = target_xy_norm

        # Find closest x point on box
        x_dists = np.abs(self._x_norms - target_x)
        closest_x_index = np.argmin(x_dists)
        closest_x_dist_px = x_dists[closest_x_index] * w_scale

        # Find closest y point on box
        y_dists = np.abs(self._y_norms - target_y)
        closest_y_index = np.argmin(y_dists)
        closest_y_dist_px = y_dists[closest_y_index] * h_scale

        # Check if the point is within interaction distance
        closest_xy_index = (closest_x_index, closest_y_index)
        closest_xy_dist_px = (closest_x_dist_px, closest_y_dist_px)
        is_interactive = all(dist < self._interact_dist_px_threshold for dist in closest_xy_dist_px)
        if is_interactive:
            is_center_point = all(idx == 1 for idx in closest_xy_index)
            is_interactive = not is_center_point

        return closest_xy_index, closest_xy_dist_px, is_interactive

    # .................................................................................................................

    def _render_overlay(self, frame):

        # Get sizing info
        frame_hw = frame.shape[0:2]
        h_scale, w_scale = tuple(float(size - 1.0) for size in frame_hw)
        all_x_px = tuple(int(x * w_scale) for x in self._x_norms)
        all_y_px = tuple(int(y * h_scale) for y in self._y_norms)
        xy1_px, xy2_px = (all_x_px[0], all_y_px[0]), (all_x_px[-1], all_y_px[-1])

        # Figure out whether we should draw interaction indicator & where
        need_draw_indicator = True
        if self._is_modifying:
            # If user if modifying the box, choose the modified point for drawing
            # -> We want to always draw the indicator for the point being dragged, even if
            #    the mouse is closer to some other point (can happen when dragging mid points)
            close_x_px = all_x_px[self._xy_modify_idx[0]]
            close_y_px = all_y_px[self._xy_modify_idx[1]]

        else:
            # If user isn't already interacting, we'll draw an indicator if the mouse is
            # close enough to a corner or mid point on the box. But we have to figure
            # out which point that would be every time we re-render, in case the mouse moved!
            (x_idx, y_idx), _, is_interactive_dist = self._check_xy_interaction(self._mouse_xy_norm, frame_hw)
            close_x_px = all_x_px[x_idx]
            close_y_px = all_y_px[y_idx]
            is_inbounds = np.min(self._mouse_xy_norm) > 0.0 and np.max(self._mouse_xy_norm) < 1.0
            need_draw_indicator = is_interactive_dist and is_inbounds
        closest_xy_px = (close_x_px, close_y_px)

        # Draw all background coloring first, so it appears entirely 'behind' the foreground
        if self._bg_color is not None:
            if need_draw_indicator:
                cv2.circle(frame, closest_xy_px, self._ind_bg_radius, self._bg_color, -1, self._ind_ltype)
            cv2.rectangle(frame, xy1_px, xy2_px, self._bg_color, self._bg_thick, self._ltype)

        # Draw box + interaction indicator circle in foreground color
        if need_draw_indicator:
            cv2.circle(frame, closest_xy_px, self._ind_fg_radius, self._fg_color, -1, self._ind_ltype)
        cv2.rectangle(frame, xy1_px, xy2_px, self._fg_color, self._fg_thick, self._ltype)

        return frame

    # .................................................................................................................

    def nudge(self, left=0, right=0, up=0, down=0):
        """Helper used to move the position of a point (nearest to the mouse) by some number of pixels"""

        # Figure out which point to nudge
        (x_idx, y_idx), _, _ = self._check_xy_interaction(self._mouse_xy_norm, self._full_frame_hw)

        # Handle left/right nudge
        is_leftright_nudgable = x_idx != 1
        leftright_nudge = right - left
        if is_leftright_nudgable and leftright_nudge != 0:
            _, w_px = self._full_frame_hw
            old_x_norm = self._x_norms[x_idx]
            old_x_px = old_x_norm * (w_px - 1)
            new_x_px = old_x_px + leftright_nudge
            new_x_norm = new_x_px / (w_px - 1)
            new_x_norm = np.clip(new_x_norm, 0.0, 1.0)

            # Update target x coord and re-compute midpoint
            self._x_norms[x_idx] = new_x_norm
            self._x_norms[1] = (self._x_norms[0] + self._x_norms[-1]) * 0.5

        # Handle up/down nudge
        is_updown_nudgable = y_idx != 1
        updown_nudge = down - up
        if is_updown_nudgable and updown_nudge != 0:
            h_px, _ = self._full_frame_hw
            old_y_norm = self._y_norms[y_idx]
            old_y_px = old_y_norm * (h_px - 1)
            new_y_px = old_y_px + updown_nudge
            new_y_norm = new_y_px / (h_px - 1)
            new_y_norm = np.clip(new_y_norm, 0.0, 1.0)

            # Update target x coord and re-compute midpoint
            self._y_norms[y_idx] = new_y_norm
            self._y_norms[1] = (self._y_norms[0] + self._y_norms[-1]) * 0.5

        # Assume we've changed the box
        self._is_changed = True

        return self

    # .................................................................................................................
