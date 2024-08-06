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

    # .................................................................................................................

    def clear(self):
        self._event_xy = CBEventXY((0, 0), (0, 0), (0, 0), (1, 1), False)
        self._is_valid = False
        self._is_clicked = False
        self._is_changed = False
        return self

    # .................................................................................................................

    def read(self):
        is_changed = self._is_changed
        self._is_changed = False
        is_clicked = self._is_clicked
        self._is_clicked = False
        return is_changed, is_clicked, self._event_xy

    # .................................................................................................................

    def on_move(self, cbxy, cbflags) -> None:
        self._is_changed = True
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

    def clear(self):
        self._xy_norm_list = []
        self._is_changed = False
        return self

    # .................................................................................................................

    def read(self):
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

    def clear(self):
        self._tlbr_norm_list = []
        self._tlbr_norm_inprog = None
        self._is_changed = False
        return self

    # .................................................................................................................

    def read(self, include_in_progress_box=True) -> tuple[bool, tuple]:

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
