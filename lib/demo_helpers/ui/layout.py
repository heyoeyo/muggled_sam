#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .base import BaseCallback


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class HStack(BaseCallback):

    # .................................................................................................................

    def __init__(self, *items, error_on_size_constraints=False):

        super().__init__(32, 256)
        self.append_children(*items)
        self._error_on_constraints = error_on_size_constraints

        # Update stack sizing based on children
        tallest_child_min_h = max(child._rdr.limits.min_h for child in self)
        total_child_min_w = sum(child._rdr.limits.min_w for child in self)
        can_expand_h = any(child._rdr.limits.expand_h for child in self)
        can_expand_w = any(child._rdr.limits.expand_w for child in self)
        self._rdr.limits.update(
            min_h=tallest_child_min_h, min_w=total_child_min_w, expand_h=can_expand_h, expand_w=can_expand_w
        )

    # .................................................................................................................

    def __repr__(self):
        child_names = [str(child._debug_name) for child in self]
        return f"{self._debug_name} [{', '.join(child_names)}]"

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        # Set up starting stack point, used to keep track of child callback regions
        x_stack = self._cb_region.x1
        y_stack = self._cb_region.y1

        # Scale child render widths to fit target width if needed
        child_render_w_list = [child._rdr.size.w for child in self]
        total_child_w = sum(child_render_w_list)
        if total_child_w != w:
            child_render_w_list = [int(w * ch_w / total_child_w) for ch_w in child_render_w_list]

        # Have each child item draw itself
        imgs_list = []
        for child, ch_render_w in zip(self, child_render_w_list):
            frame = child._render_up_to_size(h, ch_render_w)
            orig_frame_h, orig_frame_w = frame.shape[0:2]

            # Adjust frame height if needed
            tpad, bpad, lpad, rpad = 0, 0, 0, 0
            if orig_frame_h < h:
                available_h = h - orig_frame_h
                tpad = available_h // 2
                bpad = available_h - tpad
                lpad, rpad = 0, 0
                ptype = child._rdr.pad.style
                pcolor = child._rdr.pad.color
                frame = cv2.copyMakeBorder(frame, tpad, bpad, lpad, rpad, ptype, value=pcolor)

            elif orig_frame_h > h:
                print(
                    f"Render sizing error! Expecting height: {h}, got {orig_frame_h} ({child._debug_name})",
                    "-> Will crop!",
                    sep="\n",
                )
                frame = frame[:h, :, :]
                orig_frame_h = h

            # Store image
            imgs_list.append(frame)

            # Provide callback region to child item
            x1, y1 = x_stack + lpad, y_stack + tpad
            x2, y2 = x1 + orig_frame_w, y1 + orig_frame_h
            child._cb_region.update(x1, y1, x2, y2)

            # Update stacking point for next child
            x_stack = x2

        return np.hstack(imgs_list)

    # .................................................................................................................

    def _get_height_and_width_without_hint(self) -> [int, int]:
        """Set height to tallest child height and then calculate width from given height"""
        tallest_h = max(child._rdr.limits.min_h for child in self)
        w = self._get_width_given_height(tallest_h)
        return tallest_h, w

    # .................................................................................................................

    def _get_width_given_height(self, h) -> int:
        """Set width to sum of child widths, when told to render at the given height"""
        child_w_list = [child._update_render_sizing(h=h).w for child in self]
        total_w = sum(child_w_list)
        return total_w

    # .................................................................................................................

    def _get_height_given_width(self, w) -> int:
        """
        For h-stacking, we normally want to set a height since this must be shared
        for all elements in order to stack horizontally. Here we don't know the height,
        which complicates things!

        To figure out the height, we need to know how tall the tallest child will be,
        once all elements are stacked to meet the target width. However, the child elements
        will generally want to scale their heights according to how much space they're given!
        So we first need to figure out how much width each child will want to take, while
        hitting the target width amount
        """

        # Get child width listings along with whether they support expanding to fill space
        child_w_list = []
        expand_w_list = []
        for child in self:
            chlim = child._rdr.limits
            expand_w_list.append(chlim.expand_w)
            child_w_list.append(chlim.min_w)

        # Error out if needed
        total_child_w = sum(child_w_list)
        if total_child_w > w and self._error_on_constraints:
            raise ValueError(f"Cannot form H-Stack to target width ({w}), minimum child width is {total_child_w}!")

        # Figure out how much space we need to fill, and whether we have expandable elements to fill with
        available_w = max(w - total_child_w, 0)
        num_child = len(child_w_list)
        num_expandables = sum(expand_w_list)
        have_expandables = num_expandables > 0

        # Figure out the target width of each child element
        shareable_w = 0 if have_expandables else available_w
        expandable_w = available_w if have_expandables else 0
        target_w_list = []
        for ch_w, ch_expand_w in zip(child_w_list, expand_w_list):

            # Calculate shared width expansion
            # -> This is done in an awkward way to avoid rounding errors
            share_amt = shareable_w // num_child
            shareable_w -= share_amt
            num_child -= 1

            # For expandable elements, calculate expansion component of width
            # -> Again, we calculate this awkwardly in order to avoid rounding errors
            expand_amt = 0
            if ch_expand_w:
                expand_amt = expandable_w // num_expandables
                expandable_w -= expand_amt
                num_expandables -= 1
            target_w_list.append(ch_w + share_amt + expand_amt)

        # Sanity check
        total_computed_w = sum(target_w_list)
        if self._error_on_constraints:
            assert (
                total_computed_w == w
            ), f"Error computing target widths ({self._debug_name})! Target: {w}, got {total_computed_w}"

        # Finally, we'll say our height, given the target width, is that of the tallest child!
        child_h_list = [c._update_render_sizing(w=w).h for c, w in zip(self, target_w_list)]
        tallest_child_h = max(child_h_list)
        return tallest_child_h

    # .................................................................................................................


class VStack(BaseCallback):

    # .................................................................................................................

    def __init__(self, *items, error_on_size_constraints=False):

        super().__init__(256, 32)
        self.append_children(*items)
        self._error_on_constraints = error_on_size_constraints

        # Update stack sizing based on children
        total_child_min_h = sum(child._rdr.limits.min_h for child in self)
        widest_child_min_w = max(child._rdr.limits.min_w for child in self)
        can_expand_h = any(child._rdr.limits.expand_h for child in self)
        can_expand_w = any(child._rdr.limits.expand_w for child in self)
        self._rdr.limits.update(
            min_h=total_child_min_h, min_w=widest_child_min_w, expand_h=can_expand_h, expand_w=can_expand_w
        )

    # .................................................................................................................

    def __repr__(self):
        child_names = [str(child._debug_name) for child in self]
        return f"{self._debug_name} [{', '.join(child_names)}]"

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        # Set up starting stack point, used to keep track of child callback regions
        x_stack = self._cb_region.x1
        y_stack = self._cb_region.y1

        # Scale child render heights to fit target height if needed
        child_render_h_list = [child._rdr.size.h for child in self]
        total_child_h = sum(child_render_h_list)
        if total_child_h != h:
            child_render_h_list = [int(h * ch_h / total_child_h) for ch_h in child_render_h_list]

        # Have each child item draw itself
        imgs_list = []
        for child, ch_render_h in zip(self, child_render_h_list):
            frame = child._render_up_to_size(ch_render_h, w)
            orig_frame_h, orig_frame_w = frame.shape[0:2]

            # Adjust frame width if needed
            tpad, bpad, lpad, rpad = 0, 0, 0, 0
            if orig_frame_w < w:
                available_w = w - orig_frame_w
                lpad = available_w // 2
                rpad = available_w - lpad
                tpad, bpad = 0, 0
                ptype = child._rdr.pad.style
                pcolor = child._rdr.pad.color
                frame = cv2.copyMakeBorder(frame, tpad, bpad, lpad, rpad, ptype, value=pcolor)

            elif orig_frame_w > w:
                print(
                    f"Render sizing error! Expecting width: {w}, got {orig_frame_w} ({child._debug_name})",
                    "-> Will crop!",
                    sep="\n",
                )
                frame = frame[:, :w, :]
                orig_frame_w = w

            # Store image
            imgs_list.append(frame)

            # Provide callback region to child item
            x1, y1 = x_stack + lpad, y_stack + tpad
            x2, y2 = x1 + orig_frame_w, y1 + orig_frame_h
            child._cb_region.update(x1, y1, x2, y2)

            # Update stacking point for next child
            y_stack = y2

        return np.vstack(imgs_list)

    # .................................................................................................................

    def _get_height_and_width_without_hint(self) -> [int, int]:
        """When not given a size, render width to the widest child and sum of all child heights at this width"""
        widest_w = max(child._rdr.limits.min_w for child in self)
        h = self._get_height_given_width(widest_w)
        return h, widest_w

    # .................................................................................................................

    def _get_height_given_width(self, w) -> int:
        """When given a target width, the vstack height is the sum of all children rendered at that width"""
        child_h_list = [child._update_render_sizing(w=w).h for child in self]
        return sum(child_h_list)

    # .................................................................................................................

    def _get_width_given_height(self, h) -> int:
        """ """

        # Get child height listings along with whether they support expanding to fill space
        expand_h_list = []
        child_h_list = []
        for child in self:
            chlim = child._rdr.limits
            expand_h_list.append(chlim.expand_h)
            child_h_list.append(chlim.min_h)

        # Error out if needed
        total_child_h = sum(child_h_list)
        if total_child_h > h and self._error_on_constraints:
            raise ValueError(f"Cannot form V-Stack to target height ({h}), minimum child height is {total_child_h}!")

        # Figure out how much space we need to fill, and whether we have expandable elements to fill with
        available_h = max(h - total_child_h, 0)
        num_child = len(child_h_list)
        num_expandables = sum(expand_h_list)
        have_expandables = num_expandables > 0

        # Figure out the target height of each child element
        shareable_h = 0 if have_expandables else available_h
        expandable_h = available_h if have_expandables else 0
        target_h_list = []
        for ch_h, ch_expand_h in zip(child_h_list, expand_h_list):

            # Calculate shared height expansion
            # -> This is done in an awkward way to avoid rounding errors
            share_amt = shareable_h // num_child
            shareable_h -= share_amt
            num_child -= 1

            # For expandable elements, calculate expansion component of height
            # -> Again, we calculate this awkwardly in order to avoid rounding errors
            expand_amt = 0
            if ch_expand_h:
                expand_amt = expandable_h // num_expandables
                expandable_h -= expand_amt
                num_expandables -= 1
            target_h_list.append(ch_h + share_amt + expand_amt)

        # Sanity check
        total_computed_h = sum(target_h_list)
        if self._error_on_constraints:
            assert (
                total_computed_h == h
            ), f"Error computing target heights ({self._debug_name})! Target: {h}, got {total_computed_h}"

        # Finally, we'll say our width, given the target height, is that of the widest child!
        child_w_list = [c._update_render_sizing(h=h).w for c, h in zip(self, target_h_list)]
        widest_child_w = max(child_w_list)
        return widest_child_w

    # .................................................................................................................


class OverlayStack(BaseCallback):

    # .................................................................................................................

    def __init__(self, base_item, *overlay_items):

        # Inherit from parent and copy base item render limits
        super().__init__(32, 32)
        self._rdr.limits.match_to(base_item._rdr.limits)

        # Store base & overlays for future reference and include all items as children for callbacks
        self._base_item = base_item
        self._overlay_items = tuple(overlay_items)
        self.append_children(self._base_item, *overlay_items)

    # .................................................................................................................

    def __repr__(self):
        base_name = self._base_item._debug_name
        olay_names = [str(olay._debug_name) for olay in self._overlay_items]
        return f"{self._debug_name} [{base_name} | {', '.join(olay_names)}]"

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        # Set up starting stack point, used to keep track of child callback regions
        x1 = self._cb_region.x1
        y1 = self._cb_region.y1

        # Have base item provide the base frame rendering and overlays handle drawing over-top
        base_frame = self._base_item._render_up_to_size(h, w).copy()
        base_h, base_w = base_frame.shape[0:2]
        self._rdr.set_render_size(base_h, base_w)

        x2, y2 = x1 + base_w, y1 + base_h
        self._base_item._cb_region.update(x1, y1, x2, y2)
        for overlay in self._overlay_items:
            overlay._cb_region.update(x1, y1, x2, y2)
            base_frame = overlay._render_overlay(base_frame)

        return base_frame

    # .................................................................................................................

    def _get_height_and_width_without_hint(self) -> [int, int]:
        return self._base_item._get_height_and_width_without_hint()

    def _get_height_given_width(self, w) -> int:
        return self._base_item._get_height_given_width(w)

    def _get_width_given_height(self, h) -> int:
        return self._base_item._get_width_given_height(h)

    # .................................................................................................................
