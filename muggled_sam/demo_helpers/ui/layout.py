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


class GridStack(BaseCallback):
    """
    Layout which combines elements into a grid with a specified number of rows and columns
    Items should be given in 'row-first' format (i.e. items fill out grid row-by-row)
    """

    # .................................................................................................................

    def __init__(self, *items, num_rows=None, num_columns=None, target_aspect_ratio=1):

        super().__init__(128, 128)
        self.append_children(*items)

        # Fill in missing row/column counts
        num_items = len(items)
        if num_rows is None and num_columns is None:
            num_rows, num_columns = self.get_row_column_by_aspect_ratio(num_items, target_aspect_ratio)
        elif num_rows is None:
            num_rows = int(np.ceil(num_items / num_columns))
        elif num_columns is None:
            num_columns = int(np.ceil(num_items / num_rows))
        self._num_rows = num_rows
        self._num_cols = num_columns

        # Update stack sizing based on children
        min_h_per_row = []
        for _, child_per_row in self.row_iter():
            min_h_per_row.append(max(child._rdr.limits.min_h for child in child_per_row))
        min_w_per_col = []
        for _, child_per_col in self.column_iter():
            min_w_per_col.append(max(child._rdr.limits.min_w for child in child_per_col))
        total_min_h = sum(min_h_per_row)
        total_min_w = sum(min_w_per_col)
        self._rdr.limits.update(min_h=total_min_h, min_w=total_min_w, expand_h=True, expand_w=True)

    # .................................................................................................................

    def __repr__(self):
        child_names = [str(child._debug_name) for child in self]
        return f"{self._debug_name} [{', '.join(child_names)}]"

    # .................................................................................................................

    def get_row_columns(self) -> tuple[int, int]:
        """Get current row/column count of the grid layout"""
        return (self._num_rows, self._num_cols)

    # .................................................................................................................

    def transpose(self):
        """Flip number of rows & columns"""
        self._num_rows, self._num_cols = self._num_cols, self._num_rows
        return self

    # .................................................................................................................

    def row_iter(self):
        """
        Iterator over items per row. Example:
            for row_idx, items_in_row in grid.row_iter():
                # ... Do something with each row ...

                for col_idx, item for enumerate(items_in_row):
                    # ... Do something with each item ...
                    pass
                pass
        """

        for row_idx in range(self._num_rows):
            idx1 = row_idx * self._num_cols
            idx2 = idx1 + self._num_cols
            items_in_row = self[idx1:idx2]
            if len(items_in_row) == 0:
                break
            yield row_idx, tuple(items_in_row)

        return

    # .................................................................................................................

    def column_iter(self):
        """
        Iterator over items per column. Example:
            for col_idx, items_in_column in grid.column_iter():
                # ... Do something with each column ...

                for row_idx, item for enumerate(items_in_column):
                    # ... Do something with each item ...
                    pass
                pass
        """

        num_items = len(self)
        for col_idx in range(self._num_cols):
            item_idxs = [col_idx + row_idx * self._num_cols for row_idx in range(self._num_rows)]
            items_in_column = tuple(self[item_idx] for item_idx in item_idxs if item_idx < num_items)
            if len(items_in_column) == 0:
                break
            yield col_idx, items_in_column

        return

    # .................................................................................................................

    def grid_iter(self):
        """
        Iterator over all items while returning row/column index. Example:
            for row_idx, col_idx, item in grid.grid_iter():
                # ... Do something with each item ...
                pass
        """

        for item_idx, item in enumerate(self):
            row_idx = item_idx // self._num_cols
            col_idx = item_idx % self._num_cols
            yield row_idx, col_idx, item

        return

    # .................................................................................................................

    def _render_up_to_size(self, h, w):

        # Set up starting stack point, used to keep track of child callback regions
        x_stack = self._cb_region.x1
        y_stack = self._cb_region.y1

        # Figure out how tall each row should be
        ideal_h_per_row = h // self._num_rows
        h_gap = h % self._num_rows
        h_per_row = [ideal_h_per_row + int(idx < h_gap) for idx in range(self._num_rows)]

        # Figure out how wide each column should be
        ideal_w_per_col = w // self._num_cols
        w_gap = w % self._num_cols
        w_per_col = [ideal_w_per_col + int(idx < w_gap) for idx in range(self._num_cols)]

        # Render all child items to target sizing
        row_images_list = []
        for row_idx, children_per_row in self.row_iter():

            row_height = h_per_row[row_idx]
            col_images_list = []
            for col_idx, child in enumerate(children_per_row):
                col_width = w_per_col[col_idx]
                frame = child._render_up_to_size(row_height, col_width)
                orig_frame_h, orig_frame_w = frame.shape[0:2]

                # Adjust frame width if needed
                tpad, bpad, lpad, rpad = 0, 0, 0, 0
                if (orig_frame_h < row_height) or (orig_frame_w < col_width):
                    available_h, available_w = row_height - orig_frame_h, col_width - orig_frame_w
                    lpad = available_w // 2
                    rpad = available_w - lpad
                    tpad = available_h // 2
                    bpad = available_h - tpad
                    ptype = child._rdr.pad.style
                    pcolor = child._rdr.pad.color
                    frame = cv2.copyMakeBorder(frame, tpad, bpad, lpad, rpad, ptype, value=pcolor)

                # Crop oversized heights
                if orig_frame_h > row_height:
                    print(
                        f"Render sizing error! ({child._debug_name})",
                        f"  Expecting height: {h}, got {orig_frame_h}",
                        "-> Will crop!",
                        sep="\n",
                    )
                    frame = frame[:row_height, :, :]
                    orig_frame_h = row_height

                # Crop oversized widths
                if orig_frame_w > col_width:
                    print(
                        f"Render sizing error! ({child._debug_name})",
                        f"  Expecting width: {w}, got {orig_frame_w}",
                        "-> Will crop!",
                        sep="\n",
                    )
                    frame = frame[:, :col_width, :]
                    orig_frame_w = col_width

                # Store image for forming row-images
                col_images_list.append(frame)

                # Provide callback region to child item
                x1, y1 = x_stack + lpad, y_stack + tpad
                x2, y2 = x1 + orig_frame_w, y1 + orig_frame_h
                child._cb_region.update(x1, y1, x2, y2)

                # Update x-stacking point for each column
                x_stack = x2 + rpad

            # Combine all column images to form one row image, padding if needed
            one_row_image = np.hstack(col_images_list)
            _, one_row_w = one_row_image.shape[0:2]
            if one_row_w < w:
                pad_w = w - one_row_w
                ptype = self._rdr.pad.style
                pcolor = self._rdr.pad.color
                one_row_image = cv2.copyMakeBorder(one_row_image, 0, 0, 0, pad_w, ptype, value=pcolor)
            row_images_list.append(one_row_image)

            # Reset x-stacking point & update y-stacking point, for each completed row
            x_stack = self._cb_region.x1
            y_stack = y_stack + row_height

        return np.vstack(row_images_list)

    # .................................................................................................................

    def _get_height_and_width_without_hint(self) -> [int, int]:
        """Set height to the total of largest heights per row, width to the total largest widths per column"""

        # Set height based on largest heights per row
        max_h_per_row = []
        for _, items_per_row in self.row_iter():
            max_h_per_row.append(max(item._rdr.limits.min_h for item in items_per_row))
        height = sum(max_h_per_row)

        # Set width based on largest widths per column
        max_w_per_col = []
        for _, items_per_col in self.column_iter():
            max_w_per_col.append(max(item._rdr.limits.min_w for item in items_per_col))
        width = sum(max_w_per_col)

        return height, width

    def _get_height_given_width(self, w) -> int:
        """Set height to the sum of the tallest elements per row"""

        # Figure out width of each column (assuming equal assignment)
        ideal_w_per_col = w // self._num_cols
        w_gap = w % self._num_cols
        w_per_col = [ideal_w_per_col + int(idx < w_gap) for idx in range(self._num_cols)]

        # Sum up all widths per row
        heights_per_row = [[] for _ in range(self._num_rows)]
        for row_idx, col_idx, child in self.grid_iter():
            heights_per_row[row_idx].append(child._update_render_sizing(w=w_per_col[col_idx]).h)

        # Set height to the total height based on tallest item per row stacked together
        max_height_per_row = [max(row_heights) for row_heights in heights_per_row]
        return sum(max_height_per_row)

    def _get_width_given_height(self, h) -> int:
        """Set width to the widest row"""

        # Figure out height of each row (assuming equal assignment)
        ideal_h_per_row = h // self._num_rows
        h_gap = h % self._num_rows
        h_per_row = [ideal_h_per_row + int(idx < h_gap) for idx in range(self._num_rows)]

        # Sum up all widths per row
        total_w_per_row = [0] * self._num_rows
        for row_idx, col_idx, child in self.grid_iter():
            total_w_per_row[row_idx] += child._update_render_sizing(h=h_per_row[row_idx]).w

        # Set width to the widest row
        return max(total_w_per_row)

    # .................................................................................................................

    @staticmethod
    def get_row_column_options(num_items) -> list[tuple[int, int]]:
        """
        Helper used to get all possible neatly divisible combinations of (num_rows, num_columns)
        for a given number of items, in order of fewest rows -to- most rows.
        For example for num_items = 6, returns:
            ((1, 6), (2, 3), (3, 2), (6, 1))
            -> This is meant to be interpreted as:
                (1 row, 6 columns) OR (2 rows, 3 columns) OR (3 rows, 2 columns) OR (6 rows, 1 column)

        As another example, for num_items = 12, returns:
            ((1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1))
        """
        return tuple((k, num_items // k) for k in range(1, 1 + num_items) if (num_items % k) == 0)

    # .................................................................................................................

    @staticmethod
    def get_aspect_ratio_similarity(row_column_options, target_aspect_ratio) -> list[float]:
        """
        Compute similarity score (0 to 1) indicating how close of match
        each row/column option is to the target aspect ratio.
        """
        target_theta, pi_over_2 = np.arctan(target_aspect_ratio), np.pi / 2
        difference_scores = (abs(np.arctan(col / row) - target_theta) for row, col in row_column_options)
        return tuple(float(1.0 - (diff / pi_over_2)) for diff in difference_scores)

    # .................................................................................................................

    @classmethod
    def get_row_column_by_aspect_ratio(cls, num_items, target_aspect_ratio=1.0) -> tuple[int, int]:
        """
        Helper used to choose the number of rows & columns to best match a target aspect ratio
        Returns: (num_rows, num_columns)
        """

        rc_options = cls.get_row_column_options(num_items)
        ar_similarity = cls.get_aspect_ratio_similarity(rc_options, target_aspect_ratio)
        best_match_idx = np.argmax(ar_similarity)

        return rc_options[best_match_idx]

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

    def add_overlays(self, *overlay_items):
        """Function used to add overlays (after init)"""

        olays_list = list(self._overlay_items)
        olays_list.extend(overlay_items)
        self._overlay_items = tuple(olays_list)
        self.append_children(*overlay_items)

        return self

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
