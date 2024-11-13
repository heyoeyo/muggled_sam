#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from dataclasses import dataclass

from .ui.window import KEY
from .ui.layout import HStack, VStack, OverlayStack
from .ui.buttons import ToggleButton, ToggleImage, ImmediateButton, RadioConstraint
from .ui.overlays import HoverOverlay, BoxSelectOverlay, PointSelectOverlay, DrawPolygonsOverlay
from .ui.images import ExpandingImage
from .ui.base import force_same_max_height, force_same_min_height

from .ui.helpers.images import CheckerPattern, blank_mask

import cv2
import numpy as np
import torch.nn as nn

# For type hints
from numpy import ndarray
from torch import Tensor
from .ui.window import DisplayWindow


# ---------------------------------------------------------------------------------------------------------------------
# %% Data types


@dataclass
class ToolButtonsGroup:
    hover: ToggleButton
    box: ToggleButton
    fgpt: ToggleButton
    bgpt: ToggleButton
    clear: ImmediateButton

    def totuple(self) -> tuple:
        """Helper which returns a tuple of tool items"""
        return (self.hover, self.box, self.fgpt, self.bgpt, self.clear)

    def enable(self, enabled=True):
        for item in self.totuple():
            item.enable(enabled)
        return self


@dataclass
class OverlayGroup:
    polygon: DrawPolygonsOverlay | None
    hover: HoverOverlay | None
    box: BoxSelectOverlay | None
    fgpt: PointSelectOverlay | None
    bgpt: PointSelectOverlay | None

    def totuple(self) -> tuple:
        """Helper which returns a tuple of all overlays"""
        return (self.polygon, self.hover, self.box, self.fgpt, self.bgpt)

    def enable(self, enabled=True):
        for item in self.totuple():
            item.enable(enabled)
        return self

    def clear_all(self, flag_is_changed=True):
        for item in self.totuple():
            try:
                item.clear(flag_is_changed)
            except TypeError:
                item.clear()
        return self


# ---------------------------------------------------------------------------------------------------------------------
# %% Builder functions


def build_tool_overlays() -> OverlayGroup:
    """Helper used to build overlay UI (draws polygon outlines, bounding-boxes, points etc.)"""

    # Set up prompt UI interactions
    hover_olay = HoverOverlay()
    box_olay = BoxSelectOverlay(thickness=2)
    fgpt_olay = PointSelectOverlay((0, 255, 0), point_radius=3)
    bgpt_olay = PointSelectOverlay((0, 0, 0), bg_color=(0, 255, 0), point_radius=3).style(bg_thickness=2)
    polygon_olay = DrawPolygonsOverlay((100, 10, 255), bg_color=(0, 0, 0))

    return OverlayGroup(polygon_olay, hover_olay, box_olay, fgpt_olay, bgpt_olay)


def build_tool_buttons(text_scale=0.75) -> tuple[ToolButtonsGroup, RadioConstraint]:
    """Helper used to build tool group UI (tool select buttons + prompt clear button)"""

    # Set up tool selection UI
    hover_btn, box_btn, fgpt_btn, bgpt_btn = ToggleButton.many(
        "Hover", "Box", "FG Point", "BG Point", text_scale=text_scale
    )
    clear_all_prompts_btn = ImmediateButton("Clear", color=(0, 0, 150), text_scale=text_scale)

    # Set up constraint so only 1 tool can be active (excluding clear button, which isn't toggled)
    tools_group = ToolButtonsGroup(hover_btn, box_btn, fgpt_btn, bgpt_btn, clear_all_prompts_btn)
    tool_constraint = RadioConstraint(hover_btn, box_btn, fgpt_btn, bgpt_btn)

    return tools_group, tool_constraint


def build_mask_preview_buttons(mask_predictions: Tensor) -> tuple[list[ToggleImage], RadioConstraint]:
    """Helper used to build mask previews UI (set of mask images showing model predictions)"""

    mask_hw = mask_predictions.shape[2:]

    # Set up mask preview selection UI
    blank_mask_btn_imgs = [blank_mask(*mask_hw)] * 4
    mask_btns_list = ToggleImage.many(*blank_mask_btn_imgs, highlight_color=(0, 120, 255))

    # Configure text, in case iou prediction needs to be drawn into mask previews
    for mbtn in mask_btns_list:
        mbtn.set_text(text=None, scale=0.35, xy_norm=(1, 1), offset_xy_px=(-6, -6), bg_color=(0, 0, 0))

    # Set up constraint so only 1 mask button can be active
    mask_constraint = RadioConstraint(*mask_btns_list, initial_selected_index=1)

    return mask_btns_list, mask_constraint


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class PromptUI:
    """
    Hacky-ish class used to bundle all 'base' UI elements which are shared
    for (most?) SAM model use cases. This consists of a regular image stacked
    alongside multiple mask images (which also act as radio buttons) for showing
    SAM mask results, along with support for rendering segmentation outlines
    as an overlay. This is stored as a 'display_block' component

    Additionally this class includes a tool-select bar for providing model prompts
    through hover/box/fg-point/bg-point buttons, along with the corresponding
    overlay graphics (i.e. rendering boxes or fg points)
    """

    # .................................................................................................................

    def __init__(
        self, full_image_bgr: ndarray, mask_predictions: Tensor, target_aspect_ratio=2.0, tool_button_text_scale=0.75
    ):

        # Build out UI component pieces
        self.olays = build_tool_overlays()
        self.tools, self.tools_constraint = build_tool_buttons(tool_button_text_scale)
        self.mask_btns, self.masks_constraint = build_mask_preview_buttons(mask_predictions)

        # Build main layout!
        self.image, self.overlay_img, self.display_block = self._build_display_block(
            full_image_bgr, mask_predictions, target_aspect_ratio
        )
        self.layout = self._build_ui_layout()

    # .................................................................................................................

    def _build_display_block(
        self, image_bgr: ndarray, mask_preds: Tensor, target_ar: float
    ) -> tuple[ExpandingImage, HStack]:
        """Function used to build out the centrl display block, containing image & mask previews"""

        # For convenience
        img_shape = image_bgr.shape
        mask_hw = mask_preds.shape[2:]

        # Set up display image
        main_display_image = ExpandingImage(image_bgr).set_debug_name("MainDisplayImage")
        imgoverlay_cb = OverlayStack(main_display_image, *self.olays.totuple())
        imgoverlay_cb._rdr.pad.color = (35, 25, 30)

        # Set up mask previews, which are oriented/arranged based on display aspect ratio
        mask_btns = self.mask_btns
        side_str, stack_order_str = find_best_display_arrangement(img_shape, mask_hw, target_ar)
        order_lut = {
            "grid": VStack(HStack(*mask_btns[:2]), HStack(*mask_btns[2:])),
            "vertical": VStack(*mask_btns),
            "horizontal": HStack(*mask_btns),
        }
        maskselect_stack = order_lut[stack_order_str]
        maskselect_stack._rdr.pad.color = (60, 60, 60)
        maskselect_stack.set_debug_name("MaskStack")

        # Set up main display block, which combines image & mask previews
        item_stack = HStack
        item_order = (imgoverlay_cb, maskselect_stack)
        if side_str != "right":
            item_stack = VStack
            item_order = reversed(item_order)
        main_display_block = item_stack(*item_order)
        main_display_block.set_debug_name("MainDisplayBlock")

        return main_display_image, imgoverlay_cb, main_display_block

    # .................................................................................................................

    def _build_ui_layout(self) -> VStack:
        """Function used to build out the final UI block which includes prompt tool buttons"""

        # Tie tool overlays to toggle buttons (so only 1 overlay responds to user input, based on tool selection)
        self.tools.hover.add_on_change_listeners(self.olays.hover.enable)
        self.tools.box.add_on_change_listeners(self.olays.box.enable)
        self.tools.fgpt.add_on_change_listeners(self.olays.fgpt.enable)
        self.tools.bgpt.add_on_change_listeners(self.olays.bgpt.enable)

        # Set up full display layout
        toolselect_bar = HStack(*self.tools.totuple())
        display_layout = VStack(toolselect_bar, self.display_block)
        display_layout.set_debug_name("DisplayLayout")

        return display_layout

    # .................................................................................................................

    def enable_tools(self, enable=True, clear_prompt_data_on_disable=True):
        """Helper used to enable/disable the tool components of the UI"""

        self.tools.enable(enable)
        if enable:
            # Only the overlay of the selected tool should be enabled
            _, tool_select_idx, _ = self.tools_constraint.read()
            olays_ordered_by_tool = [self.olays.hover, self.olays.box, self.olays.fgpt, self.olays.bgpt]
            olay_select = olays_ordered_by_tool[tool_select_idx]
            olay_select.enable(enable)

        # Wipe out prompt data if needed (to clear display of prompts)
        if not enable and clear_prompt_data_on_disable:
            self.clear_prompts()

        return self

    # .................................................................................................................

    def enable_masks(self, enable=True):
        """Enables/disables ability to interact with mask previews"""

        for mask_btn in self.mask_btns:
            mask_btn.enable(enable)

        return

    # .................................................................................................................

    def clear_prompts(self):
        """Helper used to wipe out prompt data from overlays"""

        self.olays.hover.clear()
        self.olays.box.clear()
        self.olays.fgpt.clear()
        self.olays.bgpt.clear()

        return self

    # .................................................................................................................


class BaseUIControl:
    """
    Helper class used to manage access to the base UI implementation
    Includes functionality for rendering alpha/checker-board masked images,
    as well as mask preview buttons & final 'hi-res' mask results
    """

    # .................................................................................................................

    def __init__(self, ui_elements: PromptUI):
        self.elems = ui_elements
        self._checker_pattern = CheckerPattern()

    # .................................................................................................................

    def attach_arrowkey_callbacks(self, window: DisplayWindow):
        """Helper used to attach keypress callbacks so that mask previews can be switched with arrow keys"""

        window.attach_keypress_callback(KEY.UP_ARROW, self.elems.masks_constraint.previous)
        window.attach_keypress_callback(KEY.DOWN_ARROW, self.elems.masks_constraint.next)

        return self

    # .................................................................................................................

    def update_main_display_image(
        self, image_bgr: ndarray, mask_uint8: ndarray, mask_contours_norm, show_with_alpha=False
    ) -> None:
        """Helper used to update the displayed image + mask outlines, with alpha checkerboarding if needed"""

        # Use checker background to suggest alpha channel if needed
        if show_with_alpha:
            image_bgr = self._checker_pattern.superimpose(image_bgr, mask_uint8)

        self.elems.olays.polygon.set_polygons(mask_contours_norm if not show_with_alpha else None)
        self.elems.image.set_image(image_bgr)

        return

    # .................................................................................................................

    def update_mask_previews(self, mask_predictions, mask_threshold=0.0, invert_mask=False) -> None:
        """Updates mask preview buttons with binary copies of predictions"""
        return update_mask_preview_buttons(mask_predictions, self.elems.mask_btns, mask_threshold, invert_mask)

    # .................................................................................................................

    def draw_iou_predictions(self, iou_predictions) -> None:
        """Draws IoU (%) on to mask preview buttons"""

        for pred_idx, mbtn in enumerate(self.elems.mask_btns):
            quality_estimate = round(float(iou_predictions[0, pred_idx].float().cpu()) * 100)
            mbtn.set_text(str(quality_estimate))

        return

    # .................................................................................................................

    @staticmethod
    def create_hires_mask_uint8(mask_predictions, mask_select_index, output_hw, mask_threshold=0.0) -> ndarray:
        """Draws binary mask matching the given output height & width. Returns: mask_uint8_1ch"""
        mask_select = mask_predictions[:, mask_select_index].float()
        return make_hires_mask_uint8(mask_select, output_hw, mask_threshold).squeeze(0)

    # .................................................................................................................


class PromptUIControl(BaseUIControl):
    """
    An extension to the BaseUIControl class, with additional
    functions for dealing with prompts
    """

    # .................................................................................................................

    def __init__(self, prompt_ui: PromptUI):
        super().__init__(prompt_ui)

    # .................................................................................................................

    def load_initial_prompts(self, initial_prompts_dict: dict):
        """
        Can be used to initialize prompts based on a given dictionary
        The dictionary should contain keys: "boxes", "fg_points", "bg_points"
        """

        # Bail if we aren't given the right data type (e.g. given 'None')
        if not isinstance(initial_prompts_dict, dict):
            return

        # Exhaust the clear button, in case it's been triggered (don't want next read to clear our init prompts!)
        self.elems.tools.clear.read()

        # Initialize overlays with starting prompts
        self.elems.olays.box.add_boxes(*initial_prompts_dict.get("boxes", []))
        self.elems.olays.fgpt.add_points(*initial_prompts_dict.get("fg_points", []))
        self.elems.olays.bgpt.add_points(*initial_prompts_dict.get("bg_points", []))

        return

    # .................................................................................................................

    def attach_arrowkey_callbacks(self, window: DisplayWindow):
        """Helper used to attach keypress callbacks so that tools & mask previews can be switched with arrow keys"""

        tool_const = self.elems.tools_constraint
        mask_const = self.elems.masks_constraint
        window.attach_keypress_callback(KEY.LEFT_ARROW, tool_const.previous)
        window.attach_keypress_callback(KEY.RIGHT_ARROW, tool_const.next)
        window.attach_keypress_callback(KEY.UP_ARROW, mask_const.previous)
        window.attach_keypress_callback(KEY.DOWN_ARROW, mask_const.next)

        return self

    # .................................................................................................................

    def read_prompts(self) -> tuple[bool, list, list, list]:
        """
        Helper used to manage prompt reading, as well as quality-of-life behaviors when interpretting prompts
        Returns:
            need_prompt_reencoding, (boxes_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
        """

        need_prompt_clear = self.elems.tools.clear.read()
        if need_prompt_clear:
            self.elems.olays.clear_all(flag_is_changed=True)

        return read_prompts(self.elems.olays, self.elems.tools, self.elems.tools_constraint)

    # .................................................................................................................


class ReusableBaseImage:
    """
    Convenience class, used to manage a re-usable (static) image
    that is re-sized to a target display size from an original
    image that is assumed to be much larger.
    This can help reduce cpu load since we avoid using
    (and repeatedly downscaling) the original image which may
    be much larger/heavier to work with!
    """

    # .................................................................................................................

    def __init__(self, full_image_bgr: ndarray):

        # Initialize state values
        self._full_img = self._disp_img = self._prev_h = self._prev_w = None
        self.set_new_image(full_image_bgr)

    def set_new_image(self, new_image_bgr: ndarray):
        """
        Store a new image to be cached. This isn't expected to happen often!
        (setting the image frequently defeats the purpose of caching)
        """

        self._full_img = new_image_bgr
        self._disp_img = new_image_bgr.copy()
        self._prev_h, self._prev_w = new_image_bgr.shape[0:2]

        return self

    def regenerate(self, new_display_hw):
        """Resizes the original input image to the given display size or re-uses a cached copy at the given size"""

        # Resize original image to given display size and store for re-use
        disp_h, disp_w = new_display_hw
        if disp_h != self._prev_h or disp_w != self._prev_w:
            self._disp_img = cv2.resize(self._full_img, dsize=(disp_w, disp_h))
            prev_disp_h, prev_disp_w = self._disp_img.shape[0:2]

        return self._disp_img

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def read_prompts(
    overlays_group: OverlayGroup,
    tools_group: ToolButtonsGroup,
    tools_constraint: RadioConstraint,
    debug_name="",
) -> tuple[bool, tuple[list, list, list]]:
    """
    Helper used to standardize prompt reading, assuming standard
    hover/box/fg/bg UI.

    Returns:
        is_prompt_changed, (boxes_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
    """

    # Read all overlay states (which is where prompts are actually held!)
    box_prompt_changed, boxes_tlbr_norm_list = overlays_group.box.read()
    fg_prompt_changed, fg_xy_norm_list = overlays_group.fgpt.read()
    bg_prompt_changed, bg_xy_norm_list = overlays_group.bgpt.read()
    _, _, selected_tool_elem = tools_constraint.read()

    # Only add hover point when the tool is active
    hover_changed = False
    if selected_tool_elem is tools_group.hover:

        # Add hover points (if any) as foreground prompts
        hover_changed, clicked_while_hovering, hover_xy_event = overlays_group.hover.read()
        if hover_xy_event.is_in_region:
            fg_xy_norm_list = tuple([*fg_xy_norm_list, hover_xy_event.xy_norm])

        # Treat hover clicks as foreground points (and switch to FG tool)
        if clicked_while_hovering:
            tools_constraint.change_to(tools_group.fgpt)
            overlays_group.fgpt.add_points(hover_xy_event.xy_norm)

    # Toggle back to hover in case where an fg point is removed and no other points remain
    if fg_prompt_changed:
        no_prompts = sum(len(pts) for pts in (fg_xy_norm_list, boxes_tlbr_norm_list, bg_xy_norm_list)) == 0
        if no_prompts:
            tools_constraint.change_to(tools_group.hover)

    # Bundle all prompt changes into single check for convenience
    is_prompt_changed = any((hover_changed, box_prompt_changed, fg_prompt_changed, bg_prompt_changed))

    return is_prompt_changed, (boxes_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)


# .....................................................................................................................


def find_best_display_arrangement(image_shape, mask_shape, target_ar=2.0, num_masks=4):
    """
    Helper function used to decide how to arrange a display made up of a
    color display image and corresponding mask predictions, stacked either
    above or to the right of the display.

    This is needed to account for displaying images of varying aspect ratios,
    where stacking multiple mask images may make the display too tall
    or too wide if not aranged properly.

    Returns:
        best_side_str, best_order_str
        -> side string is one of: "vertical", "horizontal" or "grid"
        -> order string is one of: "right" or "top"
    """

    # For convenience
    img_h, img_w = image_shape[0:2]
    mask_h, mask_w = mask_shape[0:2]

    # Set shared values for figuring out stacking
    right_side_str, top_side_str = "right", "top"
    vert_str, horz_str, grid_str = "vertical", "horizontal", "grid"
    tallmask_w = mask_w * (img_h / mask_h)
    widemask_h = mask_h * (img_w / mask_w)

    # Set up sizing configuration for each right/top + vert/horz/grid stacking arrangment
    configurations_list = [
        (right_side_str, vert_str, 0, (tallmask_w // num_masks)),
        (right_side_str, horz_str, 0, img_w + (tallmask_w * num_masks)),
        (right_side_str, grid_str, 0, tallmask_w),
        (top_side_str, vert_str, widemask_h * num_masks, 0),
        (top_side_str, horz_str, (widemask_h // num_masks), 0),
        (top_side_str, grid_str, widemask_h, 0),
    ]

    # Figure out which arrangement gives the best aspect ratio for display
    ardelta_side_order_list = []
    for side_str, order_str, add_h, add_w in configurations_list:
        ar_delta = abs(target_ar - (img_w + add_w) / (img_h + add_h))
        ardelta_side_order_list.append((ar_delta, side_str, order_str))
    _, best_side, best_order = min(ardelta_side_order_list)

    return best_side, best_order


# .....................................................................................................................


def update_mask_preview_buttons(
    mask_predictions: Tensor,
    mask_buttons: list[ToggleImage],
    mask_threshold=0.0,
    invert_mask=False,
) -> None:
    """
    Helper used to draw updated mask preview buttons (used for selecting which mask to use)
    Supports inversion as well as adjusting the threshold level used for the display.
    Assumes predictions are given with shape: 1x4xHxW
    -> Where 1 is batch size (only supports batch of 1)
    -> Expects 4 masks of any H/W

    Returns nothing!
    """

    # Update mask selection images
    mask_preds_uint8 = ((mask_predictions.squeeze(0) > mask_threshold) * 255).byte().cpu().numpy()
    for pred_idx, (mpred_uint8, mbtn) in enumerate(zip(mask_preds_uint8, mask_buttons)):
        mbtn.set_image(mpred_uint8 if not invert_mask else np.bitwise_not(mpred_uint8))

    return


# .....................................................................................................................


def make_hires_mask_uint8(mask_prediction: Tensor, output_hw=(1024, 1024), mask_threshold=0.0) -> ndarray:
    """
    Helper used to draw a high-resolution binary uint8 (numpy)
    version of a given mask prediction. Expects a single mask
    prediction, but will work with a variety of input shapes.
    Supports inputs of shape:
        BxNxHxW, NxHxW, HxW

    Will return result with matching shape dimensions,
    but with output H/W sizing in numpy uint8 format
    """

    # Set up dimension padding (for interpolation func) & post-indexing to remove padded dimensions
    preds, squeeze_idx = mask_prediction, slice(None)
    if preds.ndim == 3:
        preds = mask_prediction[None]
        squeeze_idx = (0,)
    elif preds.ndim == 2:
        preds = mask_prediction[None, None]
        squeeze_idx = (0, 0)

    mask_upscale = nn.functional.interpolate(preds, size=output_hw, mode="bilinear", align_corners=False)
    return ((mask_upscale[squeeze_idx] > mask_threshold) * 255).byte().cpu().numpy()
