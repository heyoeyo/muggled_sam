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
class ToolsGroup:
    hover: ToggleButton
    box: ToggleButton
    fgpt: ToggleButton
    bgpt: ToggleButton
    clear: ImmediateButton

    def totuple(self) -> tuple:
        """Helper which returns a tuple of tool items"""
        return (self.hover, self.box, self.fgpt, self.bgpt, self.clear)


@dataclass
class OverlaysGroup:
    hover: HoverOverlay
    box: BoxSelectOverlay
    fgpt: PointSelectOverlay
    bgpt: PointSelectOverlay
    polygon: DrawPolygonsOverlay

    def totuple(self) -> tuple:
        """Helper which returns a tuple of all overlays (with polygon first!)"""
        return (self.polygon, self.hover, self.box, self.fgpt, self.bgpt)


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SharedUI:
    """
    Somewhat hacky class used to bundle all 'standard' UI elements which are shared
    for (most?) SAM model use cases. This includes providing UI for selecting between
    hover/box/fg/bg prompt inputs, providing overlays for drawing prompts/mask outlines
    as well as providing UI for mask preview images.
    """

    # .................................................................................................................

    def __init__(self, full_image_bgr: ndarray, mask_predictions: Tensor):

        # Build out UI component pieces
        self.olays = self._build_overlays()
        self.tools, self.tools_constraint = self._build_tools()
        self.mask_btns, self.masks_constraint = self._build_mask_preview_buttons(mask_predictions)

        # Build main layout!
        self.image, self.display_block, self.layout = self._build_ui_layout(
            full_image_bgr, mask_predictions, self.olays, self.tools, self.mask_btns
        )

    # .................................................................................................................

    def _build_overlays(self) -> OverlaysGroup:
        """Helper used to build overlay UI (draws polygon outlines, bounding-boxes, points etc.)"""

        # Set up prompt UI interactions
        hover_olay = HoverOverlay()
        box_olay = BoxSelectOverlay(thickness=2)
        fgpt_olay = PointSelectOverlay((0, 255, 0), point_radius=3)
        bgpt_olay = PointSelectOverlay((0, 0, 0), bg_color=(0, 255, 0), point_radius=3).style(bg_thickness=2)
        pgon_olay = DrawPolygonsOverlay((100, 10, 255), bg_color=(0, 0, 0))

        return OverlaysGroup(hover_olay, box_olay, fgpt_olay, bgpt_olay, pgon_olay)

    # .................................................................................................................

    def _build_tools(self) -> tuple[ToolsGroup, RadioConstraint]:
        """Helper used to build tool group UI (tool select buttons + prompt clear button)"""

        # Set up tool selection UI
        hover_btn, box_btn, fgpt_btn, bgpt_btn = ToggleButton.many("Hover", "Box", "FG Point", "BG Point")
        clear_all_prompts_btn = ImmediateButton("Clear", color=(0, 0, 150))

        # Set up constraint so only 1 tool can be active (excluding clear button, which isn't toggled)
        tools_group = ToolsGroup(hover_btn, box_btn, fgpt_btn, bgpt_btn, clear_all_prompts_btn)
        tool_constraint = RadioConstraint(hover_btn, box_btn, fgpt_btn, bgpt_btn)

        return tools_group, tool_constraint

    # .................................................................................................................

    def _build_mask_preview_buttons(self, mask_predictions: Tensor) -> tuple[list[ToggleImage], RadioConstraint]:
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

    # .................................................................................................................

    def _build_ui_layout(
        self,
        full_image_bgr: ndarray,
        mask_preds: Tensor,
        overlays: OverlaysGroup,
        tools: ToolsGroup,
        mask_preview_buttons: list[ToggleImage],
    ) -> tuple[ExpandingImage, VStack | HStack, VStack]:
        """Function used to build out the final UI block, containing tool UI, overlays and mask previews"""

        # For convenience
        img_shape = full_image_bgr.shape
        mask_hw = mask_preds.shape[2:]

        # For tool button bar
        toolselect_bar = HStack(*tools.totuple())

        # Set up display image
        main_display_image = ExpandingImage(full_image_bgr).set_debug_name("ColorImg")
        imgoverlay_cb = OverlayStack(main_display_image, *overlays.totuple())
        imgoverlay_cb._rdr.pad.color = (35, 25, 30)

        # Set up mask previews, which are oriented/arranged based on display aspect ratio
        side_str, stack_order_str = find_best_display_arrangement(img_shape, mask_hw)
        order_lut = {
            "grid": VStack(HStack(*mask_preview_buttons[:2]), HStack(*mask_preview_buttons[2:])),
            "vertical": VStack(*mask_preview_buttons),
            "horizontal": HStack(*mask_preview_buttons),
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

        # Set up full display layout
        display_layout = VStack(toolselect_bar, main_display_block)
        display_layout.set_debug_name("DisplayLayout")

        # Tie tool overlays to toggle buttons (so only 1 overlay responds to user input, based on tool selection)
        tools.hover.add_on_change_listeners(overlays.hover.enable)
        tools.box.add_on_change_listeners(overlays.box.enable)
        tools.fgpt.add_on_change_listeners(overlays.fgpt.enable)
        tools.bgpt.add_on_change_listeners(overlays.bgpt.enable)

        return main_display_image, main_display_block, display_layout

    # .................................................................................................................


class UIControl:
    """
    Helper class used to manage access to the standard/shared UI implementation
    Helps to handle various typical interactions with prompt tools & mask display
    """

    # .................................................................................................................

    def __init__(self, shared_ui: SharedUI):
        self.elems = shared_ui
        self.checker_pattern = CheckerPattern()

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

    def update_main_display_image(
        self, image_bgr: ndarray, mask_uint8: ndarray, mask_contours_norm, show_with_alpha=False
    ) -> None:
        """Helper used to update the displayed image + mask outlines, with alpha checkerboarding if needed"""

        # Use checker background to suggest alpha channel if needed
        if show_with_alpha:
            image_bgr = self.checker_pattern.superimpose(image_bgr, mask_uint8)

        self.elems.olays.polygon.set_polygons(mask_contours_norm if not show_with_alpha else None)
        self.elems.image.set_image(image_bgr)

        return

    # .................................................................................................................

    def update_mask_previews(self, mask_predictions, mask_select_index, mask_threshold=0.0, invert_mask=False) -> None:
        """Updates mask preview buttons with binary copies of predictions"""

        # Update mask selection images
        mask_preds_uint8 = ((mask_predictions.squeeze(0) > mask_threshold) * 255).byte().cpu().numpy()
        for pred_idx, (mpred_uint8, mbtn) in enumerate(zip(mask_preds_uint8, self.elems.mask_btns)):
            mbtn.set_image(mpred_uint8 if not invert_mask else np.bitwise_not(mpred_uint8))

        return

    # .................................................................................................................

    def create_hires_mask_uint8(self, mask_predictions, mask_select_index, output_hw, mask_threshold=0.0) -> ndarray:
        """Draws binary mask matching the given output height & width"""

        mask_select = mask_predictions[:, mask_select_index, :, :].unsqueeze(1).float()
        mask_upscale = nn.functional.interpolate(mask_select, size=output_hw, mode="bilinear", align_corners=False)
        mask_uint8 = ((mask_upscale.squeeze() > mask_threshold) * 255).byte().cpu().numpy()

        return mask_uint8

    # .................................................................................................................

    def draw_iou_predictions(self, iou_predictions) -> None:
        """Draws IoU (%) on to mask preview buttons"""

        for pred_idx, mbtn in enumerate(self.elems.mask_btns):
            quality_estimate = round(float(iou_predictions[0, pred_idx].float().cpu()) * 100)
            mbtn.set_text(str(quality_estimate))

        return

    # .................................................................................................................

    def read_prompts(self) -> tuple[bool, list, list, list]:
        """
        Helper used to manage prompt reading, as well as quality-of-life behaviors when interpretting prompts
        Returns:
            need_prompt_reencoding, box_tlbr_list, fg_xy_list, bg_xy_list
        """

        # Read prompting controls
        is_tool_changed, tool_idx, selected_tool = self.elems.tools_constraint.read()
        clear_prompts = self.elems.tools.clear.read()
        if clear_prompts:
            self.elems.olays.box.clear()
            self.elems.olays.fgpt.clear()
            self.elems.olays.bgpt.clear()

        # Read prompt inputs
        box_prompt_changed, box_tlbr_norm_list = self.elems.olays.box.read()
        fg_prompt_changed, fg_xy_norm_list = self.elems.olays.fgpt.read()
        bg_prompt_changed, bg_xy_norm_list = self.elems.olays.bgpt.read()

        # Only add hover point when the tool is active
        hover_prompt_changed, hover_xy_norm_list = False, []
        if selected_tool is self.elems.tools.hover:
            hover_prompt_changed, hover_is_clicked, hover_xy_event = self.elems.olays.hover.read()
            if hover_xy_event.is_in_region:
                hover_xy_norm_list = [hover_xy_event.xy_norm]

            # If the user clicks while hovering, treat it as a foreground point (and switch to FG tool)
            if hover_is_clicked:
                self.elems.tools_constraint.change_to(self.elems.tools.fgpt)
                self.elems.olays.fgpt.add_points(hover_xy_event.xy_norm)

        # Add hover points (if any) as foreground prompts
        fg_xy_norm_list = [*fg_xy_norm_list, *hover_xy_norm_list]

        # Toggle back to hover in case where an fg point is removed and no other points remain
        if fg_prompt_changed:
            no_prompts = sum(len(pts) for pts in (fg_xy_norm_list, box_tlbr_norm_list, bg_xy_norm_list)) == 0
            if no_prompts:
                self.elems.tools_constraint.change_to(self.elems.tools.hover)

        # Decide if anything changed that would require us to re-encode prompts
        prompt_changed = any((box_prompt_changed, fg_prompt_changed, bg_prompt_changed, hover_prompt_changed))
        need_prompt_reencode = prompt_changed or is_tool_changed or clear_prompts

        return need_prompt_reencode, box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list

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
        self._full_img = full_image_bgr.copy()
        self._disp_img = full_image_bgr.copy()
        self._prev_h, self._prev_w = full_image_bgr.shape[0:2]

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
