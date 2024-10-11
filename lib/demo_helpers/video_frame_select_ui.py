#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os.path as osp
import cv2

from lib.demo_helpers.ui.window import DisplayWindow, KEY
from lib.demo_helpers.ui.sliders import HSlider
from lib.demo_helpers.ui.images import ExpandingImage
from lib.demo_helpers.ui.layout import VStack, HStack
from lib.demo_helpers.ui.buttons import ImmediateButton
from lib.demo_helpers.ui.static import StaticMessageBar

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def create_video_capture(video_path: str) -> tuple[bool, cv2.VideoCapture | None, ndarray | None]:

    # Initialize outputs
    is_valid = False
    vcap = None
    sample_frame = None

    # Bail if we got an image (opencv will read images as videos without obvious error!)
    test_img = cv2.imread(video_path)
    if test_img is not None:
        return is_valid, vcap

    # See if we can open the file as a video
    try:
        vcap = cv2.VideoCapture(video_path)
    except:
        return is_valid, vcap

    # Check that we can read the first frame of the video
    is_valid, sample_frame = vcap.read()
    if is_valid:
        vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return is_valid, vcap, sample_frame


# .....................................................................................................................


def make_video_frame_select_ui(
    title_text: str, frame: ndarray, max_frame_index: int
) -> tuple[VStack, tuple[ExpandingImage, HSlider, ImmediateButton]]:

    # Create UI elements for showing/controlling selected video frame
    main_img_elem = ExpandingImage(frame)
    frame_select_slider = HSlider("Frame index", 0, 0, int(max_frame_index), step_size=1)
    done_btn = ImmediateButton("Done", color=(125, 185, 0))

    # Hacky - Force done button to be smaller than slider using hidden UI properties
    frame_select_slider._rdr.limits.update(min_w=100)
    done_btn._rdr.limits.update(min_w=20)

    # Message bars for feedback
    header_bar = StaticMessageBar(title_text)
    footer_bar = StaticMessageBar("Use slider to select a frame", text_scale=0.35)

    # Bundle UI elements into single layout
    disp_layout = VStack(
        header_bar,
        main_img_elem,
        HStack(frame_select_slider, done_btn),
        footer_bar,
    )

    return disp_layout, (main_img_elem, frame_select_slider, done_btn)


# .....................................................................................................................


def run_video_frame_select_ui(
    video_path: str,
    render_height=800,
    initial_frame_index=0,
    window_title="Select frame - q to close",
):

    # Create video capture or bail if not possible
    ok_video, vcap, frame = create_video_capture(video_path)
    if not ok_video:
        return ok_video, frame

    # Create UI elements
    max_frame_idx = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    title_txt = osp.basename(video_path)
    vidui, (img_elem, frame_select_slider, done_btn) = make_video_frame_select_ui(title_txt, frame, max_frame_idx)

    # Setup window for display
    window = DisplayWindow(window_title, display_fps=30)
    window.attach_mouse_callbacks(vidui)
    window.attach_keypress_callback(KEY.ENTER, done_btn.click)
    window.move(200, 50)

    try:
        while True:

            # If the frame slider changes, try to read the selected frame
            frame_select_changed, frame_idx = frame_select_slider.read()
            if frame_select_changed:

                # Set frame index and read the frame
                vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok_frame, frame = vcap.read()

                # If reading fails, reset to first frame as fallback
                # -> This can happen with videos that have corrupted frame data, especially at the end of the file
                if not ok_frame:
                    vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok_frame, frame = vcap.read()
                    assert ok_frame, f"Error reading frame index {frame_idx}"

                # Update the displayed image
                img_elem.set_image(frame)

            # Update display
            display_img = vidui.render(h=900)
            req_break, keypress = window.show(display_img, 1 if frame_select_changed else None)
            if req_break:
                break

            # Finish when done is clicked
            if done_btn.read():
                break

    except KeyboardInterrupt:
        print("", "Quit by ctrl+c...", sep="\n")
        ok_video = False

    finally:
        cv2.destroyAllWindows()
        vcap.release()

    return ok_video, frame
