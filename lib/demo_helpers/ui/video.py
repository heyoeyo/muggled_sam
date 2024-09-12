#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2

from .base import BaseCallback
from .helpers.images import blank_image, get_image_hw_for_max_side_length

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class LoopingVideoReader:
    """
    Helper used to provide looping frames from video, along with helpers
    to control playback & frame sizing
    Example usage:

        vreader = LoopingVideoReader("path/to/video.mp4")
        for is_paused, frame_idx, frame in vreader:
            # Do something with frames...
            if i_want_to_stop:
                break

    """

    # .................................................................................................................

    def __init__(self, video_path: str, display_size_px: int | None = None, initial_position_0_to_1: float = 0.0):

        # Store basic video data
        self._video_path = video_path
        self._vcap = cv2.VideoCapture(self._video_path)
        self.total_frames = int(self._vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._max_frame_idx = self.total_frames - 1
        self._fps = self._vcap.get(cv2.CAP_PROP_FPS)

        # Jump ahead to a different starting position if needed
        if initial_position_0_to_1 > 0.001:
            self._vcap.set(cv2.CAP_PROP_POS_FRAMES, self._max_frame_idx * initial_position_0_to_1)

        # Read sample frame & reset video
        rec_frame, first_frame = self._vcap.read()
        if not rec_frame:
            raise IOError(f"Can't read frames from video! ({video_path})")
        self._vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.sample_frame = first_frame

        # Set up display sizing
        self._need_resize = display_size_px is not None
        self._scale_wh = (first_frame.shape[1], first_frame.shape[0])
        if self._need_resize:
            self._scale_wh = get_image_hw_for_max_side_length(first_frame, display_size_px)
        self.shape = (self._scale_wh[1], self._scale_wh[0], 3)

        # Allocate storage for 'previous frame', which is re-used when paused &
        self._is_paused = False
        self._frame_idx = 0
        self._pause_frame = self.scale_to_display_wh(first_frame) if self._need_resize else first_frame

    # .................................................................................................................

    def scale_to_display_wh(self, image) -> ndarray:
        """Helper used to scale a given image to a target display size (if configured)"""
        return cv2.resize(image, dsize=self.disp_wh)

    def release(self):
        """Close access to video source"""
        self._vcap.release()
        return self

    def pause(self, set_is_paused=True) -> bool:
        """Pause/unpause the video"""
        self._is_paused = set_is_paused
        return self._is_paused

    def toggle_pause(self) -> bool:
        """Helper used to toggle pause state (meant for keypress events)"""
        new_pause_state = not self._is_paused
        self._is_paused = new_pause_state
        return new_pause_state

    # .................................................................................................................

    def get_sample_frame(self) -> ndarray:
        """Helper used to retrieve a sample frame (the first frame), most likely for init use-cases"""
        return self.sample_frame.copy()

    # .................................................................................................................

    def get_pause_state(self) -> bool:
        """Helper used to figure out if the video is paused (separately from the frame iteration)"""
        return self._is_paused

    # .................................................................................................................

    def get_frame_delay_ms(self, max_allowable_ms=1000) -> int:
        """Returns a frame delay (in milliseconds) according to the video's reported framerate"""
        frame_delay_ms = 1000.0 / self._fps
        return int(min(max_allowable_ms, frame_delay_ms))

    # .................................................................................................................

    def get_playback_position(self, normalized=True) -> int | float:
        """Returns playback position either as a frame index or a number between 0 and 1 (if normalized)"""
        if normalized:
            return self._vcap.get(cv2.CAP_PROP_POS_FRAMES) / self._max_frame_idx
        return int(self._vcap.get(cv2.CAP_PROP_POS_FRAMES))

    # .................................................................................................................

    def set_playback_position(self, position: int | float, is_normalized=False) -> int:
        """Set position of video playback. Returns frame index"""

        frame_idx = round(position * self._max_frame_idx) if is_normalized else position
        frame_idx = max(min(frame_idx, self._max_frame_idx), 0)

        self._vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._frame_idx = frame_idx

        # If we're paused, but set a new frame, then update the pause frame
        # -> This is important for paused 'timeline scrubbing' to work intuitively
        if self._is_paused:
            ok_read, frame = self._vcap.read()
            self._pause_frame = frame if ok_read else self._pause_frame

        return frame_idx

    # .................................................................................................................

    def __iter__(self):
        """Called when using this object in an iterator (e.g. for loops)"""
        if not self._vcap.isOpened():
            self._vcap = cv2.VideoCapture(self._video_path)
        return self

    # .................................................................................................................

    def __next__(self) -> [bool, int, ndarray]:
        """
        Iterator that provides frame data from a video capture object.
        Returns:
            is_paused, frame_index, frame_bgr
        """

        # Don't read video frames while paused
        if self._is_paused:
            return self._is_paused, self._frame_idx, self._pause_frame.copy()

        # Read next frame, or loop back to beginning if there are no more frames
        self._frame_idx += 1
        read_ok, frame = self._vcap.read()
        if not read_ok:
            self._frame_idx = 0
            self._vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            read_ok, frame = self._vcap.read()
            assert read_ok, "Error looping video! Unable to read first frame"

        # Scale frame for display & store in case we pause
        if self._need_resize:
            frame = self.scale_to_display_wh(frame)
        self._pause_frame = frame

        return self._is_paused, self._frame_idx, self._pause_frame.copy()

    # .................................................................................................................


class ReversibleLoopingVideoReader(LoopingVideoReader):
    """
    Simple variant on the basic looping reader.
    This version supports reading frames in reverse, though
    the implementation is extremely inefficient!

    To use, simply call the '.toggle_reverse_state(True)' function.
    (This can be done inside of a frame reading loop)
    """

    # .................................................................................................................

    def __init__(self, video_path: str, display_size_px: int | None = None, initial_position_0_to_1: float = 0.0):

        # Inherit from parent
        super().__init__(video_path, display_size_px, initial_position_0_to_1)

        # Flag used to keep track of playback direction
        self._is_reversed = False

    # .................................................................................................................

    def toggle_reverse_state(self, set_is_reversed: bool | None = None) -> bool:
        """
        Used to switch from forward-to-reverse frame reading
        If the given target state is None, then the current
        state will be toggled, otherwise it wil be set to
        the given state (True to reverse, False for forward reading).

        Returns: updated_reversal_state
        """

        self._is_reversed = (not self._is_reversed) if set_is_reversed is None else set_is_reversed

        return self._is_reversed

    # .................................................................................................................

    def get_reverse_state(self):
        """Used to check the current forward/reverse state"""
        return self._is_reversed

    # .................................................................................................................

    def __next__(self):
        """
        Iterator that provides frame data from a video capture object.
        Returns:
            is_paused, frame_index, frame_bgr
        """

        # Don't read video frames while paused
        if self._is_paused:
            return self._is_paused, self._frame_idx, self._pause_frame.copy()

        if self._is_reversed:

            # Repeatedly 'rewind' the video backwards and read frames, looping to end if needed
            self._frame_idx = (self._frame_idx - 1) % self._max_frame_idx
            self._vcap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_idx)
            read_ok, frame = self._vcap.read()
            while not read_ok:
                self._vcap.set(cv2.CAP_PROP_POS_FRAMES, self._max_frame_idx)
                read_ok, frame = self._vcap.read()
                if not read_ok:
                    self._max_frame_idx -= 1
                    print("Error reading last frame. Will try with reduced indexing:", self._max_frame_idx)
                self._frame_idx = self._max_frame_idx

        else:

            # Read next frame, or loop back to beginning if there are no more frames
            self._frame_idx += 1
            read_ok, frame = self._vcap.read()
            if not read_ok:
                self._frame_idx = 0
                self._vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                read_ok, frame = self._vcap.read()
                assert read_ok, "Error looping video! Unable to read first frame"

        # Scale frame for display & store in case we pause
        if self._need_resize:
            frame = self.scale_to_display_wh(frame)
        self._pause_frame = frame

        return self._is_paused, self._frame_idx, self._pause_frame.copy()

    # .................................................................................................................


class LoopingVideoPlaybackSlider(BaseCallback):
    """
    Implements a 'playback slider' UI element that is specific to working with videos.
    After initializing with a reference to the video reader whose playback is to be controlled,
    the playback slider only needs a single call (inside of the video loop) to work:
        slider.update(frame_index)

    This will update the slider position indicator (according to the given frame index) as well
    as internally keep track of changes to the slider (i.e. user adjustments).
    To check if the user is actively adjusting playback, use:
        slider.is_adjusting()
    """

    # .................................................................................................................

    def __init__(
        self,
        looping_video_reader: LoopingVideoReader,
        bar_height: int = 60,
        bar_bg_color=(40, 40, 40),
        indicator_line_width: int = 1,
        stay_paused_on_change=False,
    ):
        # Store reference to video capture
        self._vreader = looping_video_reader
        self._total_frames = looping_video_reader.total_frames
        self._reader_pause_state = looping_video_reader.get_pause_state()
        self._stay_paused_on_change = stay_paused_on_change

        # Storage for slider value
        self._max_frame_idx = int(self._total_frames) - 1
        self._slider_idx = looping_video_reader.get_playback_position(normalized=False)

        # Storage for slider state
        self._x_px = 0
        self._is_pressed = False
        self._is_changed = False

        # Display config
        self._bar_bg_color = bar_bg_color
        self._indicator_thickness = indicator_line_width
        self._base_image = blank_image(1, 1)

        # Inherit from parent & set default helper name for debugging
        super().__init__(bar_height, 128, expand_w=True)

    # .................................................................................................................

    def is_adjusting(self):
        return self._is_pressed

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-render label & marker lines onto new blank background if render size changes
        base_h, base_w = self._base_image.shape[0:2]
        if base_h != h or base_w != w:
            self._base_image = blank_image(h, w, self._bar_bg_color)

        # Draw indicator line
        img = self._base_image.copy()
        slider_norm = self._slider_idx / self._max_frame_idx
        line_x_px = round(slider_norm * (w - 1))
        img = cv2.line(img, (line_x_px, -1), (line_x_px, h + 1), (255, 255, 255), self._indicator_thickness)

        # Draw top/bottom divider lines (can't draw box, because it hides far left/right indicator line position!)
        img = cv2.line(img, (-1, 0), (w + 1, 0), (0, 0, 0), 1, cv2.LINE_4)
        img = cv2.line(img, (-1, h - 1), (w + 1, h - 1), (0, 0, 0), 1, cv2.LINE_4)
        return img

    # .................................................................................................................

    def update(self, frame_index):
        """Helper used to update playback position if the slider changes, otherwise updates the indicator line"""

        is_changed, new_frame_idx = self.read()
        if is_changed:
            self._vreader.set_playback_position(new_frame_idx)
        else:
            self._slider_idx = frame_index

        return self

    # .................................................................................................................

    def read(self) -> tuple[bool, float | int]:
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._slider_idx

    def set(self, slider_value, use_as_default_value=True):
        new_value = max(0, min(self._max_frame_idx, slider_value))
        if use_as_default_value:
            self._initial_position = new_value
        self._slider_idx = new_value
        return self

    def step_forward(self, num_increments=1):
        return self.set(self._slider_idx + num_increments, use_as_default_value=False)

    def step_backward(self, num_decrements=1):
        return self.set(self._slider_idx - num_decrements, use_as_default_value=False)

    # .................................................................................................................

    def on_left_down(self, cbxy, cbflags) -> None:

        # Ignore clicks outside of the slider
        if not cbxy.is_in_region:
            return

        # Prevent video playback while adjusting slider
        self._is_pressed = True
        self._reader_pause_state = self._vreader.get_pause_state()
        self._vreader.pause()

        # Record changes
        x_px = cbxy.xy_px[0]
        self._is_changed |= x_px != self._x_px
        if self._is_changed:
            self._x_px = x_px
            self._mouse_x_norm_to_slider_idx(cbxy.xy_norm[0])

        return

    def on_left_up(self, cbxy, cbflags) -> None:

        # Don't react to mouse up if we weren't being interacted with
        if not self._is_pressed:
            return
        self._is_pressed = False

        # Adjust pause state after user stops interacting with slider
        if self._stay_paused_on_change:
            self._reader_pause_state = self._vreader.pause()
        else:
            # Restore pause state (prior to modifying slider)
            # -> If the video was paused, this will keep it paused, otherwise unpause it
            self._reader_pause_state = self._vreader.pause(self._reader_pause_state)

        return

    def on_right_click(self, cbxy, cbflags) -> None:

        # Toggle pause when right clicked
        self._vreader.toggle_pause()

        return

    def on_drag(self, cbxy, cbflags) -> None:

        # Update slider value while dragging
        x_px = cbxy.xy_px[0]
        self._is_changed |= x_px != self._x_px
        if self._is_changed:
            self._x_px = x_px
            self._mouse_x_norm_to_slider_idx(cbxy.xy_norm[0])

        return

    # .................................................................................................................

    def _mouse_x_norm_to_slider_idx(self, x_norm: float) -> float | int:
        """Helper used to convert normalized mouse position into slider values"""

        # Map normalized x position to a frame index & make sure it's in the valid range
        frame_idx = round(x_norm * self._max_frame_idx)
        self._slider_idx = max(0, min(self._max_frame_idx, frame_idx))

        return self._slider_idx

    # .................................................................................................................


class ValueChangeTracker:
    """
    Simple helper object that be can be used to keep track of when a specific value of interest changes
    This is meant to be used to keep track of moments where potentially important state changes
    occur (e.g. pausing during video playback) and some important code may need to be run once
    in response to the change (but doesn't need to continue to run afterwards!)

    Main use is to check for changes using:
        value_changed = keeper.is_changed(curr_value, record_value=False)
    And then after responding to the change:
        if value_changed and some_other_condition:
            # do important/heavy work, then record value so we don't react to seeing it again
            keeper.record(curr_value)
    -> If the change will always be reacted to, it can be recorded during the .is_changed (...) check,
       which allows the later .record(...) to be left out!
    """

    # .................................................................................................................

    def __init__(self, initial_value=None):
        self._prev_value = initial_value

    def is_changed(self, value, record_value=False) -> bool:
        """Function used to check if the given value is different from the previously recorded value"""
        value_changed = value != self._prev_value
        if value_changed and record_value:
            self._prev_value = value
        return value_changed

    def record(self, value):
        """Records the given value for use in future 'did it change' checks"""
        self._prev_value = value
        return self

    def clear(self, clear_value=None):
        """Helper that is identical to 'record' but may be nicer to use to indicate intent!"""
        self._prev_value = clear_value

    # .................................................................................................................
