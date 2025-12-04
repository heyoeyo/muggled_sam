#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from time import perf_counter
import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class DisplayWindow:
    """Class used to manage opencv window, mostly to make trackbars & callbacks easier to organize"""

    WINDOW_CLOSE_KEYS_SET = {ord("q"), 27}  # q, esc

    def __init__(self, window_title, display_fps=60, limit_threading=True):

        # Clear any existing window with the same title
        # -> This forces the window to 'pop-up' when initialized, in case a 'dead' window was still around
        try:
            cv2.destroyWindow(window_title)
        except cv2.error:
            pass

        if limit_threading:
            self.limit_threading()

        # Store window state
        self.title = window_title
        self._frame_delay_ms = 1000 // display_fps
        self._last_display_ms = -self._frame_delay_ms

        # Allocate variables for use of callbacks
        self._mouse_cbs = CallbackSequencer()
        self._using_mouse_cbs = False
        self._keypress_callbacks_dict: dict[int, callable] = {}

        # Fill in blank image to begin (otherwise errors before first image can cause UI to freeze!)
        cv2.namedWindow(self.title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        self.show(np.zeros((50, 50, 3), dtype=np.uint8), 100)

    def limit_threading(self, thread_limit=1):
        """
        Helper used to reduce opencv (often excessive) thread usage
        Note: this is a global setting, and may negatively affect other opencv functionality!
        """
        cv2.setNumThreads(thread_limit)
        return self

    def move(self, x, y):
        cv2.moveWindow(self.title, x, y)
        return self

    def add_trackbar(self, trackbar_name, max_value, initial_value=0):
        return WindowTrackbar(self.title, trackbar_name, max_value, initial_value)

    def attach_mouse_callbacks(self, *callbacks):
        """
        Attach callbacks for handling mouse events
        Callback functions should have a call signature as folows:

            def callback(event: int, x: int, y: int, flags: int, params: Any) -> None:

                # Example to handle left-button down event
                if event == EVENT_LBUTTONDOWN:
                    print("Mouse xy:", x, y)

                return
        """

        # Record all the given callbacks
        self._mouse_cbs.add(*callbacks)

        # Attach callbacks to window for the first time, if needed
        if not self._using_mouse_cbs:
            cv2.setMouseCallback(self.title, self._mouse_cbs)
            self._using_mouse_cbs = True

        return self

    def attach_keypress_callback(self, keycode: int | str, callback):
        """
        Attach a callback for handling a keypress event
        Keycodes can be given as strings (i.e. the actual key, like 'a') or for
        keys that don't have simple string representations (e.g. the Enter key),
        the raw keycode integer can be given. To figure out what these are,
        print out the window keypress result while pressing the desired key!

        Callbacks should have no input arguments and no return values!
        """
        if isinstance(keycode, str):
            keycode = ord(keycode.lower())
        self._keypress_callbacks_dict[keycode] = callback
        return self

    def show(self, image, frame_delay_ms=None) -> [bool, int]:
        """
        Function which combines both opencv functions: 'imshow' and 'waitKey'
        This is meant as a convenience function in cases where only a single window is being displayed.
        If more than one window is displayed, it is better to use 'imshow' and 'waitKey' separately,
        so that 'waitKey' is only called once!
        Returns:
            request_close, keypress
        """

        # Figure out frame delay (to achieve target FPS) if we're not given one
        if frame_delay_ms is None:
            curr_ms = int(1000 * perf_counter())
            time_elapsed_ms = curr_ms - self._last_display_ms
            frame_delay_ms = max(self._frame_delay_ms - time_elapsed_ms, 1)

        cv2.imshow(self.title, image)
        keypress = cv2.waitKey(int(frame_delay_ms)) & 0xFF
        request_close = keypress in self.WINDOW_CLOSE_KEYS_SET
        self._last_display_ms = int(1000 * perf_counter())

        # Run keypress callbacks
        for cb_keycode, cb in self._keypress_callbacks_dict.items():
            if keypress == cb_keycode:
                cb()

        return request_close, keypress

    def imshow(self, image):
        """Wrapper around opencv imshow, fills in 'winname' with the window title"""
        cv2.imshow(self.title, image)
        return self

    @classmethod
    def waitKey(cls, frame_delay_ms=1) -> [bool, int]:
        """
        Wrapper around opencv waitkey (triggers draw to screen)
        Returns:
            request_close, keypress
        """

        keypress = cv2.waitKey(int(frame_delay_ms)) & 0xFF
        request_close = keypress in cls.WINDOW_CLOSE_KEYS_SET
        return request_close, keypress

    def close(self):
        return cv2.destroyWindow(self.title)

    @staticmethod
    def close_all(self):
        cv2.destroyAllWindows()


class WindowTrackbar:
    """Class used to keep track of strings that opencv uses to reference trackbars on windows"""

    def __init__(self, window_name, trackbar_name, max_value, initial_value=0):

        self.name = trackbar_name
        self._window_name = window_name
        self._prev_value = int(initial_value)
        cv2.createTrackbar(trackbar_name, window_name, int(initial_value), int(max_value), lambda x: None)
        self._max_value = max_value
        self._read_lamda = lambda x: x

    def read(self):
        raw_value = cv2.getTrackbarPos(self.name, self._window_name)
        return self._read_lamda(raw_value)

    def write(self, new_value):
        safe_value = max(0, min(new_value, self._max_value))
        return cv2.setTrackbarPos(self.name, self._window_name, safe_value)

    def set_read_lambda(self, read_lambda):
        """
        Function which allows for setting function which is applied when reading
        values from the trackbar and can be used to map raw trackbar values to
        some other value range (including converting to different data types!)

        An example of a read lambda which divides the raw value by 100:
            read_lambda = lambda raw_value: raw_value/100
        """
        assert callable(read_lambda), "Must provide a 'read_lamda' which is function taking a single integer argument"
        try:
            read_lambda(0)
        except TypeError:
            raise TypeError("Window trackbar 'read_lambda' must take in only a single argument!")
        self._read_lamda = read_lambda
        return self


class CallbackSequencer:
    """
    Simple wrapper used to execute more than one callback on a single opencv window

    Example usage:

        # Set up window that will hold callbacks
        winname = "Display"
        cv2.namedWindow(winname)

        # Create multiple callbacks and combine into sequence so they can both be added to the window
        cb_1 = MakeCB(...)
        cb_2 = MakeCB(...)
        cb_seq = CallbackSequence(cb_1, cb_2)
        cv2.setMouseCallback(winname, cb_seq)
    """

    def __init__(self, *callbacks):
        self._callbacks = [cb for cb in callbacks]

    def add(self, *callbacks):
        self._callbacks.extend(callbacks)

    def __call__(self, event, x, y, flags, param) -> None:
        for cb in self._callbacks:
            cb(event, x, y, flags, param)
        return

    def __getitem__(self, index):
        return self._callbacks[index]

    def __iter__(self):
        yield from self._callbacks


# ---------------------------------------------------------------------------------------------------------------------
# %% Define window key codes


class KEY:

    LEFT_ARROW = 81
    UP_ARROW = 82
    RIGHT_ARROW = 83
    DOWN_ARROW = 84

    ESC = 27
    ENTER = 13
    BACKSPACE = 8
    SPACEBAR = ord(" ")
    TAB = ord("\t")

    SHIFT = 225
    ALT = 233
    CAPSLOCK = 229
    # CTRL = None # No key code for this one surprisingly!?
