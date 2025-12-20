#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import subprocess
from shutil import which


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def get_default_ffmpeg_command():
    """Helper used to guess at the default name for an ffmpeg executable"""
    return "ffmpeg.exe" if os.name == "nt" else "ffmpeg"


# .....................................................................................................................


def verify_ffmpeg_path(ffmpeg_executable_path: str | None = None) -> tuple[bool, str | None]:
    """
    Helper used to check that a given path to ffmpeg is valid.
    Returns:
        is_ffmpeg_path_valid, full_ffmpeg_path

    If given 'None' as an input, will return: (False, None)
    If given a valid path, returns:           (True, /path/to/ffmpeg)
    If given a valid path that doesn't seem to be ffmpeg, will raise errors
    """

    # Handle missing input
    ok_path, full_ffmpeg_path = False, None
    if ffmpeg_executable_path is None:
        return ok_path, full_ffmpeg_path

    # Bail if the given path isn't valid
    full_ffmpeg_path = which(ffmpeg_executable_path)
    ok_path = full_ffmpeg_path is not None
    if not ok_path:
        raise FileNotFoundError(f"Invalid FFmpeg path: {ffmpeg_executable_path}")

    # Try to run 'ffmpeg --help' to make sure it works
    # (i.e. we weren't given a path to a valid but non-executable file/folder)
    try:
        subprocess.run([full_ffmpeg_path, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        ok_path = True

    except PermissionError:
        # This seems to happen when given non-executable files
        print("", "Bad FFmpeg path:", "@ {full_ffmpeg_path}", "-> Must provide an executable", sep="\n", flush=True)
        raise SystemExit("Invalid FFmpeg path")

    except subprocess.CalledProcessError:
        print("", "Unable to run FFmpeg:", "@ {full_ffmpeg_path}", sep="\n", flush=True)
        raise SystemExit("Unable to run FFmpeg")

    return ok_path, full_ffmpeg_path


# .....................................................................................................................


def save_video_stream(
    ffmpeg_path: str,
    save_path_no_ext: str,
    video_fps: float,
    save_frames_dict: dict,
    print_progress_indicator: bool = True,
) -> tuple[bool, str]:
    """
    Function used to save a video by 'streaming' image data to ffmpeg,
    this does not involve any disk io!
    Returns:
        ok_save, save_path
    """

    # Set up ffmpeg commands
    save_path = f"{save_path_no_ext}.mp4"
    ffmpeg_config = f"-r {video_fps} -f image2pipe -i - -vcodec libx264"
    ffmpeg_command_list = [ffmpeg_path, *ffmpeg_config.split(" "), save_path]  # Using split on paths isn't safe!
    io_kwargs = {"stdin": subprocess.PIPE, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}

    # Helper for printing out progress
    prog_print = lambda message, end="": print(message, end=end, flush=True) if print_progress_indicator else None

    ok_save = False
    try:
        frame_idxs = sorted(save_frames_dict.keys())
        total_frames = max(len(frame_idxs), 1)
        num_progress_print_total, num_progress_printed = 10, 0
        prog_print("0%[")
        with subprocess.Popen(ffmpeg_command_list, **io_kwargs) as ffmpeg_proc:
            for list_idx, fidx in enumerate(frame_idxs):
                png_encoding = save_frames_dict[fidx]
                ffmpeg_proc.stdin.write(png_encoding.tobytes())

                # Update progress indicator
                progress_dots = int(((list_idx + 1) / total_frames) * num_progress_print_total)
                num_dots_to_print = progress_dots - num_progress_printed
                if num_dots_to_print > 0:
                    prog_print("|" * num_dots_to_print)
                    num_progress_printed += num_dots_to_print

            # Signal end of video so ffmpeg knows we're done
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
            prog_print("]100%", end="\n")
        ok_save = True

    except Exception as err:
        print("Error encoding video with FFmpeg:", err, sep="\n")

    return ok_save, save_path
