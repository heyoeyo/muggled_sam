#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import os.path as osp
import subprocess
import tempfile
import shutil


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def get_default_ffmpeg_command():
    """Helper used to guess at the default name for an ffmpeg executable"""
    return "ffmpeg.exe" if os.name == "nt" else "ffmpeg"


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
    full_ffmpeg_path = shutil.which(ffmpeg_executable_path)
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


def render_png_dict_to_video(
    save_folder: str, save_index: str, object_index: int, png_dict: dict, ffmpeg_bin: str, fps: float | None = None
) -> str | None:
    """Render PNG frames (from `png_dict`) into a video using the provided ffmpeg binary.

    The function writes the PNGs to a temporary folder (inside `save_folder`) named sequentially
    and invokes ffmpeg to encode them to an MP4 placed alongside other saved results.
    Returns the output video path on success, or None on failure.
    """

    if ffmpeg_bin is None:
        return None

    # Create a temporary directory to hold sequentially named PNGs
    tmpdir = tempfile.mkdtemp(prefix=f"{save_index}_obj{1+object_index}_", dir=save_folder)
    try:
        # Sort frame indices and write files sequentially starting at 0
        frame_idxs = sorted(png_dict.keys())
        if len(frame_idxs) == 0:
            return None

        min_idx = frame_idxs[0]
        max_idx = frame_idxs[-1]
        for seq_idx, frame_idx in enumerate(frame_idxs):
            png_enc = png_dict[frame_idx]
            out_name = osp.join(tmpdir, f"{seq_idx:0>8}.png")
            with open(out_name, "wb") as fh:
                fh.write(png_enc.tobytes())

        # Determine framerate
        frm = int(fps) if (fps is not None and fps > 0) else 30

        out_video_name = f"{save_index}_obj{1+object_index}_{min_idx}_to_{max_idx}.mp4"
        out_video_path = osp.join(save_folder, out_video_name)

        cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(frm),
            "-i",
            osp.join(tmpdir, "%08d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            out_video_path,
        ]

        try:
            subprocess.run(cmd, check=True)
            return out_video_path
        except Exception as e:
            print(f"Warning: ffmpeg render failed: {e}")
            return None

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
