# Saving Video Segmentation Results

By default, the [run_video](https://github.com/heyoeyo/muggled_sam?tab=readme-ov-file#run-video-or-webcam) script saves each frame as a transparent .png file in order to maintain lossless segmentation of the original video, as well as supporting potential jumps in the video timeline. While this perserves as much data as possible, it may not be as convenient as a video file. Luckily, it's fairly easy to make a video from these frames using [FFmpeg](https://www.ffmpeg.org/) as explained below.

Note that FFmpeg does not come included with this repo, it must be [installed separately](https://www.ffmpeg.org/download.html).

## Option 1 - Direct from tarfile

One approach, if a tarfile is saved (the default behavior), is to create a video directly from the tarfile after-the-fact using FFmpeg. This can be done as follows:

```bash
# Extract tarfile directly into ffmpeg
tar -xOf name_of_saved_frames.tar | \
  ffmpeg -f image2pipe -r 30.0 -i - -vcodec ffv1 output.mkv
```

Just be sure to edit the `name_of_saved_frames.tar` to point at the saved frame data, and adjust the framerate (the `-r` flag) as needed. This command uses the [ffv1](https://trac.ffmpeg.org/wiki/Encode/FFV1) codec which saves videos in a lossless format. If the `ffv1` codec isn't available or you'd prefer a smaller file size, consider using `libx264`.

This approach provides a lot of control and the ability to test different encodings, but you'll end up with both a tarfile and a video file.


## Option 2 - Save as video

Another approach is to save a video directly from the `run_video` script using the `--ffmpeg` flag:

```bash
python run_video.py --ffmpeg
```

If FFmpeg is not available system wide, a path to the executable can (optionally) be provided with the `--ffmpeg` flag:

```bash
python run_video.py --ffmpeg 'C:\path\to\ffmpeg.exe'
```

With this approach, you'll only end up with a video file! The downside of this approach is that the encoding settings cannot be changed. For compatbility, this approach saves an mp4 video encoded with `libx264` using all default settings.


## Option 3 - Extract & Encode

This last approach provides the most control but is the most involved. The idea is to save results as a tarfile, then extract the .pngs and combine them into a video using FFmpeg, potentially modifying the frames prior to encoding the video.

### Extract pngs

It's recommended that the tarfile be extracted using a file explorer so you can confirm you're working with the correct files, but it can also be done in the terminal:

```bash
# Create 'pngs_folder' folder and unpack frames into it
mkdir pngs_folder
tar -xf name_of_saved_frames.tar -C pngs_folder
```

At this point, you can change which frames end up in the resulting video by moving/deleting them from the folder. The frames can also be directly edited of course. This may be useful if, for example, the original segmented object went in-and-out of view and you'd like to save separate videos for each appearance. Or if you'd like to merge masks of multiple objects into a single frame, this would be the best place to do so.


### Combine pngs into video

With the .pngs extracted (and possibly edited) they can now be encoded into a video. There are many encoding options, for example, the lossless encoding described earlier:

```bash
# Create (lossless) 30.0 fps video from pngs in 'pngs_folder' folder
ffmpeg -r 30.0 -i './pngs_folder/%08d.png' -vcodec ffv1 output.mkv
```

To save the video using a more conventional codec (h264), try the following:

```bash
# Create video with h264 codec
ffmpeg -r 30.0 \
  -i './pngs_folder/%08d.png' \
  -vcodec libx264 \
  -crf 20 \
  output.mp4
```

The `-r` flag controls the framerate while `-crf` controls the amount of compression. Higher values lead to lower video quality, but smaller file sizes.


## Background Color

When encoding videos in a lossy format (e.g. h264), masked areas will usually not be transparent. Instead, they take on the color of the underlying masked region. The `run_video` script defaults to using (transparent) magenta to avoid overlap with the video content, but this color can be changed using the `--bg_color_hex` flag. For example, to produce a [green-screen](https://en.wikipedia.org/wiki/Chroma_key) effect, use:
```bash
python run_video.py --bg_color_hex 00ff00
```

The color is meant to be specified as a [hexidecimal value](https://www.color-hex.com/) and can be either 2 digits (`00` is black, `ff` is white), 6 digits (RRGGBB) or 8 digits (RRGGBBAA). When using 2 or 6 digits, the color is assumed to be fully opaque (e.g. no transparency). Choosing a color that doesn't appear in the original video will make it easier to re-segment inside of other software.