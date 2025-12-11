# Videos from segmented frames

The [run_video](https://github.com/heyoeyo/muggled_sam?tab=readme-ov-file#run-video-or-webcam) script saves each frame as a transparent .png file in order to maintain lossless segmentation of the original video, as well as supporting potential jumps in the video timeline. While this perserves as much data as possible, it may not be as convenient as a video file. Luckily, it's fairly easy to make a video from these frames, as explained below.


## Direct from .tar

If you don't need to check or edit the .pngs, it's possible to create a video directly from the .tar file using [FFmpeg](https://www.ffmpeg.org/). This can be done as follows:

```bash
# Unpack .tar directly into ffmpeg
tar -xOf name_of_saved_frames.tar | \
  ffmpeg -i - -framerate 30 -vcodec ffv1 output.mkv
```

Just be sure to edit the `name_of_saved_frames.tar` to point at the saved frame data, and adjust the `framerate` as needed.


## Unpack & Encode

As an alternative, a video can be created from the saved frame data by first unpacking the .tar file and then combining the results using FFmpeg.

### Untar pngs

It's recommended to unpack the .tar file using a file explorer, so you can confirm you're working with the correct files, but it can also be done in the terminal:

```bash
# Create 'pngs_folder' folder and unpack frames into it
mkdir pngs_folder
tar -xf name_of_saved_frames.tar -C pngs_folder
```

At this point, you can 'edit' which frames end up in the resulting video by moving/deleting them from the folder. This may be useful if, for example, the original segmented object went in-and-out of view and you'd like to save separate videos for each appearance.


### Combine pngs into video

With the .pngs unpacked, they can be combined into a video using FFmpeg. There are an enormous number of options when it comes to using FFmpeg, but a basic example for merging frames into a lossless video (using the [ffv1](https://trac.ffmpeg.org/wiki/Encode/FFV1) codec) is:

```bash
# Create (lossless) video from pngs in 'pngs_folder' folder
ffmpeg -i './pngs_folder/%08d.png' -framerate 30 -vcodec ffv1 output.mkv
```

To save the video using a more conventional codec (h264), try the following:

```bash
# Create video with h264 codec
ffmpeg -i './pngs_folder/%08d.png' \
  -framerate 30 \
  -vcodec libx264 \
  -crf 20 \
  output.mp4
```

The `crf` value controls the amount of compression. Higher values lead to lower video quality, but smaller file sizes. Note that compression may introduce masking errors if you're planning to re-segment the video later!

## Background Color

Videos (generally) don't support transparency, so when creating a video from .pngs, the underlying color of transparent areas will be used instead.
This coloring can be adjusted by using the `--background_color` flag. For example, to produce a [green-screen](https://en.wikipedia.org/wiki/Chroma_key) effect, use:
```bash
python run_video.py --background_color 0,255,0
```
Choosing a color that doesn't appear in the original video will make it easy to re-segment the video inside of video editing software.