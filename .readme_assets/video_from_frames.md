# Saving Video Segmentation Results

## PNG Frames (TAR Archive) — Default

By default, "Save Buffer" saves segmented frames as a TAR archive of PNG images with transparency and optional background color.

```bash
python run_video.py -i video.mp4
```

Output: `saved_images/run_video/{video_name}/001_obj0_100_to_200_frames.tar`

Extract with:
```bash
tar -xf 001_obj0_100_to_200_frames.tar
```

## MP4 Video (FFmpeg)

To render frames directly to MP4, use the `--ffmpeg` flag.

### Installation

- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`

### Usage

```bash
# Use ffmpeg from PATH
python run_video.py -i video.mp4 --ffmpeg

# Use specific ffmpeg executable
python run_video.py -i video.mp4 --ffmpeg "C:\path\to\ffmpeg.exe"
```

Output: `saved_images/run_video/{video_name}/001_obj0_100_to_200.mp4`

**Encoding:** H.264 (libx264), YUV 4:2:0, framerate from source video

## Background Color

Customize background for both outputs (useful for green-screen):

```bash
python run_video.py -i video.mp4 --background_color "0,255,0"        # RGB green
python run_video.py -i video.mp4 --background_color "0,255,0,128"    # RGBA green, 50% opacity
```

Works with both TAR and MP4 outputs.

## Troubleshooting

**FFmpeg not found:**
```
Warning: --ffmpeg specified but 'ffmpeg' was not found on PATH. Video rendering will be disabled.
```

Solution: Install FFmpeg and add to PATH, or provide full path to executable.
