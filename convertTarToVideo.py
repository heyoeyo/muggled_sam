import os
import sys
import tarfile
import subprocess
import tempfile
from PIL import Image
import argparse

def extract_tar(tar_path, extract_path):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_path)

def convert_images_to_video(image_folder, output_video_path, framerate, ffmpeg_path='ffmpeg'):
    # Replace transparency with green screen
    for image_name in sorted(os.listdir(image_folder)):
        if image_name.lower().endswith('.png'):
            image_path = os.path.join(image_folder, image_name)
            img = Image.open(image_path).convert("RGBA")
            datas = img.getdata()
            newData = [(0,255,0,255) if pixel[3]==0 else pixel for pixel in datas]
            img.putdata(newData)
            img.save(image_path)

    # Use FFmpeg with sequential input
    subprocess.run([
        ffmpeg_path,
        '-framerate', str(framerate),
        '-i', os.path.join(image_folder, '%08d.png'),  # match 00000000.png, 00000001.png, etc.
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description='Convert TAR of images to video with green screen.')
    parser.add_argument('tar_path', type=str, help='Input TAR file')
    parser.add_argument('output_video_path', type=str, help='Output MP4 video')
    parser.add_argument('--framerate', type=int, default=30, help='Video frame rate')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='FFmpeg executable path')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        extract_tar(args.tar_path, temp_dir)
        convert_images_to_video(temp_dir, args.output_video_path, args.framerate, ffmpeg_path=args.ffmpeg)

if __name__ == '__main__':
    main()
