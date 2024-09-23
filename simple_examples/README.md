# Muggled SAM - Simple Examples

This folder contains several scripts that contain minimal/simplified examples of how the SAM models can be used for various tasks. They are provided as an alternative to the UI scripts, which tend to hide the underlying model usage due to all the extra UI code!

> [!Note]
> All of these scripts require modifying hard-coded pathing (usually near the top of the script) to point at files/models to load for the example.


## Image Segmentation

This script contains the most basic usage of the SAM models, which is to segment an image based on a set of provided prompts (bounding boxes, foreground points, background points or masks).

## Image Segmentation with Batches

This is an extension of the basic image segmentation script, modified to show how an image batch can be processed. Processing a batch of images is the same as processing multiple images, one after another, except batching can be slightly faster when using a GPU. For simplicity, this example just repeats a single image to form a batch, but normally many images would be loaded in and processed together.


## Speed Benchmarking

This script runs each of the image segmentation components repeatedly while timing the average execution speed. It can be used to get a sense of how fast each of the different model variants will run and supports changing the image size used by the model. For example, here are some examples results (using an RTX 3090) for SAMv1 & V2 models at different input image sizes, using bfloat16 & square image sizing:

| Model | Encoding @ 1024px | Encoding @ 512px | Mask Generation |
| ----- | ----------------- | ---------------- | --------------- |
| V1-Base | 44 ms | 10 ms | 1.5 ms | 
| V1-Large  | 101 ms | 26 ms | 1.5 ms |
| V2-Tiny  | 10 ms | 3.1 ms | 1.6 ms |
| V2-Large  | 47 ms | 13 ms | 1.6 ms |

The script also prints out an estimate of VRAM usage (if using cuda):

| Model | VRAM @ 1024px | VRAM @ 512px |
| ----- | ------------- | ------------ |
| V1-Base | 2.4 GB | 1.1 GB |
| V1-Large | 3.3 GB | 1.6 GB |
| V2-Tiny | 1.1 GB | 0.9 GB  |
| V2-Large | 1.7 GB | 1.4 GB |



## Video Segmentation

This script provides a basic example of how to implement video segmentation using the SAMv2 model (V1 does not support video segmentation!). For simplicity, this script assumes that tracking begins with a prompt on the first frame of the video, but any frame could be used. It also only stores results for one object, though again this can be changed to handle multiple objects by creating instances of the video storage data for each object.

Segmentation results are displayed per-frame for verification, and the total inference time is also printed out. For example, the table below shows the per-frame inference times for different models while tracking one object at different image sizes using bfloat16 (all other settings are left at defaults):

| Model | Inference @ 1024 | Inference @ 512 |
| ----- | ---------------- | --------------- |
| V2-Tiny | 26 ms | 8 ms |
| V2-Small | 28 ms | 8 ms |
| V2-Base | 38 ms | 11 ms |
| V2-Large | 64 ms | 17 ms |


## Video Segmentation (Multi-object)

This is an extension of the more basic video segmentation script, which shows how multiple objects can be segmented/tracked through a video. For the sake of demonstration, this example works by having all prompts known ahead of time, but of course they could be generated dynamically (e.g. by an object detection model). No results are saved from this script, but the combined mask results are displayed and the corresponding code for generating the display output can hopefully provide ideas about how to handle the model outputs for other use cases.

The prompts that are hard-coded into this example script are set up to track a few horses from a short video by [Adrian Hoparda](https://www.pexels.com/@adrian-hoparda-1684220/) which can be freely downloaded:
https://www.pexels.com/video/horses-running-on-grassland-4215784/