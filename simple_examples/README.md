# MuggledSAM - Simple Examples

This folder contains several scripts that contain minimal/simplified examples of how the SAM models can be used for various tasks. They are provided as an alternative to the UI scripts, which tend to hide the underlying model usage due to all the extra UI code!

> [!Note]
> All of these scripts require modifying hard-coded pathing (usually near the top of the script) to point at files/models to load for the example.


## Auto Mask Generator

_(Supports SAMv1, SAMv2, SAMv3)_

This script runs the SAM model as an 'auto-mask generator', similar to the capability provided by the [original repo implementation](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/automatic_mask_generator.py#L36). It works by running the SAM model with a (dense) grid of single point prompts to generate masks from all parts of the image, while filtering bad/overlapping results. This version provides a visualization (if enabled in the settings) of the results as they're being generated.


## Image Segmentation

_(Supports SAMv1, SAMv2, SAMv3)_

This script contains the most basic usage of the SAM models, which is to segment an image based on a set of provided prompts (bounding boxes, foreground points, background points or masks).

## Image Segmentation with Batches

_(Supports SAMv1, SAMv2, SAMv3)_

This is an extension of the basic image segmentation script, modified to show how an image batch can be processed. Processing a batch of images is the same as processing multiple images, one after another, except batching can be slightly faster when using a GPU. For simplicity, this example just repeats a single image to form a batch, but normally many images would be loaded in and processed together.


## Object Detection

_(Supports SAMv3)_

This script contains the most basic usage of the SAMv3 detection functionality. It allows multiple objects to be detected using a single text prompt or by specifying a part of the image (using points or bounding boxes) as a reference for what to detect.


## Speed Benchmarking

_(Supports SAMv1, SAMv2, SAMv3)_

This script runs each of the image segmentation components repeatedly while timing the average execution speed. It can be used to get a sense of how fast each of the different model variants will run and supports changing the image size used by the model. For example, here are some examples results (using an RTX 3090) for SAMv1, v2 & v3 models at different input image sizes, using bfloat16 & square image sizing:

| Model | Encoding @ 1024px | Encoding @ 512px | Mask Generation |
| ----- | ----------------- | ---------------- | --------------- |
| V1-Base   | 44 ms  | 10 ms  | 1.5 ms |
| V1-Large  | 101 ms | 26 ms  | 1.5 ms |
| V2-Tiny   | 10 ms  | 3.1 ms | 1.6 ms |
| V2-Large  | 47 ms  | 13 ms  | 1.6 ms |
| V3        | 148ms  | 39 ms  | 1.5 ms |

| Model | Encoding @ 1008px | Encoding @ 504px | Mask Generation |
| ----- | ----------------- | ---------------- | --------------- |
| V3        | 109ms  | 41 ms  | 1.5 ms |

The SAMv3 results are shown for both 1024px and 1008px (it's native default), since the model takes a _significant_ performance hit at the v1/v2 default 1024px sizing. Strangely, it's also consistently slower at 504px vs. 512px.

The script also prints out an estimate of VRAM usage (if using cuda):

| Model | VRAM @ 1024px | VRAM @ 512px |
| ----- | ------------- | ------------ |
| V1-Base | 1.7 GB | 0.4 GB |
| V1-Large | 2.6 GB | 0.9 GB |
| V2-Tiny | 0.5 GB | 0.3 GB  |
| V2-Large | 1.0 GB | 0.7 GB |
| V3 | 1.5 GB | 1.2 GB |


## Video Segmentation

_(Supports SAMv2, SAMv3)_

This script provides a basic example of how to implement video segmentation using the SAMv2 or v3 model (V1 does not support video segmentation!). For simplicity, this script assumes that tracking begins with a prompt on the first frame of the video, but any frame could be used. It also only stores results for one object, though again this can be changed to handle multiple objects by creating instances of the video storage data for each object.

Segmentation results are displayed per-frame for verification, and the total inference time is also printed out. For example, the table below shows the per-frame inference times for different models while tracking one object at different image sizes using bfloat16 (all other settings are left at defaults):

| Model | Inference @ 1024 | Inference @ 512 |
| ----- | ---------------- | --------------- |
| V2-Tiny | 26 ms | 8 ms |
| V2-Small | 28 ms | 8 ms |
| V2-Base | 38 ms | 11 ms |
| V2-Large | 64 ms | 17 ms |
| V3 | 172ms | 46ms |

| Model | Inference @ 1008 | Inference @ 504 |
| ----- | ---------------- | --------------- |
| V3 | 130ms | 47ms |

Again, the results for the v3 model using 1008px are included as it runs much slower when using the v1/v2 default sizing.

## Video Segmentation (from mask)

_(Supports SAMv2, SAMv3)_

This is a variation of the basic video segmentation example, but uses a mask prompt to begin tracking. Both a binary mask and corresponding image (i.e. the RGB image from which the mask was generated) must be provided, and replaces the need to provide box or point prompts. Note that the mask & corresponding image don't need to be from the video! Mixing the mask image and video can give results similar to the [video with image priors](https://github.com/heyoeyo/muggled_sam/tree/main/experiments#video-with-image-priors) experimental script.

## Video Segmentation (Multi-object)

_(Supports SAMv2, SAMv3)_

This is an extension of the more basic video segmentation script, which shows how multiple objects can be segmented/tracked through a video. For the sake of demonstration, this example works by having all prompts known ahead of time, but of course they could be generated dynamically (e.g. by an object detection model). No results are saved from this script, but the combined mask results are displayed and the corresponding code for generating the display output can hopefully provide ideas about how to handle the model outputs for other use cases.

The prompts that are hard-coded into this example script are set up to track a few horses from a short video by [Adrian Hoparda](https://www.pexels.com/@adrian-hoparda-1684220/) which can be freely downloaded:
https://www.pexels.com/video/horses-running-on-grassland-4215784/

## Video Segmentation using SAMURAI

_(Supports SAMv2, SAMv3)_

This variation of the video segmentation script uses an alternative method of selecting masks during tracking based on the paper: "[SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory](https://arxiv.org/abs/2411.11922)". The idea is to independently track object bounding boxes using a separate tracking method (a [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) in this case) and use this to select which masks should be propagated during tracking (as opposed to just using the SAM model IoU predictions). The implementation here is more similar to the description in the paper itself, rather than the [available code](https://github.com/yangchris11/samurai/blob/master/sam2/sam2/utils/kalman_filter.py), but should be [easy to modify](https://github.com/heyoeyo/muggled_sam/blob/3ed04b646005d1b1242b8d07008573ef00815405/muggled_sam/demo_helpers/samurai.py#L22) if needed.

This demo is set up to track only one object, but can be changed to handle multiple objects by creating additional prompt/prev. frame memory storage as well as instances of the SAMURAI class for each object.

## Video Segmentation from Detections

_(Supports SAMv3)_

This script uses the new detection capabilities of SAMv3 in order to generate initial objects for tracking. It's a bit like a combination of the tracking from masks & multi-object examples above, but ends up being a fully automatic way of doing video segmentation.

Very basic support for repeat detections (every 'N' frames) is included so that newly appearing objects are tracked over time. The logic for this is overly simple, but is hopefully a useful starting point. A visualization can be enabled which shows existing objects (magenta boxes), new detections (green boxes) and detections that overlap existing objects (yellow boxes) to help provide some intuition about what's going on.

For fun, the script also supports using different image encoding configurations for tracking vs. detections (see `track_imgenc_config_dict`). Reducing the `max_side_length` setting (e.g. set to `512`) on the tracking config can be used to speed up tracking significantly at the cost of segmentation quality. It also supports using a SAMv2 model for tracking, though the v3 model is still required for detections.
