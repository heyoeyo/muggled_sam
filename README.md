# Muggled SAM

This repo contains a simplified implementation of the awesome 'Segment Anything' models from [facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2) (and [SAMV1](https://github.com/facebookresearch/segment-anything)), with the intention of [removing the magic](https://en.wikipedia.org/wiki/Muggle) from the original code base to make it easier to understand. Most of the changes come from separating/simplifying the different components of the model structure.

<p align="center">
  <img src=".readme_assets/demo_anim.gif">
</p>

While the focus of this implementation is on interactivity and readability of the code, this implementation provides support for arbitrary input resolutions, which can improve performance in some cases.

> [!Note]
> This repo is a (messy) work-in-progress! The end goal is to have something resembling [MuggledDPT](https://github.com/heyoeyo/muggled_dpt).

## Getting started

This repo includes two demo scripts, [run_image.py](https://github.com/heyoeyo/muggled_sam/blob/main/run_image.py) and [run_video.py](https://github.com/heyoeyo/muggled_sam/blob/main/run_video.py) (along with a number of [simple examples](https://github.com/heyoeyo/muggled_sam/tree/main/simple_examples)). To use these scripts, you'll first need to have [Python](https://www.python.org/) (v3.10+) installed, then set up a virtual environment and install some additional requirements.

### Install
First create and activate a virtual environment (do this inside the repo folder after [cloning/downloading](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) it):
```bash
# For linux or mac:
python3 -m venv .env
source .env/bin/activate

# For windows (cmd):
python -m venv .env
.env\Scripts\activate.bat
```

Then install the requirements (or you could install them manually from the [requirements.txt](https://github.com/heyoeyo/muggled_sam/blob/main/requirements.txt) file):
```bash
pip install -r requirements.txt
```

<details>
<summary>Additional info for GPU usage</summary>

If you're using Windows and want to use an Nvidia GPU or if you're on Linux and don't have a GPU, you'll need to use a slightly different install command to make use of your hardware setup. You can use the [Pytorch installer guide](https://pytorch.org/get-started/locally/) to figure out the command to use. For example, for GPU use on Windows it may look something like:
```bash
pip3 uninstall torch  # <-- Do this first if you already installed from the requirements.txt file
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

**Note**: With the Windows install as-is, you may get an error about a `missing c10.dll` dependency. Downloading and installing this [mysterious .exe file](https://aka.ms/vs/16/release/vc_redist.x64.exe) seems to fix the problem.

</details>



### Model Weights

Before you can run a model, you'll need to download it's weights. There are 3 officially supported SAMv1 models (vit-base, vit-large and vit-huge) and four v2 models (tiny, small, base-plus and large). This repo uses the exact same weights as the original implementations (or any fine-tuned variant of the original models), which can be downloaded from the **Download Checkpoints** section of [SAMv2 repo](https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints) and the **Model Checkpoints** section of the [SAMv1 repo](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints).

After downloading a model file, you can place it in the `model_weights` folder of this repo or otherwise just keep note of the file path, since you'll need to provide this when running the demo scripts. If you do place the file in the [model_weights](https://github.com/heyoeyo/muggled_sam/tree/main/model_weights) folder, then it will auto-load when running the scripts.

<details>

<summary>Direct download links</summary>

The tables below include direct download links to all of the supported models. **Note:** These are all links to the original repos, none of these files belong to MuggledSAM!

| SAMv2 Models | Size (MB) |
| -----| -----|
| [sam2_hiera_tiny](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt) | 160 |
| [sam2_hiera_small](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt) | 185 |
| [sam2_hiera_base_plus](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt) | 325 |
| [sam2_hiera_large](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt) | 900 |

| SAMv1 Models | Size (MB) |
| -----| -----|
| [sam-vit-base](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | 375 |
| [sam-vit-large](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) | 1250 |
| [sam-vit-huge](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | 2560 |

</details>

### Simple Example
Here is an [example](https://github.com/heyoeyo/muggled_sam/tree/main/simple_examples/image_segmentation.py) of using the model to generate masks from an image:
```python
import cv2
from lib.make_sam import make_sam_from_state_dict

# Define prompts using 0-to-1 xy coordinates
box_tlbrs = []  # Example [((0.25, 0.25), (0.75, 0.75))]
fg_xys = [(0.5, 0.5)]
bg_xys = []

# Load image & model
image_bgr = cv2.imread("/path/to/image.jpg")
_, model = make_sam_from_state_dict("/path/to/model.pth")

# Process data
encoded_img, _, _ = model.encode_image(image_bgr)
encoded_prompts = model.encode_prompts(box_tlbrs, fg_xys, bg_xys)
mask_preds, iou_preds = model.generate_masks(encoded_img, encoded_prompts)
```

## Run Image

<p align="center">
  <img src=".readme_assets/run_image_anim.gif">
</p>

The `run_image.py` script will run the segment-anything model on a single image with an interactive UI running locally. To use the script, make sure you've activated the virtual environment (from the installation step) and then, from the repo folder use:
```bash
python run_image.py
```

You can also add  `--help` to the end of this command to see a list of additional flags you can set when running this script. One especially interesting flag is `-b`, which allows for processing images at different resolutions and `-ar` for processing images at their original aspect ratio (SAMv1 has better support for this than SAMv2!).

If you don't provide an image path (using the `-i` flag), then you will be asked to provide one when you run the script, likewise for a path to the model weights. Afterwards, a window will pop-up, with options for how to 'prompt' the model (e.g. bounding boxes or clicking to add points) along the top and various sliders to alter the segmentation results at the bottom. Results can be saved by pressing the `s` key.


## Run Video (or webcam)

The `run_video.py` script allows for segmentation of videos based on prompts on paused frames of the video using an interactive UI running locally. However, it only works with SAMv2 models!
To use the script, make sure you've activated the virtual environment (from the installation step) and then, from the repo folder use:
```bash
python run_video.py
```

As with the image script, you can add `--help` to the end of this command to see a list of additional flags. For example, you can add the flag `--use_webcam` to run segmentation on a live webcam feed.

This script is a messy work-in-progress for now, more features & stability updates to come!


# Acknowledgements

The code in this repo is entirely based off the original segment-anything github repos:

[facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```


[facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint},
  year={2024}
}
```


# TODOs
- Clean up code base (especially the image encoder, which is unfinished)
- Add interactive script replicating the original 'automatic mask geneartor'
- Add model structure documentation
- Add various experiment scripts (onnx export, block norm vis, mask prompts, imgenc size effect etc.)
- Inevitable bugfixes