# Muggled SAM - Experiments

This folder contains random experiments using the SAM models, mostly out of curiosity. These scripts have configurable options which can be viewed by running the scripts with the `--help` flag.


## Video with Image Priors

This experimental script is a follow-up to a [post on the SAMv2 issues board (#210)](https://github.com/facebookresearch/segment-anything-2/issues/210), where the idea of re-using the SAMv2 memory bank across videos/images was suggested. This script begins by having the user 'record' prompts from a loaded image and then uses these prompts as the initial memory (with no other prompts) to run segmentation on a separate (potentially unrelated) video:

<p align="center">
  <img src=".readme_assets/vidimgprior_prompt.webp" alt="">
  <img src=".readme_assets/vidimgprior_anim.gif" alt="">
</p>

Here for example, a single picture of a cat is used to segment a fox from a separate video. Running this experiment with different combinations of image prompts and videos can give some idea of the models' sense of 'similarity' between images.

