# Muggled DPT - Experiments

This folder contains random experiments using the SAM models, mostly out of curiosity. These scripts have configurable options which can be viewed by running the scripts with the `--help` flag.


## Video with Image Priors

This experimental script is a follow-up to a [post on the SAMv2 issues board (#210)](https://github.com/facebookresearch/segment-anything-2/issues/210), where the idea of re-using the SAMv2 memory bank across videos/images was suggested. This script begins by having the user 'record' prompts from a loaded image and then uses these prompts as the initial memory (with no other prompts) to run segmentation on a separate (potentially unrelated) video:


<div style="display:flex;gap:2rem;justify-content:center;align-items:center;">
  <img src=".readme_assets/vidimgprior_prompt.webp" alt="">
  <img src=".readme_assets/vidimgprior_anim.gif" alt="">
</div>

While the results can be hit or miss (especially for the small & tiny model variants), the models do occasionally exhibit a surprising 'intuition' when mapping prompts from images to videos. For example, in the image/animation above, an image of a cat is used to segment a fox from an unrelated video. For reference, this example uses the base+ model.

