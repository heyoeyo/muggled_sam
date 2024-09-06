# Muggled SAM - Experiments

This folder contains random experiments using the SAM models, mostly out of curiosity. These scripts have configurable options which can be viewed by running the scripts with the `--help` flag.


## Block Norm Visualization

This script is a companion to an earlier [block norm visualization](https://github.com/heyoeyo/muggled_dpt/tree/main/experiments#block-norm-visualization) script for depth-prediction models. The display shows the 'block norms' of the image features at every layer of the model (SAMv1 or v2), alongside a per-channel visualization. A paper titled [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) suggests that vision transformers (like the image encoder inside the SAM models) will end up with tokens that have unusually high value norms if they don't include 'register' tokens (neither version of SAM includes these). This script can help detect these artifacts in both SAMv1 or v2 as well as any fine-tuned variants.

<p align="center">
  <img src=".readme_assets/blocknorm_example.webp" alt="">
</p>

Interestingly, while the base+ and large SAMv2 models _do_ have these high-norm tokens as expected (see the blacked-out tiles in the example image above), the SAMv1 models **do not**! There are other interesting patterns as well, for example, the high-norm blocks of SAMv2 seem to exclusive appear in stage 3. Additionally, the tokens for the v1 models show surprisingly little differences from one block to another (vaguely suggesting that the models could make due with far fewer blocks?), while the v2 models show similarly small differences between blocks within each stage, but drastic differences between stages (likely due to pooling).


## Video with Image Priors

This experimental script is a follow-up to a [post on the SAMv2 issues board (#210)](https://github.com/facebookresearch/segment-anything-2/issues/210), where the idea of re-using the SAMv2 memory bank across videos/images was suggested. This script begins by having the user 'record' prompts from a loaded image and then uses these prompts as the initial memory (with no other prompts) to run segmentation on a separate (potentially unrelated) video:

<p align="center">
  <img src=".readme_assets/vidimgprior_prompt.webp" alt="">
  <img src=".readme_assets/vidimgprior_anim.gif" alt="">
</p>

Here for example, a single picture of a cat is used to segment a fox from a separate video. Running this experiment with different combinations of image prompts and videos can give some idea of the models' sense of 'similarity' between images.


## ViT Position Encoding Visualization

This script was made after observing that SAMv1 tends to outperform SAMv2 at image segmentation when working with downscaled or non-square images, figuring that it may have something to do with the positional encodings (in retrospect, the window sizing of v2 is probably the issue). The visualization here is for the encodings that are added to the initial patch embedding tokens to help represent the positioning of each token within the image.

Since they are added to the patch tokens just before the vision transformer, the encodings have a natural 'per-pixel' (or really per-token) format that can be visualized as an image by assigning a color to them based on the relative values of each token/pixel position. The UI includes a slider to adjust which feature 'channel' is being visualized, along with the [L2 norm](https://en.wikipedia.org/wiki/Norm_(mathematics)) (a.k.a the [hypotenuse](https://en.wikipedia.org/wiki/Hypotenuse)) of the encodings is also shown (by default) beside the per-channel visualization. There are also sliders for adjusting the patch sizing.

<p align="center">
  <img src=".readme_assets/vitposenc_example.webp" alt="">
</p>

It's interesting to compare the v1 and v2 encodings, as they are dramatically different! For example, SAMv1 has an obvious ordering and structure to the features. There are 4 distinct patterns that are evenly spaced along the channels, which start out as simple low-frequency bars and turn into complicated high-frequency patterns. By comparison, the v2 encodings are made of two parts, a window-tiling component (which can be disabled) and an underlying low-frequency sinuisoidal pattern which doesn't appear to be ordered along channels like the v1 models. Surprisingly, the v2-base model has noticably higher-frequency patterns in it's position encodings compared to all other models.

It's worth noting that the default [bfloat16](https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype) data type seems to corrupt the SAMv1 encodings to some extent, though it doesn't affect segmentation performance (...not sure why?). Switching to float32 (using the `-f32` flag when running the script) avoids this distortion.

## Window Size Visualization

This script was inspired by a [pull request](https://github.com/facebookresearch/segment-anything/pull/594) for the SAMv1 model which suggested some improvements when using a window size of 16 (which evenly 'tiles' into the default 64x64 patch sizing) instead of the base size of 14. SAMv2 has a far more complex use of windowing which varies by stage in a somewhat unpredictable pattern, leading to even more questions about the consequences of changing the window sizing.

<p align="center">
  <img src=".readme_assets/windowsizing_example.webp" alt="">
</p>

The UI allows for independently updating the window size of each stage (for both SAMv1 & v2) while watching what happens to the raw mask predictions (i.e. without thresholding) of the model for a given prompt. It's also possible to change the prompt as this is happening.

<p align="center">
  <img src=".readme_assets/windowsizing_anim.gif" alt="">
</p>




One interesting observation from playing with this script is that adjustments to stages 1, 2 & 3 of SAMv2 show consistent but distinct effects. For example, changes to stage 1 tend to shuffle around small artifacts, increasing sizing on stage 2 has a blurring effect while decreasing sizing on stage 3 destabilizes the masking entirely. By comparison, while SAMv1 is affected by changes to window sizing on any stage, it tends to be less consistent and less detrimental. As the original (SAMv1) pull request suggested, increasing the window size to 16 actually _speeds up_ the model slightly (this is true for v2 as well) without dramatically harming the masking results (not true for v2!).