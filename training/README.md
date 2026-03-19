# MuggledSAM - Training

This folder contains some experimental scripts that are specific to training/fine-tuning the SAM models. It currently supports (very basic) distillation of the SAMv3 image and/or text encoders.

## Text encoder distillation

Distillation takes a SAMv3 text encoder, shrinks it, and then trains the smaller model to 'match' the behavior of the original larger model. In this repo, this is done by matching _only_ the encoded text tokens from the smaller 'student' model to another 'teacher' model (typically the original SAMv3 model). While this can be somewhat limiting, the advantage of this approach is that no 'ground-truth' data is required, only a list of text prompts.

Distillation is done as a 3-part process:
1. Use the pruning script to create a copy of SAMv3 with a smaller text encoder
2. Use the text encoder distillation script to train the smaller (student) model to match the output of the original (teacher) model
3. Training saves a relatively small set of (LoRA) weights. So the merge script is used to merge the trained weights back into the pruned model

As mentioned, distillation requires a list of text prompts to train on. A file containing a few example prompts is auto-generated the first time this script is run. It can be manually edited to change or add more prompts. If you don't have specific text prompts that you're interested in, consider using a standard set of classes, like those from [coco128](https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml). After merging, the resulting model can be used with other MuggledSAM scripts, and can even be used with the original SAMv3 code with some minor modifications (these are explained when creating the pruned model).

It's worth noting that the main benefit of pruning the text encoder is to reduce the file size of the model, as the original text encoder makes up around 40% of the model file size. For use with a small number of text prompts, the encoder can be trained to work with just 1 layer (from 24 originally!), saving around 1.2GB. However this doesn't have a significant effect on the inference speed of the model, as the encoder accounts for only 1-2% of the inference time in typical detection use cases.

Distilling the text encoder may also be worthwhile for curiosity sake, as it's relatively quick to train. The distillation script is interactive and it's easy to experiment with the effect of adjusting the learning rate, lora sizing, batch accumulation or the effect of disabling training of certain (e.g. linear) layers. Adjusting the training prompts can also have a interesting impact on the model's ability to learn.



## Image Encoder distillation


<p align="center">
  <img src=".readme_assets/img_distill_anim.gif">
</p>
<p align="center">(This animation shows 40 seconds of training on coco128, sped up by a factor of 6)
</p>

This script allows for distilling the SAMv3 image encoder using an interactive UI. Results from the trained 'student' model are shown in green, while the ground-truth (from the 'teacher' model) is shown in purple, this happens in real-time as training progresses. To use the script, run (in a terminal):

```python
python3 run_distill_image_encoder.py
```

There are several optional flags that can be given to adjust the behavior of this script, which can be seen by running the command above with the `-h` flag. One very important option is the `--low_memory` flag, which can reduce the memory requirements down to around 4GB at the expense of slightly slower training speed. Adjusting `-r` (lora rank) can also have a significant impact on training results.

### Input requirements

Training the image encoder requires several resources (the script will prompt for these on startup):

1. A path to a test/validation image. This is used to provide visual feedback about the performance of the student model during training, it doesn't directly affect the results in any way.

2. A path to images used for training. This can be a path to a folder containing images, or otherwise a path to a text file that lists paths to the images. These should only be actual images and not masks (this distillation script doesn't use masks). The images provided here are what the student model will 'get good at'. A simple starting set of images (if not using your own) would be [COCO128](https://cocodataset.org/#home) which is a small dataset that can be downloaded from places like [ultralytics](https://github.com/ultralytics/yolov5/releases/tag/v1.0), [roboflow](https://universe.roboflow.com/team-roboflow/coco-128/dataset/1) or [kaggle](https://www.kaggle.com/datasets/ultralytics/coco128). 

3. A path to a student model. The student should be made by running the image encoder pruning script to produce a smaller version of the original SAMv3 model. The smaller the student model, the more difficult it will be to train to match the original SAMv3 model (but it will also run faster).

4. A path to the teacher model. This script trains the student model to mimic the teacher, so it acts as the 'ground truth' for training. Generally this should be the [original SAMv3](https://github.com/heyoeyo/muggled_sam?tab=readme-ov-file#model-weights) model (e.g. `sam3.pt` file).


### Final trained weights

At any point during training, the current 'weights' can be saved. However, this only saves [LoRA](https://arxiv.org/abs/2106.09685) components, not a full model! This is done to allow for saving many files without excessive disk usage (the LoRA weights are only 10-100 MB).

In order to get back a full model, the merging script must be run. This script takes in the model that was used for training along with the trained weights and merges them to create a new (trained) model file. The resulting file is structured to be compatible with the original SAMv3 repo, but may require adjustments to the original code to account for sizing changes (these adjustments are reported by the pruning script).
