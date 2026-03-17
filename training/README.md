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

This is the same as distilling the text encoder (see above), though it requires providing a set of images for training. A simple starting set of images (if not using your own) would be [COCO128](https://cocodataset.org/#home) which is a small dataset that can be downloaded from places like [ultralytics](https://github.com/ultralytics/yolov5/releases/tag/v1.0), [roboflow](https://universe.roboflow.com/team-roboflow/coco-128/dataset/1) or [kaggle](https://www.kaggle.com/datasets/ultralytics/coco128).

Shrinking the image encoder has a more noticable impact on inference time, especially if used with video tracking. However it also trains much slower than the text encoder, so using reduced resolutions (at least initially) can help to speed things up.
