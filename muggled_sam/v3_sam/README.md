# SAMv3

The [SAMv3](https://github.com/facebookresearch/sam2) model is a follow-up to [SAMv2](https://github.com/facebookresearch/sam2) model, but includes a new 'detection' capability that was not part of the previous v1 or v2 models. It still supports prompt-based mask generation and video segmentation as well.

---

Here are some rough notes on the model implementation, with a focus on the parts that match the existing v1/v2 models:

#### General
- The v3 model is really made out of 2 different parts which are almost completely separate, except that they share an image encoder
- One part is a copy of the SAMv2.1 model (but with a different image encoder)
- The other part is used for object detection and is entirely new

#### Image Encoder
- The v3 image encoder is almost identical to the image encoder used in _v1_, except it does away with the complicated [decomposed relative position encodings](https://github.com/heyoeyo/muggled_sam/tree/main/muggled_sam/v1_sam/components#decomposed-relative-position-encoder) in favor of instead using [RoPE](https://arxiv.org/abs/2104.09864) on each layer
- Like v1, it's a 4-stage 'ViTDet' model based on the paper: [Exploring Plain Vision Transformer Backbones for Object Detection](https://arxiv.org/abs/2203.16527). The v3 model is sized similarly to the v1-huge model
- Although this is a very large model, the memory usage doesn't scale as poorly as the v1 image encoder. This allows the model to take in much higher resolution images. For example, it can process a 4000x4000px input using around 7GB of VRAM. By comparison, the v1 model takes around 20GB for a 2000x2000px input
- The model includes a strange [tiled](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/vitdet.py#L211-L214) position encoding prior to it's transformer stages, though it doesn't have much of an effect. This actually resembles the approach [used in SAMv2](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/backbones/hieradet.py#L277-L279)
- The image encodings have 2 different 'projections', depending on whether the encoded image features are being used for the older SAMv1/v2 masking tasks or the  newer v3 detection task. It literally refers to these as [sam2_out vs. sam3_out](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/necks.py#L126)

#### Prompt encoding/mask decoder
- This is exactly the same as the SAMv2.1 model (which itself is nearly the same as v1)

#### Video memory encoding
- Again, this is exactly the same as SAMv2.1

---

Here are some notes specific to the newer detection task supported by the v3 model:

#### General
- The new object detection task is implemented in a way that mirrors the existing 'encode image, encode prompts, generate masks' sequence that SAMv1/v2 use, except it's now 'encode image, encode exemplars (prompts), generate detections'. In this repo, the new image encoding step is called 'encode_detection_image' to help distiguish it from the existing encoding, which isn't compatible!
- At a high level, the task is handled by an image encoder which produces encoded image tokens, a text encoder & image sample encoder (analogous to prompt encoding from v1/v2) which produces encoded 'exemplar' tokens, and a detector & segmentation model which produces the output masks & boxes
- The detection model **always** generates 200 detections, this is the result of learned embeddings and cannot be changed (without re-training). However, the model also predicts scores that can be used to filter out detections which aren't relevant

#### Text encoder
- The text encoder is very new and unlike any existing model component. It uses [byte pair encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding) (BPE) to 'compress' any possible input text into blocks of known words or word pieces that belong to a (learned?) [vocabulary](https://github.com/facebookresearch/sam3/tree/main/sam3/assets)
- For some reason, the included vocab has far more entries (260000+) than [what the model uses](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/tokenizer_ve.py#L144) (~50000). This repo only includes the [parts of the vocab](https://github.com/heyoeyo/muggled_sam/tree/main/muggled_sam/v3_sam/resources) that are being used by the model
- The way the vocab is built seems like it might favor ASCII text (e.g. typical english), though in theory it supports all possible text inputs
- The model has a built-in maximum [context length of 32](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/text_encoder_ve.py#L263) (this is 32 'vocabulary pieces' not words) due to a learned position encoding. Though in this implementation, larger inputs can be given and the position encoding will just be interpolated to fit (probably not useful in practice, but avoids runtime errors)


#### Sampling encoder
- This model is analogous to the 'prompt encoder' from v1/v2, but instead of just encoding the prompt coordinates, it also samples from the encoded image tokens
- Weirdly, in addition to directly encoding the prompt coordinates themselves, there is also a 'position encoding' of the coordinates (similar to how v1/v2 does it). So the resulting point (and box) encodings are formed like:

point_encoding = [sample_image(img, point_xy)](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/geometry_encoders.py#L605) + [direct_encoder(point_xy)](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/geometry_encoders.py#L594) + [position_encoder(point_xy)](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/geometry_encoders.py#L618)

- Box prompts are formed by creating a ([somewhat complicated](https://docs.pytorch.org/vision/main/generated/torchvision.ops.roi_align.html)) dense grid of sampling points over the provided box region, averaging groups of these points together to form a 7x7 grid of samples and then uses a final (learned) weighted average to form a single 'encoded image sample'
- The original implementation includes a [mask encoder](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/geometry_encoders.py#L700) as part of the 'geometry encoder', but the model doesn't include weights for this (so it isn't currently supported in this repo)
- In the original implementation, this is referred to as [encoding prompts](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/sam3_image.py#L166) (using something called a [geometry encoder](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/sam3_image.py#L188)), though this is completely separate from the existing prompt encoder from v1/v2

#### Image-Exemplar Fusion model
- In this repo, the term 'exemplar' refers to any combination of encoded text and/or encoded point/box samples. They're analogous to encoded prompts from the v1/v2 mask generation task
- All this model does is 'fuse' information from the exemplar tokens into the encoded image tokens, forming a new set of image tokens which are used by the detector and segmentation models
- In the original implementation, this component is just referred to as the [encoder](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/sam3_image.py#L211) (or 'transformer.encoder')

#### Exemplar detector
- This model takes in the updated ('fused') image tokens and exemplar tokens (e.g. text, points & boxes) and produces candidate detections in the form of bounding box predictions along with confidence scores for each box (and a global 'are there any objects' score).
- It also generates a set of encoded 'query' tokens which are used by the segmentation model
- Note that the detector doesn't generate masks for object detections, just the bounding boxes!
- In the original implementation, this is just referred to as the [decoder](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/sam3_image.py#L251) (or 'transformer.decoder')

#### Exemplar segmentation
- This model takes in the 'query' tokens from the detector, along with the fused image tokens and exemplar tokens to produce segmentation masks for all detections
- This is it's own standalone model, segmentation is not done using the existing mask generation components from v1/v2
- It will generate one mask for each query token it's given, so it's possible to reduce the computational workload by pre-filtering query tokens associated with low-score detections, though this may impact mask quality due to the way the model is structured
- Somewhat surprisingly, this model doesn't make use of the bounding box predictions. It independently generates the masks for each object
- It makes use of what seems to be a [better way](https://distill.pub/2016/deconv-checkerboard/) of handling [mask upscaling](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/maskformer_segmentation.py#L174), compared to how the v1/v2 [mask decoder works](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/sam/mask_decoder.py#L221-L224) (i.e. interpolation + convolution vs. transpose convolutions)
- It also generates a [semantic segmentation](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/maskformer_segmentation.py#L325) output, though it's not clear what this represents (may be used for training)
- In the original implementation, this component is referred to as the [segmentation head](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/sam3_image.py#L385)