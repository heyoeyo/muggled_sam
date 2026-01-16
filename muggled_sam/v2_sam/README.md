# SAMv2

The [SAMv2](https://github.com/facebookresearch/sam2) (and v2.1) models are a follow-up to the original [Segment-Anything](https://github.com/facebookresearch/segment-anything) models, with a focus on faster execution and reduced memory requirements. [Like SAMv1](https://github.com/heyoeyo/muggled_sam/tree/main/muggled_sam/v1_sam), the v2 models act like a more intelligent form of a magic wand tool for segmenting parts of an image based on user-provided prompts. Additionally, SAMv2 introduces support for video segmentation, which is able to automatically propagate a starting segmentation through frames of a video, tracking objects as the move or change in appearance.

More details to come! (WIP)

---

Here are some rough notes on the model implementation compared to SAMv1:

#### General
- For image segmentation, the v2 model makes use of the same structure as the v1 model, though the image encoder implementation is very different
- For video segmentation, the v2 model introduces two new components: a memory encoder and a memory-image fusion model. The existing mask decoder is also re-purposed to be able to generate 'future masks' by using data from the fusion model (rather than prompts)

#### Image encoder
- The v2 image encoder is called [hiera](https://github.com/facebookresearch/hiera), which makes heavy use of hierarchical encoding (i.e. encoding image features at multiple resolutions)
- It's a 4-stage model, like in v1, but is extremely complex (structurally). However, it is also much faster and far more efficient with VRAM usage
- The hiera model has an oddly elaborate use of pooling and windowing, which seems to make it poorly suited to handling input image resolutions that are not integer multiples of the original (1024) trained resolution
- Unlike SAMv1, the v2 image encoder outputs 3 sets of image tokens with different resolutions and channel counts. Two of these are considered 'high-resolution' encodings, while the other is the base or low-resolution encoding and is analogous to the output of SAMv1 (and is the most important/impactful)
- The high-res image tokens don't matter much, they could be multiplied by zero ([try it!](https://github.com/heyoeyo/muggled_sam/blob/043a1c696178f597e6ccc9b15585f4ed15ec9ec0/muggled_sam/v2_sam/image_encoder_model.py#L134-L135)) and the model will often still work

#### Prompt encoding/mask decoder
- This is mostly the same as SAMv1, with some minor changes
- One difference is the use of the high-resolution image encodings when masks are being upscaled (see [SAMv1](https://github.com/heyoeyo/muggled_sam/blob/043a1c696178f597e6ccc9b15585f4ed15ec9ec0/muggled_sam/v1_sam/mask_decoder_model.py#L193) vs. [SAMv2](https://github.com/heyoeyo/muggled_sam/blob/043a1c696178f597e6ccc9b15585f4ed15ec9ec0/muggled_sam/v2_sam/mask_decoder_model.py#L261))
- The other difference is the addition two new outputs: an object score and an object pointer
- The object score is an indicator of whether the object to be masked is even in the image (this is important for video segmentation, it indicates a loss of tracking)
- The object pointer is a kind of representation of the masked object and plays a (mostly negligble) role in tracking objects over time

---

Here are some rough notes on the new components of SAMv2:

#### General
- There are really only 2 new components, a 'memory encoder' and a 'memory-image fusion' model
- The memory-image fusion step is referred as [preparing memory conditioned features](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L497) in the original implementation
- The original implementation actually refers to 3 new components: a memory encoder, memory attention and a memory bank. The memory encoder is the same as described above. The memory attention model is a part of the memory-image fusion process, while the memory bank is not considered a model component in this repo (see below)

#### Memory bank
- This is very important for video tracking as it holds data that is used to keep track of objects over time
- It is made up of four lists: 1 list of encoded 'prompt memory' (these are used to 'begin' tracking), 1 list of prior frame memory encodings (these keep track of the object appearance over time) and then 1 list of 'prompt object pointers' and 1 list of 'prior frame object pointers' these go along with the memory encodings but don't do very much
- The model is only trained to represent 6 positions backwards in time, so the list of prior frame memorys should generally only keep the last 6 frames
- The object pointers don't have the same sort of learned position encoding, so are not limited to storing only 6. By default, the model uses 16 though this can be varied without much consequence
- The original implementation manages the memory bank internally as if it were a component of the model. This makes the model much harder to use since there is no easy way to control this storage
- In this repo, the memory bank is treated as an input to the model (similar to how images are inputs) and it's up to the user to manage it themselves

#### Memory encoder
- This could also be called a 'mask-image fusion' model. It takes in image tokens and a mask prediction (from the mask decoder) to produce a memory encoding (or 'memory token')
- Memory encodings are 'image-like' in that they are shaped like image tokens, but with fewer channels
- If the mask prediction is the result of a direct user prompt (e.g. a point or box prompt), then the encoding is considered a 'prompt memory', whereas if the mask prediction is produced without a prompt (like during tracking) then it is a 'previous frame memory'

#### Memory-image fusion
- This model takes in all prompt memory encodings, previous frame memory encodings and the corresponding object pointers and 'fuses' this information into a provided set of image tokens to produce a new (updated) set of image tokens
- The fusion of past memory data is in some ways like baking in a prompt about what the object looks like. This allows the mask decoder to generate a segmentation mask without requiring a user prompt (i.e. the past memory encoding _is the prompt_)
- This component involves some of the most complex logic of the entire SAMv2 model, mostly due to handling temporal position encoding of past memory encodings and some complexity due to object pointers