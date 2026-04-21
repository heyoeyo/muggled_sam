# SAMv3.1

This folder contains an implementation of SAMv3.1, an [update](https://github.com/facebookresearch/sam3/commit/9f22cb976fb6e38dad5bb34940fad852dd897d0e) released on March 27, 2026. There are several significant structural differences compared to the original v3.0 model, which is why this is handled as a standalone implementation.

The biggest change comes from an update to mask generation and the way memory encoding works during video segmentation. In SAM v2 & v3.0, each mask prediction (e.g. for each separate object) was encoded into it's own memory encoding. In order to segment objects on future frames, memory encodings would need to be processed separately for each object. This means that the inference time per frame in v2 or v3.0 scales linearly with the number of objects being tracked. In the v3.1 update, up to 16 masks can be used to produce a single set of memory tokens (referred to as [multiplexing](https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/sam3_multiplex_base.py#L196) in the original codebase). Similarly, the mask decoder component has been updated to always predict 16 masks per frame. As a result, the time needed to process 16 objects is the same as the time needed for just 1 object and so timing scales sub-linearly with the object count. Though this does come at the cost of (substantial) added complexity in managing the data required for multi-object tracking over longer periods of time.

The other big change in the v3.1 update is a more complete separation between model components used for 'interactive' segmentation (e.g. SAMv1 task), video segmentation (e.g. SAMv2 task) and object detection (e.g. SAMv3 task). These are almost entirely separate models now, sharing only the image encoder.

In terms of mask quality, the v3.1 model appears similar to v3.0. However there's a substantial improvement in 'whole object' IoU predictions during interactive segmentation, making the v3.1 model much better suited for certain use cases (e.g. [auto mask generation](https://github.com/heyoeyo/muggled_sam/tree/main/simple_examples#auto-mask-generator)).

## Detailed changes from v3.0:

In addition to major updates to the mask decoder and memory encoder, there are a surprising number of smaller changes between v3.0 and v3.1. It's also quite hard to find these changes as the v3.1 code is extraordinarily difficult to follow (imo at least).
To help document the differences, they're described below in terms of their effect on the MuggledSAM implementation:

---
### Image encoder & projection models

#### Image encoder
- There are no changes here actually! Just wanted to make this explicit as it's a major model component

#### Image projection
- There are now 3 different projection outputs. One for interactive (SAMv1) tasks, one for video (SAMv2) and one for object detection (SAMv3). For comparison, the v3.0 implementation has 2 projections, one used for detections and the other is shared for both interactive & video usage

---
### Mask decoder model

#### MaskDecoder (IMPORTANT)
- The model now includes two distinct variants of the mask decoder, one for interactivity and one for video segmentation. These have very similar structure, but completely separate weights! The 'video' variant is stored in a new [multiplex_video_masking](https://github.com/heyoeyo/muggled_sam/blob/bff0fe8bf7adc3c2eef837dc3a278903ff3d9741/muggled_sam/v3p1_sam/multiplex_video_masking_model.py#L22) model, to help distinguish it from the original mask decoder
- All 'cls' tokens now have an additional multiplex dimension added. For the image variant, the additional dimension is size 1 (doesn't do anything), but for the video variant it's size 16. This leads to the (video) decoder always generating 16 copies of each of the normal outputs. This is the major change in v3.1 (e.g. always making 16 different mask predictions)
- To maintain backwards compatibility (with v1/v2 & v3.0), multiplexed outputs are stored in the batch dimension (index 0) of each output to keep the same shape. This prevents normal use of batching for the video variant, but this shouldn't be an issue in practice
- There is now a new component ('MultiplexEncoder') which [adds an embedding](https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/multiplex_mask_decoder.py#L272) to the 'cls' mask tokens to indicate which of the 16 multiplex slots are being used. This only happens in the video variant of the model. This unfortunately leads to the decoder needing to know which of the multiplex masks to use _ahead of time_ (forces a minor break in backwards compatibility)

#### MaskHintEncoder
- There is now an 'image' and 'video' variant of the mask hint encoder. The image variant is the same as the [v3.0 implementation](https://github.com/heyoeyo/muggled_sam/blob/334e588fa4139c0d55ac733d2cba135f02e72c1d/muggled_sam/v3_sam/mask_decoder_model.py#L336), while the video variant does nothing (mask hints are not expected to be used in video segmentation)


#### ObjectPointerGen
- The additive 'no object embedding' used to encode pointers when there is no object (e.g. object score is too low) has been removed. It's now handled as a linear projection of the original pointer
- There's a slight deviation between the original and MuggledSAM implementation here. The MSAM version only includes the 'no object' projection on the video variant of the mask decoder, while the original code [computes it for both ](https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L884-L895)versions. In practice this shouldn't matter, since an object would always (typically) be present in interactive-mode

#### Coordinate encoder
- There are now two copies of the coordinate encoder, one for each mask decoder variant (with separate weights). The video variant stores it's coordinate encoder inside the new 'multiplex_video_masking' module, as it is only used by the decoder itself

---
### Memory encoder model

#### MemoryEncoder (IMPORTANT)
- Memory tokens now have 256 features per token (previously 64)
- There is no longer an [output projection](https://github.com/heyoeyo/muggled_sam/blob/334e588fa4139c0d55ac733d2cba135f02e72c1d/muggled_sam/v3_sam/memory_encoder_model.py#L61) layer
- Now expects mask inputs to have a shape of BxMxHxW (B batches, M multiplex count, H & W are height and width)
- Also expects object score input to have a shape of BxM
- The 'M' dimension should always be 16 (with default model config), with batching used to handle cases where >16 masks are needed
- There is a new helper function `_make_multiplex_batches` which will pad/reshape inputs to the correct Bx16 sizing. This is very important to properly support batched multiplexing!

#### MaskDownsampler (IMPORTANT)
- The input is now expected to be up to 16 masks, which is part of the major update in v3.1 (the other part being changes to the mask decoder to generate 16 masks)
- A new 'is_prompt_flags' tensor (called [embedded_conditions](https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L1693) in the original code) is appended to the 16 input masks (one entry for each mask). This is used to indicate which of the mask inputs is meant as a prompt encoding
- The [mask downsampler](https://github.com/heyoeyo/muggled_sam/blob/334e588fa4139c0d55ac733d2cba135f02e72c1d/muggled_sam/v3_sam/components/memory_encoder_components.py#L21) previously had a channel sequence like: [1, 4, 16, 64, 256]. In v3.1 it's now: [32, 16, 64, 256, 1024]. The new 32 channel input accounts for the 16 masks + 16 flags
- The [scaling and bias](https://github.com/heyoeyo/muggled_sam/blob/334e588fa4139c0d55ac733d2cba135f02e72c1d/muggled_sam/v3_sam/components/memory_encoder_components.py#L75-L76) parameters have been changed from +20 and -10 (respectively) to +2 and -1 in v3.1
- Masks are always (unconditionally) 'sigmoided' in v3.1. Previously 'prompt' masks were instead [thresholded](https://github.com/heyoeyo/muggled_sam/blob/334e588fa4139c0d55ac733d2cba135f02e72c1d/muggled_sam/v3_sam/components/memory_encoder_components.py#L112)

---
### Memory image fusion model

#### MemoryImageFusion

- The transformer now uses [8 heads](https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model_builder.py#L867) in v3.1 (originally [1 head](https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model_builder.py#L371) in v3)
- Memory tokens and object pointers now have a matching feature count (256) as well as a multiplex dimension. This changes how they're handled during memory concatenation and removes the need for the odd reshaping step that was [required previously](https://github.com/heyoeyo/muggled_sam/blob/334e588fa4139c0d55ac733d2cba135f02e72c1d/muggled_sam/v3_sam/memory_image_fusion_model.py#L255-L264)
- Memory concatenation now expects 'previous frame image tokens' as an input (in addition to memory tokens and object pointers) and produces a concatenated copy of these tokens with a shape matching the memory tokens
- The order of memory token temporal position encodings is reversed. In v3.0, there are 6 temporal position encodings + 1 encoding to represent prompt memory. The temporal entries were ordered so that the 'most recent' memory was stored in index 0 of the learned encodings. In v3.1, the 'oldest' encoding is in index 0 and the 'most recent' is index 5
- The last-most temporal encoding was reserved for prompt memory in v3 (and v2), but is now meant to represent encodings that are [too far away](https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L1433) to represent with the other encodings. However, in practice only prompt memories are kept long enough to get this encoding
- There are two small deviations between MSAM and the original code here: The first is that the original encodes non-prompt frames temporally backwards at first. For example, if the [relative position](https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L1437) in time (`tpos`) is printed out it reads: (1), (2, 6), (3, 5, 6), (4, 4, 5, 6), ..., (7, 1, 2, 3, 4, 5, 6). The first entry is always the prompt encoding and counts up normally (first 1 frame backwards, then 2, then 3 etc.), but as each new non-prompt encoding arrives, it counts the other way for some reason (e.g. the first non-prompt frame is seen as being 6 frames back in time, as the second arrives they're seen as 5 & 6 frames back etc.). Once there are 6 non-prompt frames, the indexing is correct. This is assumed to be a bug, so it isn't implemented in MSAM which just uses normal counting
- The second difference is that MSAM always encodes prompt memory as if it's '7 or more' frames backwards (i.e. the special 'too far away' index). This is much simpler and doesn't have a noticable impact on results

#### MemoryImageFusionTransformerLayer

- The layers now separately take in 'encoded image tokens (includes position encoding)', 'current frame image tokens' and 'previous frame image tokens' along with memory tokens & position encodings. These are [confusingly named](https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/decoder.py#L1209-L1212): `tgt`, `image` and `memory_image` respectively, in the original code
- The final MLP layer now uses [GELU](https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model_builder.py#L887) instead of [RELU](https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model_builder.py#L397) as it's activation function

#### RoPEAttentionBNC

- The q, k, v and output projection layers have been removed from this model. They are instead stored on the components using this block
- This change is needed to accommodate updates to the cross-attention implementation


#### RoPESelfAttention

- The q, k, v and output projection layers (originaly on [RoPEAttentionBNC](https://github.com/heyoeyo/muggled_sam/blob/334e588fa4139c0d55ac733d2cba135f02e72c1d/muggled_sam/v3_sam/components/memory_image_fusion_attention.py#L20)) are now stored and computed in this model directly, but it's the same implementation otherwise

#### RoPECrossAttention

- As with the self-attention block, the q, k, v and output projections are now computed directly on this component
- This block now takes in 'encoded image tokens (includes position encoding)', 'current image tokens' and 'previous image tokens' (in addition to memory tokens and memory position encodings) as inputs
- Each input has it's own q/k/v projection (there are now a total of 5 qkv projections, 1 for v, 2 q's and 2 k's)

---
### Detection

There aren't any changes to the detection components in the v3.1 update
