# SAMv1 State-Dict Conversion

The code in this folder is responsible for converting the original SAMv1 model weights into a format that properly matches the MuggledSAM implementation. This is mostly a matter of renaming weights, but in some cases also involves reshaping weight data.

This code is also responsible for determining the size/configuration of a model based on it's weights alone. This is why MuggledSAM does not require the user to specify a `model_type` (or use [hard-coded configs](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/build_sam.py#L14-L21)), as in the original repo.

### What is the 'state dict'?

The [state dict](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html) is how pytorch model weights are stored for re-use. It's literally a [dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) where each key is a string label that identifies the weight data, and the corresponding values are the actual (numeric) weights.

The labels are structured as a sequence of module names and potentially indices separated by periods, which indicate where a given weight belongs in the hierarchical structure of the model. For example, this label is from MuggledSAM:

```
prompt_encoder.box_encoder.tl_embed
```

This references an embedding belonging to the `prompt_encoder` [part of the model](https://github.com/heyoeyo/muggled_sam/tree/main/muggled_sam/v1_sam#prompt-encoder). Within the [prompt_encoder](https://github.com/heyoeyo/muggled_sam/blob/9e2322bf14c6f21f29584182f98198b3777db2b1/lib/v1_sam/sam_v1_model.py#L45) is a sub-module called the [box_encoder](https://github.com/heyoeyo/muggled_sam/blob/9e2322bf14c6f21f29584182f98198b3777db2b1/lib/v1_sam/prompt_encoder_model.py#L62) and within this sub-module is a (learned) parameter used to represent the top-left coordinate of box prompts, called [tl_embed](https://github.com/heyoeyo/muggled_sam/blob/9e2322bf14c6f21f29584182f98198b3777db2b1/lib/v1_sam/prompt_encoder_model.py#L164). All weight labels of the state_dict have this sort of 'reverse hierarchical' structure, that is, `tl_embed` belongs to `box_encoder` which belongs to `prompt_encoder` which belongs to the (SAM) model being loaded. The value corresponding to this label (key) in the state_dict is a [Tensor](https://pytorch.org/docs/stable/tensors.html) with a shape of 1x256, holding all of the learned numeric data associated with this embedding.

In order for a model to load correctly, the labels in the state_dict must _exactly_ match the structure given in the associated python code. Since MuggledSAM is a re-implementation of the SAM model, with a simplified structure and renamed weights, a somewhat elaborate 'find-and-replace' must be performed on these labels prior to loading the weights.


## Config from Original State Dict

This script is responsible for figuring out the 'config' of a model based on it's weights. For example, this helps determine if the base vs. large vs. huge model (or even some custom variant) is being loaded.

One of the defining features of the model sizes is the number of transformer blocks in the image encoder. In the original SAMv1 weights, these labels look like:

```
image_encoder.blocks.0.norm1.weight
...
image_encoder.blocks.4.attn.rel_pos_h
...
image_encoder.blocks.10.mlp.lin1.weight
...
```

The first index in these weights (0, 4 and 10 in the example above) corresponds to a block index in the model. The total number of blocks must be specified prior to loading. So rather than hard-coding the block count (12, 24 and 32 for the [base model](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/build_sam.py#L40), [large model](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/build_sam.py#L30) and [huge model](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/build_sam.py#L17), respectively), it's possible to look for the label with the highest `image_encoder.block.#` index, and use that value as the 'config' for loading the model. There are many other parameters that are determined this way, and often this process requires looking at the shape of the weight tensors in addition to the labels.

While this 'config-less' loading is nice, it does come at the cost of slower initial loading times (due to extra checks being performed). However, these checks are needed to perform the label conversion (see below) for loading MuggledSAM from the original SAM weights, so the slight slow down seems like a worthwhile tradeoff in this case.

## Convert Original State Dict Keys

While the config script is responsible for figuring out the sizing/shape of model parameters, the _conversion_ script is responsible for renaming (and in rare cases, reshaping) weights to match the names used in the MuggledSAM codebase.

As an example, the original SAM model does not explicitly represent the [4-stage structure](https://github.com/heyoeyo/muggled_sam/tree/main/muggled_sam/v1_sam#vision-transformer) or [windowing vs. global blocks](https://github.com/heyoeyo/muggled_sam/tree/main/muggled_sam/v1_sam/components#global--windowed-attention-blocks) of the image encoder. Instead, all blocks are arranged in a single sequence (with special per-block-index toggling to handle the structuring details). In MuggledSAM, the 4-stage structure is explicitly represented in the model code, along with the distinction between windowed and global attention blocks, and this affects the labeling of the weights. The last-most weight label of the base model image encoder is shown below, comparing the original SAM model vs. MuggledSAM:

```
Original SAMv1:
image_encoder.blocks.11.mlp.lin2.bias

MuggledSAM:
image_encoder.stages.3.global_attn_block.mlp.layers.2.bias
```

To be clear, these labels refer to the exact same data (and use within the model), the difference is purely due to differences in how the code represents the model structure. The goal of the conversion script is to perform these sorts of 'find-and-replace' operations needed to convert all of the original strings into their MuggledSAM counterparts so that the models can be loaded without errors.

### Reshaping layernorms

One other notable feature of the conversion script is the reshaping of weights associated with [layernorm2D](https://github.com/heyoeyo/muggled_sam/tree/main/muggled_sam/v1_sam/components#layernorm2d) instances. In the original codebase, the layernorm weights were [stored as 1D vectors](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/common.py#L34-L35), however at inference time, these values would be [reshaped to a 3D size](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/common.py#L42). In MuggledSAM, the weights are instead reshaped to the target size on the initial loading of weights, so that reshaping isn't needed at inference time.

## Key Regex

The key_regex.py script contains a collection of [regex](https://en.wikipedia.org/wiki/Regular_expression) and other string parsing functions. These are used to help find and rename weight labels in the model state dictionary. For example, these can be used to find all weights that include a pattern like: "block.0.attn.2", where the 0 and 2 can vary, which is helpful for renaming and restructuring weights.