# Changelog

Major breaking changes will be documented here. Smaller changes/feature updates are generally listed in [commit](https://github.com/heyoeyo/muggled_sam/commits/main/) messages.


## Upcoming changes

- Legacy implementations of model functionality (currently on the 'SAM Core' classes) will be removed in June 2026
- The 'step_video_masking' function will be updated to support tensor memory inputs, along with an indexing vector. This should allow for slightly more efficient (less memory thrashing) tracking setups as well as restoring potential for exact numerical consistency with the original SAM implementations (currently broken by 2026-05-19 update)

## Breaking changes

### 2026-05-20

- Sleep delays and warnings have been added to 'legacy' functionality (mostly anything not using the new 'context' API) to warn of future removal
- The legacy function implementations can be used to help understand how to update the model usage, if needed. See for example, the [SAM3 core class](https://github.com/heyoeyo/muggled_sam/blob/d45ebd7e39d00f4fd7de2363bdd2770d60199da5/muggled_sam/v3_sam/sam_v3_model.py#L118-L235)

### 2026-05-19

- Updated video segmentation 'memory encodings' to now always bundle memory tokens and object pointers together
- This change (very slightly) breaks numerical consistency with the original SAM implementations, which use different numbers of tokens & pointers
- This change simplifies model usage considerably and is planned to be updated to support tensor inputs to allow for mismatched sizings in the future
- If needed, the last commit that supports numerical consistency is: [5ff1547](https://github.com/heyoeyo/muggled_sam/tree/5ff15477294d9bed26bdccd2373130f15b5348d1)

For example, old code looked like:
```python
# Make prompt memory
init_mask, init_mem, init_ptr = track_model.encode_prompt_memory(...)

# Memory storage
prompt_mems = deque([init_mem])
prompt_ptrs = deque([init_ptr])
frame_mems = deque([], maxlen=6)
frame_ptrs = deque([], maxlen=15)
```

Now this changes to:
```python
# Make prompt memory (memory tokens & pointers are now bundled in 'init_mem')
init_mask, init_mem = track_model.encode_prompt_memory(...)

# Memory storage
prompt_mems = deque([init_mem])
frame_mems = deque([], maxlen=6)
```

- Similarly updated the 'make_sam_from_state_dict' to only return the sam model itself (no longer returns config dict)
- The model config is still available from the model itself

For example, the old code looked like:
```python
model_config_dict, sam_core = make_sam_from_state_dict(model_path)
```

Now this changes to
```python
sam_core = make_sam_from_state_dict(model_path)
model_config_dict = sam_core.get_config() # Only if needed
```

- The 'encode_image' function was also updated to only return the image encoding (no longer returns token or preencoding sizings)
- The sizing info can still be obtained using helper functions in demo_helpers (see example scripts)

For example, the old code looked like:
```python
encoded_img, token_hw, preencode_hw = model.encode_image(...)
```

The new code looks like:

```python
encoded_img = model.encode_image(...)

# Optional
from muggled_sam.demo_helpers.model_info import get_token_hw, get_preencoding_hw
token_hw = get_token_hw(encoded_img)
preencode_hw = get_preencoding_hw(model, ...)
```

### 2026-05-15

- Major re-naming/re-structure of video segmentation functions!
- 'initialize_video_masking' is now 'encode_prompt_memory'
- 'initialize_from_mask' is now 'encode_prompt_memory_from_mask'
- 'step_video_masking' isn't renamed, but now only returns model predictions (no memory encodings/pointers)
- A new 'encode_frame_memory' function has been added, which is used to generate the 'mem_enc/obj_ptr' outputs previously from the step function
- Also renamed step function inputs from 'previous_memory_encoding' to 'frame_memory_encodings' (and similar for 'previous_object_pointers')
- These changes are meant to better communicate the memory usage of the SAM video models and also make it much more flexible
- The 'SimpleSAMURAI' implementation was also renamed to MuggledSAMURAI and the interace was updated to account for memory encoding changes

For example, the old code used for video segmentation looked something like:

```python
# Setup
init_mask, init_mem, init_ptr = track_model.initialize_video_masking(...)
prompt_mems = deque([init_mem])
prompt_ptrs = deque([init_ptr])
prev_mems = deque([], maxlen=6)
prev_ptrs = deque([], maxlen=15)

# Video loop
for frame in frames_list:
    encoded_img, _, _ = track_model.encode_image(frame)
    obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = track_model.step_video_masking(
        encoded_imgs_list, prompt_mems, prompt_ptrs, prev_mems, prev_ptrs
    )
    if obj_score > 0:
        prev_mems.appendleft(mem_enc)
        prev_ptrs.appendleft(obj_ptr)
```

After the newest renaming, it looks like:

```python
# Setup
init_mask, init_mem, init_ptr = track_model.encode_prompt_memory(...)
prompt_mems = deque([init_mem])
prompt_ptrs = deque([init_ptr])
prev_mems = deque([], maxlen=6)
prev_ptrs = deque([], maxlen=15)

# Video loop
for frame in frames_list:
    encoded_img, _, _ = track_model.encode_image(frame)
    mask_preds, ious_preds, ptrs, obj_score = track_model.step_video_masking(
        encoded_img, prompt_mems, prompt_ptrs, prev_mems, prev_ptrs, return_best_only=True
    )
    if obj_score > 0:
        mem_enc, obj_ptr = track_model.encode_frame_memory(
            encoded_img, mask_preds, ptrs, obj_score
        )
        prev_mems.appendleft(mem_enc)
        prev_ptrs.appendleft(obj_ptr)
```

To help make the differences more apparent, try comparing the location and usage of the `mem_enc` and `obj_ptr` in the for loops.

### 2026-05-14

- Removed 'get_best_mask_index' function from all SAM models (was just a simple wrapper around torch.argmax)
- Removed 'check_have_prompts' function from all SAM models. The function is instead available in demo_helpers/prompts.py
- Renamed 'filter_results' to 'filter_detections' on SAM3 models
- Renamed all 'tlbr' (top-left/bottom-right) variables to xy1xy2
- These changes are mostly meant to simplify the SAM model API for easier maintance & readability

### 2026-05-04

- All models updated to use a new 'context' interface. With this update, after loading a SAM model (now called 'sam core'), the user is expected to call '.get_interactive_context()' or '.get_tracking_context()' or 'get_detector_context()', corresponding to the SAMv1, v2 or v3 tasks, respectively.
- Contexts do not involve any memory allocation, they're only being used to separate related functionality for each task

### 2026-04-28
- Completely removed support for pruning features from the text or image encoders as part of the distillation scripts. This was just too messy to maintain, and distilling models with reduced features doesn't work well anyways (at least in the simplistic way it was being done). If needed, it's of course still possible to do using older commits (e.g. [d2d0339](https://github.com/heyoeyo/muggled_sam/tree/d2d0339f488d9581f3c388ba7c81401f55b5144c))

### 2026-04-24

- Updated SAMv3.0 implementation to produce image encodings that mirror the format used by v3.1 (e.g. a 'list of lists of tensors'), in order to make v3/v3.1 models more interchangable (see [new](https://github.com/heyoeyo/muggled_sam/blob/2a085e72072dd30a9580044365a0bf4ea3e055c6/muggled_sam/v3_sam/sam_v3_model.py#L210) vs [old](https://github.com/heyoeyo/muggled_sam/blob/eee05a73b316781d167465bc0d4f6a62d8cbfba1/muggled_sam/v3_sam/sam_v3_model.py#L198)). This also makes the model less error prone when mixing video/detection encodings. However, it comes with a slight _reduction_ (~10%) in speed when detection isn't being used, as the detection encodings are computed regardless of whether they're needed. It's possible to avoid this performance hit by [manually computing](https://github.com/heyoeyo/muggled_sam/blob/2a085e72072dd30a9580044365a0bf4ea3e055c6/muggled_sam/v3_sam/image_projection_model.py#L70-L78) only the required projections, it's just no longer the default behavior

### 2026-04-23

- Updated SAMv2, v3 and v3.1 to output 'proper' object pointers when initializing from a mask. A recent update added blank (e.g. all zeros) pointers as outputs, but the're now computed more consistently with the original repos (see [new](https://github.com/heyoeyo/muggled_sam/blob/15c202fba481f1c1a4f3b0dda57fa3922ba075d1/muggled_sam/v2_sam/sam_v2_model.py#L345) vs. [old](https://github.com/heyoeyo/muggled_sam/blob/543403d6d0ecf060909759ef165ba72dc44eabf9/muggled_sam/v2_sam/sam_v2_model.py#L347)). In rare cases, it seems as though this can lead to a minor _degradation_ in tracking behavior. If needed, the prior behavior can be replicated by simply zeroing the pointers before storing them

### 2026-04-10

- Added blank pointer output to 'initialize_from_mask' functions, which previously returned only memory encodings. Any code using this function needs to be updated to accept two outputs (see [new](https://github.com/heyoeyo/muggled_sam/blob/334e588fa4139c0d55ac733d2cba135f02e72c1d/simple_examples/video_segmentation_from_mask.py#L56) vs. [old](https://github.com/heyoeyo/muggled_sam/blob/3f026b3debee3145ebe85efa60bd20a732b37a3a/simple_examples/video_segmentation_from_mask.py#L56)) or it will cause errors during video segmentation

### 2026-03-25

- The SAMv1 model implementation now wraps image encodings in a list (see [new](https://github.com/heyoeyo/muggled_sam/blob/37ee13bc1c4f5e6b497478cfb8d5ca0c5b6f1514/muggled_sam/v1_sam/sam_v1_model.py#L168) vs. [old](https://github.com/heyoeyo/muggled_sam/blob/71d9b3b5317b3ad077d58dbe812f44417fdc4e15/muggled_sam/v1_sam/sam_v1_model.py#L156)). While this is unnecessary on it's own, it makes the output more consistent with the formatting used by the SAMv2 & v3 implementations, so that models are more easily interchangable in various scripts

### 2026-03-03

- All models now disable gradients  (aka freeze weights) on start-up (see the [init](https://github.com/heyoeyo/muggled_sam/blob/70e373087e6e9d32db1be217979fe76930f692c8/muggled_sam/v1_sam/sam_v1_model.py#L55-L56) functions), this only matters if using the models in a training script as the trainable weights must now be explicitly re-enabled

### 2026-02-27

- Added support for loading MuggledSAM 'weights' directly. This is expected to be a somewhat unstable feature (loading will break any time module names are updated, though it will usually be easy to fix), but can be useful for reducing model file size and speeding up model loading. As an example for how to save/load MuggledSAM weights:

```python
import torch
from muggled_sam.make_sam import make_sam_from_state_dict

# Saving MuggledSAM weights
_, model = make_sam_from_state_dict("/path/to/sam_model.pt")
model.to(device="cuda", dtype=torch.bfloat16)
torch.save(model.state_dict(), "/path/to/mugsam_model.pt")

# Loading MuggledSAM weights works the same as the original weights
_, mugsam_model = make_sam_from_state_dict("/path/to/mugsam_model.pt")
```

For example, doing this with SAMv3 will result in a copy of the model (`mugsam_model.pt`) that loads faster and is only 1.7GB (the original is 3.5GB) due to the use of bfloat16. This can be done with any version of SAM. The downside is that these weights won't be usable with the original SAM code!

### 2025-12-04

- Major folder re-structure. The `lib` folder has been renamed to `muggled_sam` for pyproject.toml compatibility, which makes the repo 'installable' as if it were a library. However, this also requires updating all existing `import from lib...` statements to `import from muggled_sam...`