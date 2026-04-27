# Changelog

Major breaking changes will be documented here. Smaller changes/feature updates are generally listed in [commit](https://github.com/heyoeyo/muggled_sam/commits/main/) messages.


## Upcoming changes

The following list is a set of planned changes that will involve breaking backwards compatibility with existing code using MuggledSAM models:

- The memory creation steps (for tracking) will be updated to bundle memory + object pointers as a single output instead of being handled/stored separately. This will reduce the complexity of managing video memory by keeping memory tokens + pointers in sync, though it will lead to a (very minor) break in consistency with the original model implementations. This is being done to help offset some of the increased complexity introduced with the v2.1/v3/v3.1 updates
- All models will be updated to require an additional instantiation step in order to 'make' an interactive, video or detector variant of the model. This is already done for the detector model, but will be extended to other use cases, to help group related functions together (and should simplify the overall model structure)
- The 'API' of the models will be updated, especially for video tracking. This is mostly for the 'initialize...' functions, which (misleadingly) imply the creation of internal state. There may also be some additional functions added at this point (e.g. for explicitly creating the video 'memory bank')
- The SimpleSAMURAI class will be renamed to MuggledSAMURAI at some point and will have similar API updates as described above

## Breaking changes

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