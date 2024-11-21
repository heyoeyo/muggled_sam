# Library scripts

This folder contains all of the code associated with the MuggledSAM implementations of the Segment-Anything v1 and v2/v2.1 models. The [make_sam.py](https://github.com/heyoeyo/muggled_sam/blob/main/lib/make_sam.py) script is a convenient helper used to initialize either the v1 or v2 SAM models using only a path to the model weights (the model version & size/config is inferred from the weights themselves).

## V1 & V2 Folders

Code for the [v1](https://github.com/heyoeyo/muggled_sam/tree/main/lib/v1_sam) and [v2](https://github.com/heyoeyo/muggled_sam/tree/main/lib/v2_sam) models is held in entirely separate folders. Much of the code between these two models is identical, but has been duplicated (i.e. not [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) code) to keep the implementations completely independent. The code in these folders is written using relative imports which, while normally not ideal, means that the folder can be copy-pasted into other projects to make use of MuggledSAM without any complicated installation steps (aside from the [requirements.txt](https://github.com/heyoeyo/muggled_sam/blob/main/requirements.txt)!).

More information about each model's structure can be found inside of the respective folders.

## Demo Helpers

The [demo_helpers](https://github.com/heyoeyo/muggled_sam/tree/main/lib/demo_helpers) folder contains code that isn't specific to the SAM models, but provides useful functionality for handling model outputs (such as processing masks or contour data) or for handling user interactions (e.g. saving/loading data). More information about these scripts can be found in the folder itself.