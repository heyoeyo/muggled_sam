# Demo Helpers

This folder contains scripts that are not strictly required to make use of the SAM models, but provide some helpful functionality for processing data and/or handling user interactions. This folder also contains all of the code for handling [UI](https://github.com/heyoeyo/muggled_sam/tree/main/lib/demo_helpers/ui) elements for the interactive scripts. A brief overview of each of the scripts in this folder is explained below:

#### contours.py

Contains functions useful for generating and processing contour data from masks. This is where mask outlines in all the demo scripts are generated.

#### crop_ui.py

This contains code for running a special cropping interface that is available when using the `--crop` script flag (e.g. the `python run_image.py --crop`).

#### history_keeper.py

Contains code to handle the re-use of previous image and/or model paths when launching scripts. That is, it helps provide the 'default' when launching scripts and getting the following input prompt:
```bash
           (default: /path/to/previously/used/image.jpg)
Enter path to image: 
```

#### loading.py

Contains a bunch of messy functions that handle the prompting for loading images & model pathing, but only if not already given valid loading paths. For example, this is where the simple cli model loading menu is generated:
```bash
Select model file:

  1: sam2_hiera_base_plus.pt
  2: sam2_hiera_large.pt (default)
  3: sam2_hiera_tiny.pt

Enter selection: 
```


#### mask_postprocessing.py

As the name suggests, this contains code for post-processing of mask data. This is mainly used in the [run_image.py](https://github.com/heyoeyo/muggled_sam?tab=readme-ov-file#run-image) script, where options are provided for padding or simplfying the SAM mask results.

#### misc.py

This contains functionality that doesn't neatly fit into the other scripts.

#### model_capture.py

Contains a helper class that can be used to 'capture' the intermediate results of a model for analysis or debugging. For example, it's used by the [block norm visualization](https://github.com/heyoeyo/muggled_sam/tree/main/experiments#block-norm-visualization) script to record all of the internal image features when running the SAM image encoder.

#### saving.py

Contains a bunch of messy functions for handling the saving of single-image mask results as well as video masking results. Also contains all of the (relatively tedious) image processing steps needed for adding alpha channels & cropping to masked regions.

#### shared_ui_layout.py

Contains the code used to generate the UI interface used across most of the scripts, which contains buttons for selecting different prompt types (e.g. hover vs. box vs foreground points) as well previewing mask results.

#### video_data_storage.py

Contains helper objects that are used exclusively with SAMv2 to help with storing and re-using the 'memory' results needed for video segmentation.


#### video_frame_select_ui.py

This contains code for running a special user interface that only appears when trying to use a video as input to a script that only supports (single) images. The interface allows the user to select a specific frame from the video for use in the script. For example it will trigger when running:
```python
python run_image.py -i /path/to/video.mp4
```