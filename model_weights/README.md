# Model weights folder

Model weights can be placed in this folder. If a single model is placed in this folder, it will auto-load when launching scripts. If multiple models are stored, then a menu will appear on start-up, for example:
```bash
Select model file:

   1: sam_vit_b_01ec64.pth
  *2: sam_vit_l_0b3195.pth (default)
   3: sam_vit_h_4b8939.pth

Enter selection: 
```
Entering an index (e.g. 1) will select that model from the menu. Alternatively, partial model names can also be entered (e.g 'vit_l"). If a model has been loaded previously, it will be marked as 'default' and entering nothing will result in the default being chosen. A full path to a model file can also be provided here, if loading a model that isn't available in the list.

Files in this folder can also be be referenced by partial name when using script flags. For example if you place model files: `sam_base.pth`, `sam_large.pth` and `sam_huge.pth` in this folder, you can reference a specific model when launching the `run_image.py` script using something like:

```bash
python run_image.py -m large
```

This will skip the menu selection and load the `sam_large.pth` file, since it contains 'large' in the filename.

You can download model files from:
- SAMv3: [facebookresearch/sam3](https://github.com/facebookresearch/sam3/tree/757bbb0206a0b68bee81b17d7eb4877177025b2f?tab=readme-ov-file#getting-started)
- SAMv2: [facebookresearch/sam2](https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description)
- SAMv1: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
