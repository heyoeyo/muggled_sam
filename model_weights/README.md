# Model weights folder

Model weights can be placed in this folder, which enables auto-loading of the smallest model file when launching scripts (without providing a model path).

It also allows models to be referenced by partial name, instead of providing a full path when using script arguments to select a model.

For example if you place model files: `sam_base.pth`, `sam_large.pth` and `sam_huge.pth` in this folder, you can reference a specific model when launching the `run_image.py` script using something like:

```bash
python run_image.py -m large
```

This will cause the script to load the `sam_large.pth` file, since it contains 'large' in the filename.

You can download model files from the original [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) repo.
