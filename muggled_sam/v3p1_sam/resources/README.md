# SAMv3 Resources

This folder contains additional data for the SAMv3 model, which is not part of the model weights or codebase, but is nevertheless required for the model to operate

### BPE Vocab

The file has been copied here to make it easier to load the SAMv3 model. This file original comes from the SAMv3 github page:
https://github.com/facebookresearch/sam3/tree/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/assets

This file contains a (fairly long) list of string pairs used by the SAMv3 text tokenizer for byte-pair encoding (BPE). It's not clear why this wasn't included in the model weights...

Note that this version has been somewhat modified:
- The file has been truncated, as the v3 model only [uses about 19%](https://github.com/facebookresearch/sam3/blob/11dec2936de97f2857c1f76b66d982d5a001155d/sam3/model/tokenizer_ve.py#L140-L144) of the original file
- The file has been compressed with lzma (originally gzipped) because it gives a slightly smaller file size