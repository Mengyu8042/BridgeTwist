# CelebA-HQ component provenance

`src/celebahq_backbone.py` contains the NCSN++ architecture definitions used
by the CelebA-HQ SB checkpoint. These definitions were extracted
from the CelebA-HQ implementation of [TFTF](https://github.com/Qqq-22-Rere/tftf),
which uses the [RectifiedFlow](https://github.com/gnobitab/RectifiedFlow)
CelebA-HQ architecture. 

The CLIP checkpoint comes from
[LAION's OpenCLIP ConvNeXt-XXLarge augreg-soup release](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup),
revision `9f3e8ee3f383c672388d9178afe70af9e63ac9df`. Its model card declares
the MIT License. The accompanying OpenCLIP license is reproduced in
`open_clip_LICENSE.txt` from the upstream `v2.32.0` release.
