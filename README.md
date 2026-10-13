# BridgeTwist: Twisting Schrödinger bridges for training-free conditional sampling

This repository provides the implementation code for BridgeTwist, containing a toy example and image experiments.

## Contents

### Experiment scripts

| File | Purpose |
| --- | --- |
| `src/toy_example.py` | Gaussian-mixture example with an analytic posterior and BridgeTwist sampling. |
| `src/run_full_image_experiments.py` | Unified entry point for MNIST, CIFAR-10, and CelebA-HQ experiments. |

### Core implementation

| File or directory | Purpose |
| --- | --- |
| `src/smc_sb_image.py` | BridgeTwist image sampler. |
| `src/evaluation.py`, `src/utilities.py` | Evaluation metrics and shared utilities. |
| `src/sb_backbones.py`, `src/classifier_backbones.py` | Backbone definitions for MNIST/CIFAR-10 SB models and likelihood classifiers. |
| `src/celebahq_backbone.py`, `src/celebahq_clip.py` | CelebA-HQ backbone and CLIP text likelihood. |
| `src/external_classifiers/` | Optional external accuracy evaluation. |

### Configuration and resources

| File or directory | Purpose |
| --- | --- |
| `environment.yml` | Shared conda environment for all experiments. |
| `scripts/download_checkpoints.py` | Checkpoint download. |
| `checkpoints/manifest.yaml` | Checkpoint information. |
| `configs/celebahq_prompts.json` | The 40 supplied sets of CelebA-HQ text prompts. |
| `torchcfm_repo/torchcfm/` | Bundled TorchCFM files needed by the UNet wrappers. |
| `checkpoints/external/` | Bundled external-classifier code and expected locations for optional weights. |
| `licenses/` | Third-party license notices and provenance notes. |

## Setup

Run the commands below from the repository root.

### Environment

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate bridge_twist
```

### Required checkpoints

All six required checkpoints for MNIST, CIFAR-10, and CelebA-HQ are hosted in the public [BridgeTwist checkpoint repository](https://huggingface.co/datasets/Mengyu8042/bridge-twist-checkpoints). Download them with:

```bash
python scripts/download_checkpoints.py \
  --base-url https://huggingface.co/datasets/Mengyu8042/bridge-twist-checkpoints/resolve/main \
  --group required \
  --verify
```

The downloader saves each file to the path listed in [the manifest](checkpoints/manifest.yaml). With `--verify`, it checks that each file exists and matches the expected byte size.

| Local path | Purpose |
| --- | --- |
| `checkpoints/mnist_ckpt.pt` | Pretrained MNIST SB model. |
| `checkpoints/cifar10_ckpt.pt` | Pretrained CIFAR-10 SB model. |
| `checkpoints/celebahq_ckpt.pt` | Pretrained CelebA-HQ SB model. |
| `checkpoints/classifier_mnist.pth.tar` | MNIST class likelihood. |
| `checkpoints/classifier_cifar10.pth.tar` | CIFAR-10 class likelihood. |
| `checkpoints/open_clip_pytorch_model.bin` | CLIP text likelihood. |

If a restricted container network reports `Cannot assign requested address`, add `--ipv4` to the download command.

### Optional external classifiers

The `--external_classifier` option adds an independent accuracy metric to class-conditional experiments. It requires upstream weights that are listed in the manifest's `external` group and downloaded separately from the required checkpoints.

For CIFAR-10, obtain the DenseNet-121 weights from [huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) and place them at:

```text
checkpoints/external/cifar10_densenet_repo/cifar10_models/densenet121.pt
```

For MNIST, obtain the Super Learner weights from [Curt-Park/handwritten_digit_recognition](https://github.com/Curt-Park/handwritten_digit_recognition) and place them at:

```text
checkpoints/external/mnist_super_learner_repo/models/VGG16.h5
checkpoints/external/mnist_super_learner_repo/models/ResNet164.h5
checkpoints/external/mnist_super_learner_repo/models/WideResNet28_10.h5
checkpoints/external/mnist_super_learner_repo/models/MobileNet.h5
checkpoints/external/mnist_super_learner_repo/models/SuperLearner.h5
```

MNIST external evaluation also requires TensorFlow and Keras in the same environment:

```bash
pip install "tensorflow-cpu>=2.15,<2.16" "keras>=2.15,<2.16"
```

Once all optional weights are in place, verify them with:

```bash
python scripts/download_checkpoints.py --group external --verify
```

## Toy example

Run the Gaussian-mixture example:

```bash
python src/toy_example.py
```

Results and figures are saved under `results/toy_example/`.

## Image experiments

### Class-conditional sampling and inpainting

The examples below use MNIST. Replace `--dataset mnist` with `--dataset cifar10` to run the same tasks on CIFAR-10.

Run class-conditional sampling:

```bash
python src/run_full_image_experiments.py \
  --dataset mnist \
  --task class_cond \
  --K 64 \
  --N 800
```

Run inpainting:

```bash
python src/run_full_image_experiments.py \
  --dataset mnist \
  --task inpainting \
  --K 16 \
  --N 800
```

For class-conditional sampling, add `--compute_kid` to compute Kernel Inception Distance (KID), or `--external_classifier` to compute external accuracy after completing the optional setup above. These two options can be used together.

Metrics are saved to `results.json` in `results/image_experiments/<dataset>_<task>/bridge_twist_<timestamp>/`. To save sample images, add `--sample_grid_4x4` for class-conditional sampling or `--save_inpainting_examples` for inpainting.

### Text-prompted generation

CelebA-HQ generation uses the text prompts in `configs/celebahq_prompts.json` and reports CLIP_L. Run the first supplied prompt:

```bash
python src/run_full_image_experiments.py \
  --dataset celebahq --task text_cond \
  --prompt_idx 0 \
  --K 25 --N 800
```

To run all 40 prompts:

```bash
python src/run_full_image_experiments.py \
  --dataset celebahq --task text_cond \
  --prompt_idx -1 \
  --K 25 --N 800
```

Metrics are saved to `results.json` in `results/celebahq_t2i/bridge_twist_<timestamp>/`. Add `--save_samples` to also save endpoint tensors and PNG grids.

## Licenses and provenance

BridgeTwist code and SB checkpoints are released under the [MIT License](LICENSE). Third-party components retain their upstream license terms and notices listed below.

### Third-party components

| Component | Source and license information |
| --- | --- |
| MNIST/CIFAR-10 likelihood classifiers | Unmodified checkpoints from [VSehwag/minimal-diffusion](https://github.com/VSehwag/minimal-diffusion), redistributed under MIT. See [the license notice](licenses/minimal_diffusion_LICENSE.txt). |
| TorchCFM UNet wrappers | Bundled files needed for checkpoint loading; MIT. See [the TorchCFM license](torchcfm_repo/LICENSE). |
| CelebA-HQ NCSN++ architecture | Extracted from [TFTF](https://github.com/Qqq-22-Rere/tftf). Upstream sources are documented in [the CelebA-HQ provenance notice](licenses/celebahq_NOTICE.md). |
| OpenCLIP ConvNeXt-XXLarge weights | Unmodified [LAION release](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup), whose model card declares MIT. See [the OpenCLIP license notice](licenses/open_clip_LICENSE.txt). |
| CIFAR-10 DenseNet external classifier | From [huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10), under MIT. See [the bundled license](checkpoints/external/cifar10_densenet_repo/LICENSE). |
| MNIST Super Learner external classifier | From [Curt-Park/handwritten_digit_recognition](https://github.com/Curt-Park/handwritten_digit_recognition). No explicit upstream license was found; provenance for each optional checkpoint is recorded in the manifest. |

### Datasets

| Dataset | Source |
| --- | --- |
| MNIST | [Dataset website](https://yann.lecun.org/exdb/mnist/) |
| CIFAR-10 | [Dataset website](https://www.cs.toronto.edu/~kriz/cifar.html) |
| CelebA-HQ | [Official project and dataset instructions](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training) |

## Citation

```bibtex
@inproceedings{li2026bridgetwist,
  title     = {{BridgeTwist}: Twisting {Schr{\"o}dinger} Bridges for Training-Free Conditional Sampling},
  author    = {Li, Mengyu and Qu, Qianqian and Liu, Jun S.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
