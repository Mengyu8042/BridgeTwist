"""External classifiers for evaluating class-conditional image generation.

Independent of the guidance classifier used during sampling:
  - MNIST   : Super Learner ensemble (Curt-Park/handwritten_digit_recognition)
  - CIFAR-10: DenseNet-121 (huyvnphan/PyTorch_CIFAR10)

Each wrapper returns a callable:
  fn(x_img) -> class scores, where x_img is (K, C, H, W) in [-1, 1].
Only the highest-scoring class is used for accuracy evaluation.
"""

from typing import Callable

import torch


def load_external_classifier(dataset: str, device) -> Callable[[torch.Tensor], torch.Tensor]:
    if dataset == "cifar10":
        from .cifar10_densenet import load_cifar10_densenet
        return load_cifar10_densenet(device)
    if dataset == "mnist":
        from .mnist_super_learner import load_mnist_super_learner
        return load_mnist_super_learner(device)
    raise ValueError(f"No external classifier available for dataset={dataset}")
