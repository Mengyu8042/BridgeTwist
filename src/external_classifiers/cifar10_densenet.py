"""CIFAR-10 DenseNet-121 external classifier (huyvnphan/PyTorch_CIFAR10)."""

import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO = os.path.join(_PROJECT_ROOT, "checkpoints", "external", "cifar10_densenet_repo")
_WEIGHTS = os.path.join(_REPO, "cifar10_models", "densenet121.pt")

_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2471, 0.2435, 0.2616)


def load_cifar10_densenet(device):
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from cifar10_models.densenet import densenet121

    model = densenet121()
    model.load_state_dict(torch.load(_WEIGHTS, map_location="cpu"))
    model.eval().to(device)

    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    from smc_sb_image import adapt_image_batch_to_shape

    @torch.no_grad()
    def classify(x_img: torch.Tensor) -> torch.Tensor:
        x = adapt_image_batch_to_shape(x_img, (3, 32, 32), pad_value=-1.0).to(device)
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x - mean) / std
        return model(x)

    return classify
