"""Compute evaluation metrics for image experiments."""

import os
import sys
import time
import glob

import torch
from pytorch_msssim import ssim
from torch_fidelity import calculate_metrics

_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_src_dir)
sys.path.insert(0, _src_dir)

from utilities import systematic_resample


def compute_classification_metrics(
    samples,
    weights,
    classifier_fn,
    target_class,
    device,
    external_classifier_fn=None,
):
    """Compute class-conditional accuracies on resampled particles."""
    sample_device = samples.device
    if weights.device != samples.device:
        weights = weights.to(sample_device)
    weights = weights.float()
    weights = weights / weights.sum().clamp_min(1e-12)
    indices = systematic_resample(weights).to(sample_device)
    resampled = samples.index_select(0, indices)

    samples_dev = samples.to(device).clamp(-1, 1)
    indices_dev = indices.to(device)

    with torch.no_grad():
        likelihood_preds = classifier_fn(samples_dev).argmax(dim=-1)
        likelihood_correct = (likelihood_preds == target_class).float()
        likelihood_acc = likelihood_correct[indices_dev].mean().item()

        external_acc = None
        if external_classifier_fn is not None:
            external_preds = external_classifier_fn(samples_dev).argmax(dim=-1).to(device)
            external_correct = (external_preds == target_class).float()
            external_acc = external_correct[indices_dev].mean().item()

    return {
        "likelihood_acc": likelihood_acc,
        "external_acc": external_acc,
        "resampled_samples": resampled,
    }


def compute_rmse(x_recon, x_orig, mask=None):
    """Compute RMSE between reconstructed and original images."""
    if x_orig.dim() == 3:
        x_orig = x_orig.unsqueeze(0)
    x_orig = x_orig.expand_as(x_recon)

    if mask is not None:
        inpaint_mask = (1.0 - mask).unsqueeze(0).expand_as(x_recon)
        n_pixels = inpaint_mask.sum(dim=(1, 2, 3)).clamp(min=1)
        mse = ((x_recon - x_orig) ** 2 * inpaint_mask).sum(dim=(1, 2, 3)) / n_pixels
    else:
        mse = ((x_recon - x_orig) ** 2).mean(dim=(1, 2, 3))

    return torch.sqrt(mse.clamp(min=1e-12))


def compute_ssim_batch(x_recon, x_orig):
    """Compute SSIM between batches using pytorch_msssim."""
    x_r = (x_recon + 1) / 2
    x_o = (x_orig.expand_as(x_recon) + 1) / 2
    return ssim(x_r, x_o, data_range=1.0, size_average=False)


TORCH_FIDELITY_INCEPTION_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
    "weights-inception-2015-12-05-6726825d.pth"
)
TORCH_FIDELITY_INCEPTION_FILENAME = os.path.basename(TORCH_FIDELITY_INCEPTION_URL)


def _iter_torch_fidelity_weight_paths(extra_paths=None):
    """Yield plausible local paths for torch-fidelity Inception weights."""
    seen = set()
    candidates = []

    env_path = os.environ.get("TORCH_FIDELITY_INCEPTION_WEIGHTS")
    if env_path:
        candidates.append(env_path)

    candidates.append(
        os.path.join(_project_root, "checkpoints", TORCH_FIDELITY_INCEPTION_FILENAME)
    )
    candidates.append(
        os.path.join(torch.hub.get_dir(), "checkpoints", TORCH_FIDELITY_INCEPTION_FILENAME)
    )

    if extra_paths:
        candidates.extend(extra_paths)

    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            yield path


def find_torch_fidelity_inception_weights(extra_paths=None):
    """Return a local torch-fidelity Inception weights path if one exists."""
    for path in _iter_torch_fidelity_weight_paths(extra_paths):
        if os.path.isfile(path):
            return path
    return None


def ensure_torch_fidelity_inception_weights(download=True, verbose=True,
                                            retries=3, retry_delay=5.0,
                                            raise_on_error=False):
    """Ensure torch-fidelity Inception weights are available locally."""
    weight_path = find_torch_fidelity_inception_weights()
    if weight_path is not None:
        if verbose:
            print(f"Using cached torch-fidelity Inception weights: {weight_path}")
        return weight_path

    if not download:
        msg = (
            "torch-fidelity Inception weights not found locally. "
            "Set TORCH_FIDELITY_INCEPTION_WEIGHTS or place the file at one of: "
            + ", ".join(_iter_torch_fidelity_weight_paths())
        )
        if raise_on_error:
            raise FileNotFoundError(msg)
        if verbose:
            print(f"Warning: {msg}")
        return None

    target_path = os.path.join(
        torch.hub.get_dir(), "checkpoints", TORCH_FIDELITY_INCEPTION_FILENAME
    )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if verbose:
                print(
                    "Preparing torch-fidelity Inception weights "
                    f"(attempt {attempt}/{retries})..."
                )
            torch.hub.load_state_dict_from_url(
                TORCH_FIDELITY_INCEPTION_URL,
                model_dir=os.path.dirname(target_path),
                file_name=TORCH_FIDELITY_INCEPTION_FILENAME,
                progress=verbose,
            )
            if os.path.isfile(target_path):
                if verbose:
                    print(f"Prepared torch-fidelity Inception weights: {target_path}")
                return target_path
        except Exception as exc:
            last_error = exc
            for partial_path in glob.glob(f"{target_path}.*.partial"):
                try:
                    os.remove(partial_path)
                except OSError:
                    pass
            if attempt < retries:
                if verbose:
                    print(
                        "Inception weight download attempt "
                        f"{attempt}/{retries} failed: {exc}. "
                        f"Retrying in {retry_delay:.0f}s..."
                    )
                time.sleep(retry_delay)

    msg = (
        "Failed to prepare torch-fidelity Inception weights. "
        f"Download {TORCH_FIDELITY_INCEPTION_URL} to {target_path}, "
        "or set TORCH_FIDELITY_INCEPTION_WEIGHTS to a local file."
    )
    if raise_on_error:
        raise RuntimeError(msg) from last_error
    if verbose:
        print(f"Warning: {msg}")
    return None


class TensorImageDataset(torch.utils.data.Dataset):
    """Wrap (N, C, H, W) tensors so torch-fidelity can consume them directly."""

    def __init__(self, imgs: torch.Tensor):
        x = imgs.detach().cpu()
        if x.dtype != torch.uint8:
            if x.min().item() < 0:
                x = (x + 1.0) / 2.0
            x = x.clamp(0, 1)
            x = (x * 255.0).round().to(torch.uint8)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        self.x = x

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx]


def _unbiased_poly_mmd2(F_x: torch.Tensor, F_y: torch.Tensor) -> float:
    """Unbiased U-statistic estimate of polynomial-kernel MMD².

    k(x, y) = (x^T y / d + 1) ** 3
    """
    d = F_x.size(1)
    Kxx = (F_x @ F_x.t() / d + 1.0).pow(3)
    Kyy = (F_y @ F_y.t() / d + 1.0).pow(3)
    Kxy = (F_x @ F_y.t() / d + 1.0).pow(3)

    m = F_x.size(0)
    n = F_y.size(0)
    Kxx.fill_diagonal_(0.0)
    Kyy.fill_diagonal_(0.0)
    term_xx = Kxx.sum() / (m * (m - 1))
    term_yy = Kyy.sum() / (n * (n - 1))
    term_xy = Kxy.sum() * (2.0 / (m * n))
    return float((term_xx + term_yy - term_xy).item())


def compute_kid(real_imgs: torch.Tensor, gen_imgs: torch.Tensor, device,
                dataset: str, features_fn=None,
                subset_size: int = None, subsets: int = 20):
    """Kernel Inception Distance (KID) — unbiased polynomial MMD² on features.

    CIFAR-10 uses torch-fidelity with Inception-v3 pool3 features.
    MNIST uses the supplied classifier features and draws subsets of real
    features without replacement, keeping the generated features fixed.
    Return the mean and standard deviation across subsets, or (None, None)
    when fewer than two real or generated images are available.

    Args:
        real_imgs: (N, C, H, W) real images in [-1, 1].
        gen_imgs:  (K, C, H, W) generated images in [-1, 1].
        dataset:   "mnist" | "cifar10".
        features_fn: required for mnist, ignored for cifar10.
        subset_size: defaults to gen_imgs.size(0) (= K).
        subsets:   number of subset estimates.
    """
    if real_imgs is None or gen_imgs is None:
        return (None, None)
    K = int(gen_imgs.size(0))
    if K < 2 or real_imgs.size(0) < 2:
        return (None, None)
    sub = int(subset_size) if subset_size is not None else K
    sub = min(sub, K, int(real_imgs.size(0)))
    if sub < 2:
        return (None, None)

    if dataset == "cifar10":
        weight_path = ensure_torch_fidelity_inception_weights(
            download=True, verbose=False,
        )
        if weight_path is None:
            return (None, None)
        use_cuda = str(device) != "cpu" and torch.cuda.is_available()
        metrics = calculate_metrics(
            input1=TensorImageDataset(real_imgs),
            input2=TensorImageDataset(gen_imgs),
            cuda=use_cuda,
            kid=True,
            kid_subset_size=sub,
            kid_subsets=int(subsets),
            feature_extractor_weights_path=weight_path,
            verbose=False,
        )
        kid_mean = metrics.get("kernel_inception_distance_mean")
        kid_std = metrics.get("kernel_inception_distance_std")
        return (
            float(kid_mean) if kid_mean is not None else None,
            float(kid_std) if kid_std is not None else None,
        )

    if dataset == "mnist":
        if features_fn is None:
            raise ValueError("MNIST KID requires features_fn; load the likelihood classifier with return_features=True.")
        with torch.no_grad():
            F_gen = features_fn(gen_imgs.to(device)).to(device).float()
            F_real_all = features_fn(real_imgs.to(device)).to(device).float()
        N = F_real_all.size(0)
        vals = []
        for _ in range(int(subsets)):
            idx = torch.randperm(N, device=device)[:sub]
            F_real_sub = F_real_all[idx]
            vals.append(_unbiased_poly_mmd2(F_real_sub, F_gen))
        vals_t = torch.tensor(vals)
        return (float(vals_t.mean().item()), float(vals_t.std(unbiased=False).item()))

    return (None, None)
