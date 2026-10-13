"""BridgeTwist image-domain sampling and model-loading utilities."""

import sys
import os
import math
import types
import importlib
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

_src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _src_dir)
from classifier_backbones import get_classifier_backbone
from sb_backbones import (
    CifarBackboneConfig,
    CifarNCSNpp,
    MnistBackboneConfig,
    MnistUNet,
    conv3x3,
)
from utilities import (
    eps_to_scaled_score,
    compute_ess,
    systematic_resample,
    log_normal_density,
    compute_g_schedule,
    LikelihoodFunction,
)


# =============================================================================
# Image-compatible Model Wrapper
# =============================================================================

class ImageSBModel:
    """
    Wrap an SB model to provide velocity and score predictions for image particles.

    The wrapped model takes (t_batch, x_image) where:
      - t_batch: (K,) scalar timesteps
      - x_image: (K, C, H, W) image tensor

    The score network predicts noise (epsilon), converted to
    scaled score via eps_to_scaled_score() for the SB predictor.
    """

    def __init__(
        self,
        model,
        sigma: float,
        image_shape: tuple,
        device: str = "cpu",
        checkpoint_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model: SB module exposing predict_velocity and predict_noise
            sigma: SB diffusion parameter
            image_shape: (C, H, W)
            device: torch device string
        """
        self.model = model
        self.sigma = sigma
        self.image_shape = image_shape
        self.C, self.H, self.W = image_shape
        self.d = self.C * self.H * self.W
        self.device = device
        self.checkpoint_info = checkpoint_info or {}
        self.model.eval()

    def predict_velocity(self, t: float, x_flat: torch.Tensor) -> torch.Tensor:
        """Predict velocity from flat particles.

        Args:
            t: scalar time
            x_flat: (K, d) flat particles

        Returns:
            vt: (K, d) velocity predictions (flat)
        """
        K = x_flat.shape[0]
        x_img = x_flat.reshape(K, self.C, self.H, self.W)
        t_batch = torch.full((K,), t, device=x_flat.device)
        vt_img = self.model.predict_velocity(t_batch, x_img)
        return vt_img.reshape(K, -1)

    def predict_noise(self, t: float, x_flat: torch.Tensor) -> torch.Tensor:
        """Predict noise (epsilon) from flat particles.

        Args:
            t: scalar time
            x_flat: (K, d) flat particles

        Returns:
            eps: (K, d) noise predictions (flat)
        """
        K = x_flat.shape[0]
        x_img = x_flat.reshape(K, self.C, self.H, self.W)
        t_batch = torch.full((K,), t, device=x_flat.device)
        eps_img = self.model.predict_noise(t_batch, x_img)
        return eps_img.reshape(K, -1)

    def predict_score(self, t: float, x_flat: torch.Tensor) -> torch.Tensor:
        """Predict scaled score st = (sigma^2/2)*nabla log p_t from flat particles."""
        eps = self.predict_noise(t, x_flat)
        return eps_to_scaled_score(eps, t, self.sigma)



# =============================================================================
# Image Likelihood Functions
# =============================================================================

class ImageClassifierLikelihood(LikelihoodFunction):
    """
    Classifier-based likelihood for class-conditional generation.

    log p(y|x_1) = log softmax(classifier(x_1))[y]

    The classifier takes images in [-1, 1] (or [0, 1] after internal conversion).
    """

    def __init__(self, classifier_fn, target_class: int, image_shape: tuple):
        """
        Args:
            classifier_fn: Callable that takes (K, C, H, W) images in [-1,1]
                          and returns (K, num_classes) logits.
            target_class: Target class index y.
            image_shape: (C, H, W)
        """
        self.classifier_fn = classifier_fn
        self.target_class = target_class
        self.image_shape = image_shape

    def log_prob(self, y_unused: torch.Tensor, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_unused: Not used (target class stored in self.target_class)
            x_flat: (K, d) flat image particles

        Returns:
            log_probs: (K,) log p(y|x_1) for each particle
        """
        K = x_flat.shape[0]
        C, H, W = self.image_shape
        x_img = x_flat.reshape(K, C, H, W)

        # Clamp to valid range before classifier
        x_img_clamped = x_img.clamp(-1, 1)

        logits = self.classifier_fn(x_img_clamped)  # (K, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[:, self.target_class]


class InpaintingLikelihood(LikelihoodFunction):
    """
    Gaussian likelihood for image inpainting.

    Given observed pixels y_obs at mask positions:
      log p(y|x_1) = -||mask * (y - x_1)||² / (2*sigma_y²) + const

    This is the partial Gaussian observation model y = A*x_1 + noise,
    where A is the mask selection operator.
    """

    def __init__(self, y_obs: torch.Tensor, mask: torch.Tensor,
                 noise_std: float, image_shape: tuple):
        """
        Args:
            y_obs: (C, H, W) observed image (only mask=1 pixels matter)
            mask: (C, H, W) or (1, H, W) binary mask. 1 = observed, 0 = to inpaint.
            noise_std: Observation noise standard deviation sigma_y.
            image_shape: (C, H, W)
        """
        self.y_obs = y_obs  # (C, H, W)
        self.mask = mask    # (C, H, W) or (1, H, W)
        self.noise_std = noise_std
        self.noise_var = noise_std ** 2
        self.image_shape = image_shape

    def log_prob(self, y_unused: torch.Tensor, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_unused: Not used (observation stored internally)
            x_flat: (K, d) flat image particles

        Returns:
            log_probs: (K,) log p(y|x_1) per particle
        """
        K = x_flat.shape[0]
        C, H, W = self.image_shape
        x_img = x_flat.reshape(K, C, H, W)

        # Residual at observed pixels: mask * (y_obs - x_img)
        y_exp = self.y_obs.unsqueeze(0).to(x_img.device)  # (1, C, H, W)
        m = self.mask.unsqueeze(0).to(x_img.device)        # (1, C, H, W)
        residual = m * (y_exp - x_img)  # (K, C, H, W)

        # log p = -||residual||² / (2*sigma_y²)
        sq_res = (residual ** 2).reshape(K, -1).sum(dim=-1)  # (K,)
        log_probs = -sq_res / (2.0 * self.noise_var)
        return log_probs


# =============================================================================
# Image-domain twisting function
# =============================================================================

def compute_image_twisting_with_grad(
    particles_flat: torch.Tensor,
    t: float,
    sb_model: ImageSBModel,
    likelihood: LikelihoodFunction,
    y: torch.Tensor,
    grad_batch_size: int = 0,
    return_model_terms: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    Compute first-order twisting function and gradient for image particles.

    Args:
        particles_flat: (K, d) flat particles
        t: current time
        sb_model: ImageSBModel wrapping the velocity and score networks
        likelihood: likelihood function
        y: observation (unused, passed to likelihood.log_prob)
        grad_batch_size: If >0, process gradient in mini-batches to save memory.
        return_model_terms: If True, also return the current velocity and score
            terms so callers can reuse them for drift computation.
    Returns:
        log_psi: (K,) twist values used for importance weights
        grad_log_psi: (K, d) twist gradients used in the proposal drift
    """
    K, _ = particles_flat.shape
    effective_grad_batch_size = grad_batch_size

    def run_chunked(batch_size: int):
        if batch_size > 0 and K > batch_size:
            all_log_psi, all_grads = [], []
            all_vt = [] if return_model_terms else None
            all_st = [] if return_model_terms else None
            for start in range(0, K, batch_size):
                end = min(start + batch_size, K)
                chunk_out = _compute_twist_chunk(
                    particles_flat[start:end], t, sb_model, likelihood, y,
                    return_model_terms=return_model_terms,
                )
                if return_model_terms:
                    lp, g, vt, st = chunk_out
                    all_vt.append(vt)
                    all_st.append(st)
                else:
                    lp, g = chunk_out
                all_log_psi.append(lp)
                all_grads.append(g)
            if return_model_terms:
                return (
                    torch.cat(all_log_psi, 0),
                    torch.cat(all_grads, 0),
                    torch.cat(all_vt, 0),
                    torch.cat(all_st, 0),
                )
            return torch.cat(all_log_psi, 0), torch.cat(all_grads, 0)
        return _compute_twist_chunk(
            particles_flat, t, sb_model, likelihood, y,
            return_model_terms=return_model_terms,
        )

    while True:
        try:
            return run_chunked(effective_grad_batch_size)
        except torch.cuda.OutOfMemoryError:
            if not particles_flat.is_cuda:
                raise
            torch.cuda.empty_cache()
            if effective_grad_batch_size <= 0:
                effective_grad_batch_size = K
            if effective_grad_batch_size <= 1:
                raise
            new_batch = max(1, effective_grad_batch_size // 2)
            if new_batch == effective_grad_batch_size:
                new_batch = effective_grad_batch_size - 1
            print(
                f"  Warning: OOM in twist at batch_size={effective_grad_batch_size}. "
                f"Retrying with batch_size={new_batch}.",
                flush=True,
            )
            effective_grad_batch_size = new_batch


def _compute_twist_chunk(
    particles_flat: torch.Tensor,
    t: float,
    sb_model: ImageSBModel,
    likelihood: LikelihoodFunction,
    y: torch.Tensor,
    return_model_terms: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Compute first-order twist for a chunk of particles."""
    particles = particles_flat.detach().requires_grad_(True)
    K = particles.shape[0]
    sigma = sb_model.sigma
    C, H, W = sb_model.image_shape
    d = C * H * W

    # Reshape to image for UNet
    x_img = particles.reshape(K, C, H, W)
    t_batch = torch.full((K,), t, device=particles.device)

    # Evaluate velocity and noise with gradients enabled.
    out = sb_model.model(t_batch, x_img)  # (K, 2C, H, W)
    vt_img = out[:, :C]
    eps_img = out[:, C:]

    # Flatten for predictor formulas
    vt = vt_img.reshape(K, d)
    eps = eps_img.reshape(K, d)

    st = eps_to_scaled_score(eps, t, sigma)

    x_hat = _compute_score_mix_predictor(particles, t, vt, st)
    log_psi = likelihood.log_prob(y, x_hat)

    grad_log_psi = torch_grad(
        log_psi.sum(), particles, create_graph=False, retain_graph=False
    )[0]

    if return_model_terms:
        return log_psi.detach(), grad_log_psi.detach(), vt.detach(), st.detach()
    return log_psi.detach(), grad_log_psi.detach()


def _compute_score_mix_predictor(
    particles: torch.Tensor,
    t: float,
    vt: torch.Tensor,
    st: torch.Tensor,
) -> torch.Tensor:
    """Predict x_1 for first-order image likelihood twisting."""
    return particles + (1 - t) * vt + 2 * t * (1 - t) * st


# =============================================================================
# Main BridgeTwist Algorithm for Images
# =============================================================================

@torch.no_grad()
def image_smc_sb_sample(
    sb_model: ImageSBModel,
    likelihood: LikelihoodFunction,
    y: torch.Tensor,
    K: int = 64,
    N: int = 800,
    g_init: Optional[float] = None,
    ess_threshold: float = 0.5,
    grad_clip_norm: float = 50.0,
    g_schedule: str = "sigma_t",
    resample_start_frac: float = 0.1,
    tempering: str = "linear",
    grad_batch_size: int = 0,
    proposal_grad_scale: float = 1.0,
    delta_g: float = 1e-6,
    proposal_cov_epsilon: float = 1e-6,
    return_trajectories: bool = False,
    return_ess_history: bool = False,
) -> dict:
    """Run BridgeTwist SMC for image conditional generation."""
    sigma = sb_model.sigma
    C, H, W = sb_model.image_shape
    d = C * H * W
    device = sb_model.device

    particles = torch.randn(K, d, device=device)
    log_weights = torch.zeros(K, device=device)
    dt = 1.0 / N
    g_base = g_init if g_init is not None else sigma / 2.0
    resample_start_step = int(resample_start_frac * N)

    def temper_coeff(t_value: float) -> float:
        if tempering == "linear":
            return t_value
        if tempering == "quadratic":
            return t_value ** 2
        if tempering == "cubic":
            return t_value ** 3
        if tempering == "none":
            return 1.0
        raise ValueError(f"Unknown tempering: {tempering}")

    t_init = 1e-4
    beta_init = temper_coeff(t_init)
    with torch.enable_grad():
        current_log_psi_raw, current_grad_log_psi_raw, current_vt, current_st = compute_image_twisting_with_grad(
            particles,
            t_init,
            sb_model,
            likelihood,
            y,
            grad_batch_size,
            return_model_terms=True,
        )
    log_weights = beta_init * current_log_psi_raw

    traj_times = {0.0, 0.3, 0.5, 0.7, 1.0}
    trajectories = {} if return_trajectories else None
    ess_history = [] if return_ess_history else None
    if return_trajectories:
        trajectories[0.0] = particles.reshape(K, C, H, W).detach().clone()

    for n in range(N):
        t = min(max(n * dt, 1e-4), 1.0 - 1e-4)
        t_next = min((n + 1) * dt, 1.0 - 1e-4)
        g = compute_g_schedule(t, g_base, sigma, g_schedule, delta_g=delta_g)

        beta = temper_coeff(t)
        log_psi = beta * current_log_psi_raw
        grad_log_psi = beta * current_grad_log_psi_raw
        grad_log_psi = torch.nan_to_num(grad_log_psi, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norms = torch.norm(grad_log_psi, dim=-1, keepdim=True)
        grad_log_psi = grad_log_psi * (grad_clip_norm / grad_norms.clamp(min=grad_clip_norm))

        vt = current_vt
        st = current_st
        v_sb = vt - (1.0 - 2.0 * t) * st
        g_ratio = (g ** 2) / (sigma ** 2)
        unconditional_drift = v_sb + g_ratio * st
        conditional_drift = unconditional_drift + proposal_grad_scale * g ** 2 * grad_log_psi
        prior_var = g ** 2 * dt
        proposal_var = prior_var + proposal_cov_epsilon
        noise = math.sqrt(proposal_var) * torch.randn_like(particles)
        new_particles = particles + conditional_drift * dt + noise
        new_particles = torch.nan_to_num(new_particles, nan=0.0, posinf=3.0, neginf=-3.0)

        beta_next = temper_coeff(t_next)
        with torch.enable_grad():
            next_log_psi_raw, next_grad_log_psi_raw, next_vt, next_st = compute_image_twisting_with_grad(
                new_particles,
                t_next,
                sb_model,
                likelihood,
                y,
                grad_batch_size,
                return_model_terms=True,
            )
        log_psi_new = beta_next * next_log_psi_raw

        log_p_trans = log_normal_density(new_particles, particles + unconditional_drift * dt, prior_var)
        log_q_trans = log_normal_density(new_particles, particles + conditional_drift * dt, proposal_var)
        log_w_inc = log_p_trans + log_psi_new - log_q_trans - log_psi
        log_w_inc = torch.clamp(log_w_inc, -20.0, 20.0)
        log_weights = log_weights + log_w_inc

        ess = compute_ess(log_weights)
        if ess_history is not None:
            ess_history.append(ess)
        if n >= resample_start_step and ess < ess_threshold * K:
            normalized_weights = torch.softmax(log_weights, dim=0)
            indices = systematic_resample(normalized_weights)
            new_particles = new_particles[indices]
            next_log_psi_raw = next_log_psi_raw[indices]
            next_grad_log_psi_raw = next_grad_log_psi_raw[indices]
            next_vt = next_vt[indices]
            next_st = next_st[indices]
            log_weights = torch.zeros(K, device=device)

        particles = new_particles
        current_log_psi_raw = next_log_psi_raw
        current_grad_log_psi_raw = next_grad_log_psi_raw
        current_vt = next_vt
        current_st = next_st

        if return_trajectories:
            t_rounded = round((n + 1) * dt, 1)
            if t_rounded in traj_times and t_rounded not in trajectories:
                trajectories[t_rounded] = particles.reshape(K, C, H, W).detach().clone()

    final_log_psi = temper_coeff(1.0 - 1e-4) * current_log_psi_raw
    log_weights = log_weights + likelihood.log_prob(y, particles) - final_log_psi
    normalized_weights = torch.softmax(log_weights, dim=0)
    final_ess = compute_ess(log_weights)
    samples_img = particles.reshape(K, C, H, W)

    result = {
        "samples": samples_img,
        "log_weights": log_weights,
        "weights": normalized_weights,
        "ess": final_ess,
        "delta_g": delta_g,
        "proposal_cov_epsilon": proposal_cov_epsilon,
    }
    if return_trajectories:
        trajectories.setdefault(1.0, samples_img.detach().clone())
        result["trajectories"] = trajectories
    if return_ess_history:
        result["ess_history"] = torch.tensor(ess_history, device=device)
    return result


# =============================================================================
# Model Loading Utilities
# =============================================================================

_TORCHCFM_UNET_WRAPPER = None


def adapt_image_batch_to_shape(
    images: torch.Tensor,
    target_shape: tuple,
    pad_value: float = -1.0,
) -> torch.Tensor:
    """Center-crop or symmetrically pad image batches to a target shape."""
    squeeze = images.dim() == 3
    if squeeze:
        images = images.unsqueeze(0)

    target_c, target_h, target_w = target_shape
    batch, channels, height, width = images.shape

    if channels != target_c:
        if channels == 1 and target_c == 3:
            images = images.repeat(1, 3, 1, 1)
        elif channels == 3 and target_c == 1:
            images = images.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Cannot adapt channels from {channels} to {target_c}")

    if height > target_h:
        top = (height - target_h) // 2
        images = images[:, :, top:top + target_h, :]
    if width > target_w:
        left = (width - target_w) // 2
        images = images[:, :, :, left:left + target_w]

    pad_h = max(target_h - images.shape[-2], 0)
    pad_w = max(target_w - images.shape[-1], 0)
    if pad_h or pad_w:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        images = F.pad(images, (left, right, top, bottom), value=pad_value)

    return images.squeeze(0) if squeeze else images


def _load_torchcfm_unet_wrapper():
    global _TORCHCFM_UNET_WRAPPER
    if _TORCHCFM_UNET_WRAPPER is not None:
        return _TORCHCFM_UNET_WRAPPER

    torchcfm_root = os.path.join(os.path.dirname(_src_dir), "torchcfm_repo", "torchcfm")
    package_paths = [
        ("torchcfm", torchcfm_root),
        ("torchcfm.models", os.path.join(torchcfm_root, "models")),
        ("torchcfm.models.unet", os.path.join(torchcfm_root, "models", "unet")),
    ]
    for name, path in package_paths:
        pkg = sys.modules.get(name)
        if pkg is None or not hasattr(pkg, "__path__"):
            pkg = types.ModuleType(name)
            pkg.__path__ = [path]
            pkg.__package__ = name
            sys.modules[name] = pkg

    module = importlib.import_module("torchcfm.models.unet.unet")
    _TORCHCFM_UNET_WRAPPER = module.UNetModelWrapper
    return _TORCHCFM_UNET_WRAPPER


class JointVelocityScoreUNet(nn.Module):
    """Joint velocity+score model with a shared UNet backbone."""

    def __init__(self, unet: nn.Module, in_channels: int, time_scale: float = 1.0):
        super().__init__()
        self.unet = unet
        self.in_channels = in_channels
        self.time_scale = time_scale

    def forward(self, t, x):
        return self.unet(t * self.time_scale, x)

    def predict_velocity(self, t, x):
        return self.forward(t, x)[:, :self.in_channels]

    def predict_noise(self, t, x):
        return self.forward(t, x)[:, self.in_channels:]


class DualSBModel(nn.Module):
    """Dual-network SB model with separate velocity and score UNets."""

    def __init__(
        self,
        velocity_unet: nn.Module,
        score_unet: nn.Module,
        in_channels: int,
        velocity_time_scale: float = 1.0,
        score_time_scale: float = 1.0,
    ):
        super().__init__()
        self.velocity_unet = velocity_unet
        self.score_unet = score_unet
        self.in_channels = in_channels
        self.velocity_time_scale = velocity_time_scale
        self.score_time_scale = score_time_scale

    def forward(self, t, x):
        velocity = self.velocity_unet(t * self.velocity_time_scale, x)
        noise = self.score_unet(t * self.score_time_scale, x)
        return torch.cat([velocity, noise], dim=1)

    def predict_velocity(self, t, x):
        return self.velocity_unet(t * self.velocity_time_scale, x)

    def predict_noise(self, t, x):
        return self.score_unet(t * self.score_time_scale, x)


class MnistVelocityAdapter(nn.Module):
    """Wrap the MNIST velocity backbone to match the local (t, x) interface."""

    def __init__(self, unet):
        super().__init__()
        self.net = unet

    def forward(self, t, x):
        return self.net(x, t)


class Cifar10VelocityAdapter(nn.Module):
    """Wrap the CIFAR-10 velocity backbone to match the local (t, x) interface."""

    def __init__(self, ncsnpp):
        super().__init__()
        self.net = ncsnpp

    def forward(self, t, x):
        time_cond = (t * 999).clamp(0, 999).long()
        return self.net(x, time_cond)


class JointMnistBackboneAdapter(nn.Module):
    """MNIST backbone widened to output [velocity, score]."""

    def __init__(self, unet_2c, in_channels: int):
        super().__init__()
        self.unet = unet_2c
        self.in_channels = in_channels
        self.time_scale = 1.0

    def forward(self, t, x):
        return self.unet(x, t)

    def predict_velocity(self, t, x):
        return self.forward(t, x)[:, :self.in_channels]

    def predict_noise(self, t, x):
        return self.forward(t, x)[:, self.in_channels:]


class JointCifar10BackboneAdapter(nn.Module):
    """CIFAR-10 backbone widened to output [velocity, score]."""

    def __init__(self, ncsnpp_2c, in_channels: int):
        super().__init__()
        self.unet = ncsnpp_2c
        self.in_channels = in_channels
        self.time_scale = 1.0

    def forward(self, t, x):
        time_cond = (t * 999).clamp(0, 999).long()
        return self.unet(x, time_cond)

    def predict_velocity(self, t, x):
        return self.forward(t, x)[:, :self.in_channels]

    def predict_noise(self, t, x):
        return self.forward(t, x)[:, self.in_channels:]


def _create_joint_model(
    dataset: str,
    device: str,
    dim_override: Optional[tuple] = None,
    num_res_blocks: Optional[int] = None,
    dropout: Optional[float] = None,
    attention_resolutions: Optional[str] = None,
    use_scale_shift_norm: bool = False,
    num_head_channels: Optional[int] = None,
    time_scale: float = 1.0,
    channel_mult=None,
    num_channels: Optional[int] = None,
    resblock_updown: bool = False,
):
    UNetModelWrapper = _load_torchcfm_unet_wrapper()

    if dataset == "mnist":
        dim = tuple(dim_override) if dim_override is not None else (1, 28, 28)
        model_channels = num_channels or (128 if num_res_blocks and num_res_blocks >= 3 else 64)
        unet = UNetModelWrapper(
            dim=dim,
            num_channels=model_channels,
            num_res_blocks=num_res_blocks or 2,
            channel_mult=channel_mult,
            num_heads=4,
            num_head_channels=num_head_channels if num_head_channels is not None else 32,
            attention_resolutions=attention_resolutions or "14,7",
            dropout=dropout if dropout is not None else 0.1,
            learn_sigma=True,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
        )
    elif dataset == "cifar10":
        dim = tuple(dim_override) if dim_override is not None else (3, 32, 32)
        unet = UNetModelWrapper(
            dim=dim,
            num_channels=num_channels or 128,
            num_res_blocks=num_res_blocks or 2,
            channel_mult=channel_mult or [1, 2, 2, 2],
            num_heads=4,
            num_head_channels=num_head_channels if num_head_channels is not None else 64,
            attention_resolutions=attention_resolutions or "16",
            dropout=dropout if dropout is not None else 0.1,
            learn_sigma=True,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return JointVelocityScoreUNet(unet, dim[0], time_scale=time_scale).to(device), dim


def _create_single_unet(
    dataset: str,
    device: str,
    dim_override: Optional[tuple] = None,
    num_res_blocks: int = 2,
    dropout: float = 0.1,
    attention_resolutions: Optional[str] = None,
    use_scale_shift_norm: bool = False,
    num_head_channels: int = 64,
    channel_mult=None,
    num_channels: Optional[int] = None,
    resblock_updown: bool = False,
):
    UNetModelWrapper = _load_torchcfm_unet_wrapper()

    if dataset == "mnist":
        if dim_override is not None:
            dim = tuple(dim_override)
        else:
            need_pad = channel_mult is not None and len(channel_mult) >= 4
            dim = (1, 32, 32) if need_pad else (1, 28, 28)
        default_ch = 128 if num_res_blocks >= 3 else 64
        unet = UNetModelWrapper(
            dim=dim,
            num_channels=num_channels or default_ch,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult or [1, 2, 2],
            num_heads=4,
            num_head_channels=num_head_channels,
            attention_resolutions=attention_resolutions or "14,7",
            dropout=dropout,
            learn_sigma=False,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
        )
    elif dataset == "cifar10":
        dim = tuple(dim_override) if dim_override is not None else (3, 32, 32)
        unet = UNetModelWrapper(
            dim=dim,
            num_channels=num_channels or 128,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult or [1, 2, 2, 2],
            num_heads=4,
            num_head_channels=num_head_channels,
            attention_resolutions=attention_resolutions or "16",
            dropout=dropout,
            learn_sigma=False,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
        )
    elif dataset == "celebahq":
        dim = tuple(dim_override) if dim_override is not None else (3, 256, 256)
        unet = UNetModelWrapper(
            dim=dim,
            num_channels=num_channels or 128,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult or [1, 1, 2, 2, 2, 2, 2],
            num_heads=4,
            num_head_channels=num_head_channels,
            attention_resolutions=attention_resolutions or "16",
            dropout=dropout,
            learn_sigma=False,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return unet.to(device), dim


def _create_dual_model(
    dataset: str,
    device: str,
    dim_override: Optional[tuple] = None,
    velocity_arch: Optional[dict] = None,
    score_arch: Optional[dict] = None,
    num_res_blocks: int = 2,
    dropout: float = 0.1,
    attention_resolutions: Optional[str] = None,
    use_scale_shift_norm: bool = False,
    num_head_channels: int = 64,
    velocity_time_scale: float = 1.0,
    score_time_scale: float = 1.0,
    channel_mult=None,
    num_channels: Optional[int] = None,
    resblock_updown: bool = False,
):
    if velocity_arch is not None:
        velocity_unet, dim = _create_single_unet(
            dataset, device, dim_override=dim_override, **velocity_arch,
        )
    else:
        velocity_unet, dim = _create_single_unet(
            dataset,
            device,
            dim_override=dim_override,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            attention_resolutions=attention_resolutions,
            use_scale_shift_norm=use_scale_shift_norm,
            num_head_channels=num_head_channels,
            channel_mult=channel_mult,
            num_channels=num_channels,
            resblock_updown=resblock_updown,
        )

    if score_arch is not None:
        score_unet, dim_score = _create_single_unet(
            dataset, device, dim_override=dim_override, **score_arch,
        )
        if dim_score != dim:
            dim = dim_score
    else:
        score_unet, _ = _create_single_unet(
            dataset,
            device,
            dim_override=dim_override,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            attention_resolutions=attention_resolutions,
            use_scale_shift_norm=use_scale_shift_norm,
            num_head_channels=num_head_channels,
            channel_mult=channel_mult,
            num_channels=num_channels,
            resblock_updown=resblock_updown,
        )

    model = DualSBModel(
        velocity_unet,
        score_unet,
        dim[0],
        velocity_time_scale=velocity_time_scale,
        score_time_scale=score_time_scale,
    ).to(device)
    return model, dim


def _create_mnist_pretrained_velocity_arch(device: str = "cpu"):
    config = MnistBackboneConfig()
    unet = MnistUNet(
        in_channels=config.channels,
        model_channels=config.dim,
        channel_mult=config.dim_mults,
    )
    return MnistVelocityAdapter(unet).to(device)


def _create_cifar10_pretrained_velocity_arch(device: str = "cpu"):
    config = CifarBackboneConfig()
    model = CifarNCSNpp(config)
    return Cifar10VelocityAdapter(model).to(device)


class CelebAHQVelocityAdapter(nn.Module):
    """Use positive floating-point times for the CelebA-HQ Fourier embedding."""

    def __init__(self, ncsnpp: nn.Module):
        super().__init__()
        self.net = ncsnpp

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        time_cond = (t * 999).clamp(min=1.0)
        return self.net(x, time_cond)


def _create_celebahq_pretrained_velocity_arch(device: str = "cpu"):
    from celebahq_backbone import NCSNpp

    model = NCSNpp(
        image_size=256, num_channels=3, nf=128,
        ch_mult=(1, 1, 2, 2, 2, 2, 2), num_res_blocks=2,
        attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True,
        fir=True, fir_kernel=[1, 3, 3, 1], skip_rescale=True,
        resblock_type="biggan", progressive="output_skip",
        progressive_input="input_skip", progressive_combine="sum",
        embedding_type="fourier", fourier_scale=16, init_scale=0.0,
        centered=True, scale_by_sigma=True,
    )
    return CelebAHQVelocityAdapter(model).to(device)


def _create_joint_mnist_pretrained_arch(device: str = "cpu"):
    config = MnistBackboneConfig()
    channels = config.channels
    unet = MnistUNet(
        in_channels=channels,
        model_channels=config.dim,
        channel_mult=config.dim_mults,
    )
    old_conv = unet.final_conv[2]
    unet.final_conv[2] = nn.Conv2d(old_conv.weight.shape[1], 2 * channels, kernel_size=3, padding=1)
    joint_model = JointMnistBackboneAdapter(unet, channels).to(device)
    dim = (channels, config.image_size, config.image_size)
    return joint_model, dim


def _create_joint_cifar10_pretrained_arch(device: str = "cpu"):
    config = CifarBackboneConfig()
    channels = config.data.num_channels
    model = CifarNCSNpp(config)
    old_conv = model.all_modules[-1]
    model.all_modules[-1] = conv3x3(old_conv.weight.shape[1], 2 * channels, init_scale=0.0)
    joint_model = JointCifar10BackboneAdapter(model, channels).to(device)
    dim = (channels, config.data.image_size, config.data.image_size)
    return joint_model, dim


def _infer_sb_model_type(ckpt: dict) -> str:
    model_type = ckpt.get("model_type")
    if model_type in {"joint", "dual"}:
        return model_type
    if any(
        key in ckpt
        for key in ("velocity_config", "score_config", "ema_velocity_model", "ema_score_model")
    ):
        return "dual"
    return "joint"


_PRETRAINED_BACKBONE_TAGS = {
    "pretrained_backbone",
    "tftf",  # Legacy metadata tag in the released checkpoints; no baseline code is loaded.
}


def _uses_pretrained_backbone(tag: Optional[str]) -> bool:
    return tag in _PRETRAINED_BACKBONE_TAGS


def _rebuild_joint_image_model(ckpt: dict, device: str):
    dataset = ckpt["dataset"]
    dim = tuple(ckpt["dim"])
    init_type = ckpt.get("init_type") or "pretrained_backbone"

    if _uses_pretrained_backbone(init_type):
        if dataset == "mnist":
            return _create_joint_mnist_pretrained_arch(device)
        if dataset == "cifar10":
            return _create_joint_cifar10_pretrained_arch(device)
        raise ValueError(f"Unknown dataset: {dataset}")

    create_kwargs = {}
    for key in (
        "num_res_blocks",
        "dropout",
        "attention_resolutions",
        "use_scale_shift_norm",
        "num_head_channels",
        "time_scale",
        "channel_mult",
        "num_channels",
        "resblock_updown",
    ):
        value = ckpt.get(key)
        if value is not None:
            create_kwargs[key] = value

    return _create_joint_model(dataset, device, dim_override=dim, **create_kwargs)


def _rebuild_dual_image_model(ckpt: dict, device: str):
    dataset = ckpt["dataset"]
    dim = tuple(ckpt["dim"])
    vel_init_type = ckpt.get("vel_init_type") or ckpt.get("init_type") or "default"
    score_init_type = ckpt.get("score_init_type") or ckpt.get("init_type")
    velocity_config = dict(ckpt.get("velocity_config") or {})
    score_config = dict(ckpt.get("score_config") or {})

    if not velocity_config and vel_init_type != "default" and not _uses_pretrained_backbone(vel_init_type):
        raise ValueError(f"Checkpoint velocity architecture '{vel_init_type}' is not bundled in this supplement")
    if not score_config and score_init_type not in {None, "default"} and not _uses_pretrained_backbone(score_init_type):
        raise ValueError(f"Checkpoint score architecture '{score_init_type}' is not bundled in this supplement")

    vel_time_scale = velocity_config.pop("time_scale", 1.0)
    score_time_scale = score_config.pop("time_scale", 1.0)
    if dataset == "celebahq" and _uses_pretrained_backbone(vel_init_type):
        if dim != (3, 256, 256) or velocity_config:
            raise ValueError("Unsupported CelebA-HQ velocity architecture metadata")
        score_unet, _ = _create_single_unet(
            dataset, device, dim_override=dim, **score_config,
        )
        velocity_unet = _create_celebahq_pretrained_velocity_arch(device)
        model = DualSBModel(
            velocity_unet, score_unet, dim[0],
            velocity_time_scale=1.0, score_time_scale=score_time_scale,
        ).to(device)
        return model, dim

    model, rebuilt_dim = _create_dual_model(
        dataset,
        device,
        dim_override=dim,
        velocity_arch=velocity_config or None,
        score_arch=score_config or None,
        velocity_time_scale=vel_time_scale,
        score_time_scale=score_time_scale,
    )

    if tuple(rebuilt_dim) != dim:
        print(f"  Warning: rebuilt dual dim {rebuilt_dim} differs from checkpoint dim {dim}")

    if _uses_pretrained_backbone(vel_init_type):
        if dataset == "mnist":
            model.velocity_unet = _create_mnist_pretrained_velocity_arch(device)
        elif dataset == "cifar10":
            model.velocity_unet = _create_cifar10_pretrained_velocity_arch(device)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        model.velocity_time_scale = 1.0

    return model, dim


def load_sb_image_model(checkpoint_path: str, device: str = "cpu") -> ImageSBModel:
    """Load a trained image-domain SB model from joint or dual checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset = ckpt["dataset"]
    sigma = ckpt["sigma"]
    dim = tuple(ckpt["dim"])
    model_type = _infer_sb_model_type(ckpt)

    if model_type == "dual":
        model, dim = _rebuild_dual_image_model(ckpt, device)
    else:
        model, dim = _rebuild_joint_image_model(ckpt, device)

    if "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif model_type == "dual" and "ema_velocity_model" in ckpt and "ema_score_model" in ckpt:
        model.velocity_unet.load_state_dict(ckpt["ema_velocity_model"])
        model.score_unet.load_state_dict(ckpt["ema_score_model"])
    elif model_type == "dual" and "velocity_model" in ckpt and "score_model" in ckpt:
        model.velocity_unet.load_state_dict(ckpt["velocity_model"])
        model.score_unet.load_state_dict(ckpt["score_model"])
    else:
        raise KeyError("Checkpoint is missing both 'ema_model' and 'model' state dicts")

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    checkpoint_info = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_name": os.path.basename(checkpoint_path),
        "model_type": model_type,
        "dataset": dataset,
        "sigma": sigma,
        "dim": list(dim),
        "step": ckpt.get("step"),
        "init_type": ckpt.get("init_type"),
        "vel_init_type": ckpt.get("vel_init_type"),
        "score_init_type": ckpt.get("score_init_type"),
        "velocity_config": ckpt.get("velocity_config"),
        "score_config": ckpt.get("score_config"),
    }

    return ImageSBModel(model, sigma, dim, device, checkpoint_info=checkpoint_info)


def load_image_classifier(checkpoint_path: str, dataset: str = "mnist",
                          device: str = "cpu",
                          return_features: bool = False):
    """
    Load a pretrained image classifier for class-conditional likelihoods.

    Returns a callable: classifier_fn(x_img) -> logits
    where x_img is (K, C, H, W) in [-1, 1].

    If return_features=True, returns (classifier_fn, features_fn) where
    features_fn(x_img) -> (K, D) captures the pre-linear pooled activations
    (D=512 for ResNet18, 2048 for ResNet50).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        arch = ckpt.get("arch", "resnet50")
    else:
        sd = ckpt
        arch = "resnet50"

    if dataset == "mnist":
        if "resnet18" in arch:
            arch = "resnet18_mnist"
        elif "resnet50" in arch or arch == "resnet50":
            arch = "resnet50_mnist"

    model = get_classifier_backbone(arch=arch, num_classes=10)

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    expected_shape = (1, 28, 28) if dataset == "mnist" else (3, 32, 32)

    def classifier_fn(x_img):
        """Takes images in [-1, 1], returns logits."""
        x_img = adapt_image_batch_to_shape(x_img, expected_shape, pad_value=-1.0)
        x = (x_img + 1) / 2  # scale to [0, 1]
        x = x.clamp(0, 1)
        return model(x)

    if not return_features:
        return classifier_fn

    _feat_cache = {}

    def _pre_linear_hook(_module, inputs):
        _feat_cache["x"] = inputs[0].detach()

    model.linear.register_forward_pre_hook(_pre_linear_hook)

    @torch.no_grad()
    def features_fn(x_img):
        """Takes images in [-1, 1], returns pre-linear pooled features (K, D)."""
        x_img = adapt_image_batch_to_shape(x_img, expected_shape, pad_value=-1.0)
        x = (x_img + 1) / 2
        x = x.clamp(0, 1)
        _ = model(x)
        return _feat_cache["x"]

    return classifier_fn, features_fn


# =============================================================================
# Mask Generation Utilities
# =============================================================================

def create_half_mask(image_shape: tuple, side: str = "left") -> torch.Tensor:
    """Create a mask that keeps one half of the image.

    Args:
        image_shape: (C, H, W)
        side: "left" (keep left, inpaint right) or "top" (keep top, inpaint bottom)

    Returns:
        mask: (C, H, W) binary tensor. 1 = observed, 0 = to inpaint.
    """
    C, H, W = image_shape
    mask = torch.zeros(C, H, W)
    if side == "left":
        mask[:, :, :W // 2] = 1.0
    elif side == "top":
        mask[:, :H // 2, :] = 1.0
    else:
        raise ValueError(f"Unknown side: {side}")
    return mask


def create_random_mask(image_shape: tuple, keep_ratio: float = 0.5,
                       seed: int = 42) -> torch.Tensor:
    """Create a random pixel mask.

    Args:
        image_shape: (C, H, W)
        keep_ratio: Fraction of pixels to keep (observed).
        seed: Random seed for reproducibility.

    Returns:
        mask: (C, H, W) binary tensor. Same mask for all channels.
    """
    C, H, W = image_shape
    rng = torch.Generator().manual_seed(seed)
    spatial_mask = (torch.rand(1, H, W, generator=rng) < keep_ratio).float()
    mask = spatial_mask.expand(C, H, W)
    return mask
