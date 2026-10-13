"""Shared BridgeTwist utilities."""

import math

import torch


def eps_to_scaled_score(eps_pred: torch.Tensor, t: float, sigma: float) -> torch.Tensor:
    """Convert epsilon prediction to the scaled SB score convention."""
    sqrt_t1t = math.sqrt(t * (1.0 - t) + 1e-8)
    return -sigma * eps_pred / (2.0 * sqrt_t1t)


def compute_ess(log_weights: torch.Tensor) -> float:
    """Compute effective sample size from unnormalized log weights."""
    log_w = log_weights - torch.logsumexp(log_weights, dim=0)
    return torch.exp(-torch.logsumexp(2.0 * log_w, dim=0)).item()


def systematic_resample(weights: torch.Tensor, new_size: int | None = None) -> torch.Tensor:
    """Systematic resampling for normalized SMC weights."""
    num_weights = weights.shape[0]
    num_samples = new_size if new_size is not None else num_weights
    cumsum = torch.cumsum(weights, dim=0)
    u = (
        torch.arange(num_samples, device=weights.device, dtype=weights.dtype)
        + torch.rand(1, device=weights.device, dtype=weights.dtype)
    ) / num_samples
    indices = torch.searchsorted(cumsum, u)
    return torch.clamp(indices, 0, num_weights - 1)


def log_normal_density(x: torch.Tensor, mu: torch.Tensor, var: float) -> torch.Tensor:
    """Log density of N(mu, var * I), summed over the final dimension."""
    dim = x.shape[-1]
    diff = x - mu
    return -0.5 * dim * math.log(2.0 * math.pi * var) - 0.5 * torch.sum(diff ** 2, dim=-1) / var


def compute_g_schedule(
    t: float,
    g_init: float,
    sigma: float,
    schedule: str = "constant",
    delta_g: float = 1e-6,
) -> float:
    """Compute the BridgeTwist diffusion coefficient at time t."""
    if schedule == "constant":
        return g_init
    if schedule == "cosine":
        return max(g_init * math.cos(math.pi * t / 2.0), 0.01)
    if schedule == "sigma_t":
        scale = sigma * math.sqrt(max(t * (1.0 - t) + delta_g, delta_g))
        return g_init * scale
    raise ValueError(f"Unknown g_schedule: {schedule}")


class LikelihoodFunction:
    """Base class for likelihood functions p(y | x_1)."""

    def log_prob(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
