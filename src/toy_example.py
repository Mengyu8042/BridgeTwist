"""Toy BridgeTwist experiment for the 2D Gaussian-mixture posterior."""

import argparse
import json
import math
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


SQRT2 = math.sqrt(2.0)
MU1 = torch.tensor([-SQRT2, -SQRT2])
MU2 = torch.tensor([SQRT2, SQRT2])
W1, W2 = 0.75, 0.25
VAR_DATA = 0.25
VAR_OBS = 0.25
SIGMA_SB = 1.0
Y_OBS = torch.tensor([0.0, 0.0])
MU_POST_1 = MU1 / 2.0
MU_POST_2 = MU2 / 2.0
VAR_POST = 1.0 / 8.0


def sigma_t_sq(t: float) -> float:
    return (1.0 - t) ** 2 + SIGMA_SB**2 * t * (1.0 - t) + t**2 * VAR_DATA


def log_normal_2d(x: torch.Tensor, mu: torch.Tensor, var: float) -> torch.Tensor:
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    diff = x - mu
    return -0.5 * (diff**2).sum(dim=-1) / var - math.log(2.0 * math.pi * var)


def component_responsibilities(x: torch.Tensor, t: float):
    var = sigma_t_sq(t)
    log_p1 = math.log(W1) + log_normal_2d(x, t * MU1.to(x.device), var)
    log_p2 = math.log(W2) + log_normal_2d(x, t * MU2.to(x.device), var)
    normalizer = torch.logsumexp(torch.stack([log_p1, log_p2], dim=0), dim=0)
    return torch.exp(log_p1 - normalizer), torch.exp(log_p2 - normalizer)


def analytical_velocity(x: torch.Tensor, t: float) -> torch.Tensor:
    mu1 = MU1.to(x.device)
    mu2 = MU2.to(x.device)
    var = sigma_t_sq(t)
    dsigma_dt = -2.0 * (1.0 - t) + SIGMA_SB**2 * (1.0 - 2.0 * t) + 2.0 * t * VAR_DATA
    alpha = dsigma_dt / (2.0 * var + 1e-10)
    g1, g2 = component_responsibilities(x, t)
    v1 = mu1 + alpha * (x - t * mu1)
    v2 = mu2 + alpha * (x - t * mu2)
    return g1.unsqueeze(-1) * v1 + g2.unsqueeze(-1) * v2


def analytical_score(x: torch.Tensor, t: float) -> torch.Tensor:
    mu1 = MU1.to(x.device)
    mu2 = MU2.to(x.device)
    var = sigma_t_sq(t)
    g1, g2 = component_responsibilities(x, t)
    s1 = -(x - t * mu1) / var
    s2 = -(x - t * mu2) / var
    return g1.unsqueeze(-1) * s1 + g2.unsqueeze(-1) * s2


def predict_x1(x: torch.Tensor, t: float) -> torch.Tensor:
    mu1 = MU1.to(x.device)
    mu2 = MU2.to(x.device)
    var = sigma_t_sq(t)
    g1, g2 = component_responsibilities(x, t)
    coeff = t * VAR_DATA / var
    e1 = mu1 + coeff * (x - t * mu1)
    e2 = mu2 + coeff * (x - t * mu2)
    return g1.unsqueeze(-1) * e1 + g2.unsqueeze(-1) * e2


def log_likelihood(x1: torch.Tensor) -> torch.Tensor:
    return log_normal_2d(x1, Y_OBS.to(x1.device), VAR_OBS)


def sample_ground_truth(n: int) -> torch.Tensor:
    n1 = int(round(n * W1))
    n2 = n - n1
    s1 = MU_POST_1 + math.sqrt(VAR_POST) * torch.randn(n1, 2)
    s2 = MU_POST_2 + math.sqrt(VAR_POST) * torch.randn(n2, 2)
    return torch.cat([s1, s2], dim=0)[torch.randperm(n)]


def compute_g_schedule(t: float, g_init: float, schedule: str, delta_g: float = 1e-6) -> float:
    if schedule == "constant":
        return g_init
    if schedule == "sigma_t":
        return g_init * math.sqrt(max(t * (1.0 - t) + delta_g, delta_g))
    raise ValueError(f"Unknown schedule: {schedule}")


def temper_coeff(t: float, tempering: str) -> float:
    if tempering == "linear":
        return t
    if tempering == "quadratic":
        return t * t
    if tempering == "none":
        return 1.0
    raise ValueError(f"Unknown tempering: {tempering}")


def systematic_resample(weights: torch.Tensor) -> torch.Tensor:
    k = weights.numel()
    cdf = torch.cumsum(weights, dim=0)
    u = (torch.arange(k, device=weights.device, dtype=weights.dtype) + torch.rand((), device=weights.device)) / k
    return torch.searchsorted(cdf, u).clamp(0, k - 1)


def bridge_twist_sample(
    K: int,
    N: int,
    g: float = SIGMA_SB,
    tempering: str = "linear",
    g_schedule: str = "constant",
    ess_threshold: float = 0.8,
    resample_start_frac: float = 0.1,
    proposal_grad_scale: float = 0.5,
    delta_g: float = 1e-6,
    proposal_cov_epsilon: float = 1e-6,
) -> dict:
    particles = torch.randn(K, 2)
    log_weights = torch.zeros(K)
    dt = 1.0 / N
    resample_start = int(resample_start_frac * N)

    t_init = 1e-4
    log_psi_raw = log_likelihood(predict_x1(particles, t_init))
    log_psi = temper_coeff(t_init, tempering) * log_psi_raw
    log_weights = log_psi.clone()
    ess_history = []

    for n in range(N):
        t = min(max(n * dt, 1e-4), 1.0 - 1e-4)
        t_next = min((n + 1) * dt, 1.0 - 1e-4)
        g_t = compute_g_schedule(t, g, g_schedule, delta_g=delta_g)
        beta = temper_coeff(t, tempering)
        beta_next = temper_coeff(t_next, tempering)

        vt = analytical_velocity(particles, t)
        st = analytical_score(particles, t)
        uncond_drift = vt + 0.5 * g_t**2 * st

        particles_g = particles.detach().requires_grad_(True)
        log_psi_g = beta * log_likelihood(predict_x1(particles_g, t))
        grad_twist = torch.autograd.grad(log_psi_g.sum(), particles_g)[0].detach()
        grad_twist = torch.nan_to_num(grad_twist, nan=0.0, posinf=0.0, neginf=0.0)
        norms = grad_twist.norm(dim=-1, keepdim=True)
        grad_twist = grad_twist * torch.clamp(20.0 / (norms + 1e-8), max=1.0)

        cond_drift = uncond_drift + proposal_grad_scale * g_t**2 * grad_twist
        prior_var = g_t**2 * dt
        proposal_var = prior_var + proposal_cov_epsilon
        noise = math.sqrt(proposal_var) * torch.randn_like(particles)
        new_particles = particles + cond_drift * dt + noise

        log_psi_new_raw = log_likelihood(predict_x1(new_particles, t_next))
        log_psi_new = beta_next * log_psi_new_raw

        log_p = log_normal_2d(new_particles, particles + uncond_drift * dt, prior_var)
        log_q = log_normal_2d(new_particles, particles + cond_drift * dt, proposal_var)
        log_weights = log_weights + log_p + log_psi_new - log_q - log_psi

        weights = torch.softmax(log_weights, dim=0)
        ess = 1.0 / (weights**2).sum().item()
        ess_history.append(ess)
        if n >= resample_start and ess < ess_threshold * K:
            idx = systematic_resample(weights)
            new_particles = new_particles[idx]
            log_psi_new = log_psi_new[idx]
            log_weights = torch.zeros(K)

        particles = new_particles
        log_psi = log_psi_new

    log_weights = log_weights + log_likelihood(particles) - log_psi
    weights = torch.softmax(log_weights, dim=0)
    final_ess = 1.0 / (weights**2).sum().item()
    return {
        "samples": particles.detach(),
        "weights": weights.detach(),
        "ess": final_ess,
        "ess_history": ess_history,
        "proposal_grad_scale": proposal_grad_scale,
        "delta_g": delta_g,
        "proposal_cov_epsilon": proposal_cov_epsilon,
    }


def mmd2_rbf(x: torch.Tensor, y: torch.Tensor, max_points: int = 2000) -> float:
    if x.size(0) > max_points:
        x = x[torch.randperm(x.size(0))[:max_points]]
    if y.size(0) > max_points:
        y = y[torch.randperm(y.size(0))[:max_points]]
    z = torch.cat([x, y], dim=0)
    dists = torch.cdist(z, z) ** 2
    bandwidth = torch.median(dists[dists > 0]).clamp_min(1e-6)
    gamma = 1.0 / (2.0 * bandwidth)
    kxx = torch.exp(-gamma * torch.cdist(x, x) ** 2).mean()
    kyy = torch.exp(-gamma * torch.cdist(y, y) ** 2).mean()
    kxy = torch.exp(-gamma * torch.cdist(x, y) ** 2).mean()
    return float((kxx + kyy - 2.0 * kxy).item())


def save_summary_plot(rows, out_dir: str):
    k_values = [r["K"] for r in rows]
    means = [r["mmd2_mean"] for r in rows]
    se_values = [r["mmd2_se"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.errorbar(k_values, means, yerr=se_values, marker="o", capsize=3, label="BridgeTwist")
    ax.set_xscale("log")
    ax.set_xlabel("Number of particles K")
    ax.set_ylabel("MMD^2")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "toy_mmd_vs_k.pdf"))
    plt.close(fig)


def save_scatter(result: dict, out_dir: str, n: int = 2000):
    gt = sample_ground_truth(n)
    samples = result["samples"]
    if samples.size(0) > n:
        samples = samples[torch.randperm(samples.size(0))[:n]]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), sharex=True, sharey=True)
    axes[0].scatter(gt[:, 0], gt[:, 1], s=4, alpha=0.45)
    axes[0].set_title("Analytic posterior")
    axes[1].scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.45)
    axes[1].set_title("BridgeTwist")
    for ax in axes:
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "toy_scatter_bridge_twist.pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="BridgeTwist toy Gaussian-mixture experiment")
    parser.add_argument("--K_list", type=int, nargs="+", default=[100, 200, 500, 1000, 2000])
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--n_repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/toy_example/figures")
    parser.add_argument("--results_dir", type=str, default="results/toy_example")
    args = parser.parse_args()

    run_timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_dir = os.path.join(args.save_dir, run_timestamp)
    results_dir = os.path.join(args.results_dir, run_timestamp)
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    rows = []
    last_result = None
    for K in args.K_list:
        values, esses = [], []
        for rep in range(args.n_repeats):
            torch.manual_seed(args.seed + 1000 * K + rep)
            result = bridge_twist_sample(
                K=K,
                N=args.N,
            )
            gt = sample_ground_truth(K)
            values.append(mmd2_rbf(result["samples"], gt))
            esses.append(float(result["ess"]))
            last_result = result
        values_t = torch.tensor(values)
        esses_t = torch.tensor(esses)
        se_denom = math.sqrt(args.n_repeats)
        mmd2_spread = values_t.std(unbiased=False)
        ess_spread = esses_t.std(unbiased=False)
        row = {
            "method": "bridge_twist",
            "K": K,
            "N": args.N,
            "mmd2_mean": float(values_t.mean().item()),
            "mmd2_se": float((mmd2_spread / se_denom).item()),
            "ess_mean": float(esses_t.mean().item()),
            "ess_se": float((ess_spread / se_denom).item()),
        }
        rows.append(row)
        print(
            f"K={K}: MMD^2={row['mmd2_mean']:.6f}+/-{row['mmd2_se']:.6f}, "
            f"ESS={row['ess_mean']:.1f}"
        )

    payload = {
        "config": vars(args),
        "timestamp": run_timestamp,
        "summary": rows,
    }
    with open(os.path.join(results_dir, "toy_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    save_summary_plot(rows, figure_dir)
    if last_result is not None:
        save_scatter(last_result, figure_dir)
    print(f"Saved results to {results_dir}")
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()
