"""BridgeTwist image experiments."""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image


_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = Path(_SRC_DIR).parent
sys.path.insert(0, _SRC_DIR)

from utilities import systematic_resample
from smc_sb_image import (
    ImageClassifierLikelihood,
    InpaintingLikelihood,
    adapt_image_batch_to_shape,
    create_half_mask,
    create_random_mask,
    image_smc_sb_sample,
    load_image_classifier,
    load_sb_image_model,
)


DEFAULTS = {
    "mnist": {
        "checkpoint": os.path.join(_PROJECT_ROOT, "checkpoints", "mnist_ckpt.pt"),
        "class_cond": {"grad_clip_norm": 20.0},
        "inpainting_half": {"proposal_grad_scale": 2.0, "sigma_y": 0.05},
        "inpainting_random50": {"proposal_grad_scale": 2.0, "sigma_y": 0.05},
    },
    "cifar10": {
        "checkpoint": os.path.join(_PROJECT_ROOT, "checkpoints", "cifar10_ckpt.pt"),
        "class_cond": {"grad_clip_norm": 50.0},
        "inpainting_half": {"proposal_grad_scale": 2.0},
        "inpainting_random50": {"proposal_grad_scale": 2.0},
    },
    "celebahq": {
        "checkpoint": os.path.join(_PROJECT_ROOT, "checkpoints", "celebahq_ckpt.pt"),
        "text_cond": {"K": 25, "grad_batch_size": 4, "grad_clip_norm": 50.0},
    },
}


def get_dataset_image_shape(dataset):
    if dataset == "mnist":
        return (1, 28, 28)
    if dataset == "cifar10":
        return (3, 32, 32)
    raise ValueError(f"Unknown dataset: {dataset}")


def get_classifier_path(dataset, override=None):
    if override:
        return os.path.abspath(override)
    name = "classifier_mnist.pth.tar" if dataset == "mnist" else "classifier_cifar10.pth.tar"
    return os.path.join(_PROJECT_ROOT, "checkpoints", name)


def get_checkpoint_path(args):
    if args.checkpoint:
        return os.path.abspath(args.checkpoint)
    return DEFAULTS[args.dataset]["checkpoint"]


def get_task_defaults(dataset, task, mask_type=None):
    dataset_defaults = DEFAULTS[dataset]
    if task in ("class_cond", "text_cond"):
        return dict(dataset_defaults[task])
    if mask_type is None:
        return {}
    return dict(dataset_defaults.get(f"inpainting_{mask_type}", {}))


def apply_task_defaults(args, explicit_keys, mask_type=None):
    defaults = get_task_defaults(args.dataset, args.task, mask_type=mask_type)
    for key, value in defaults.items():
        if key not in explicit_keys:
            setattr(args, key, value)


def collect_explicit_cli_keys(argv):
    keys = set()
    for token in argv[1:]:
        if token.startswith("--"):
            keys.add(token[2:].split("=", 1)[0].replace("-", "_"))
    return keys


def make_run_dirs(dataset, task):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"bridge_twist_{timestamp}"
    run_dir = os.path.join(_PROJECT_ROOT, "results", "image_experiments", f"{dataset}_{task}", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_name, run_dir


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)


def load_dataset(dataset, train=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])
    root = os.path.join(_PROJECT_ROOT, "data")
    if dataset == "mnist":
        return datasets.MNIST(root, train=train, download=True, transform=transform)
    if dataset == "cifar10":
        return datasets.CIFAR10(root, train=train, download=True, transform=transform)
    raise ValueError(f"Unknown dataset: {dataset}")


def collect_real_images(dataset, classes, max_per_class):
    data = load_dataset(dataset, train=True)
    buckets = {c: [] for c in classes}
    for image, label in data:
        label = int(label)
        if label in buckets and len(buckets[label]) < max_per_class:
            buckets[label].append(image)
        if all(len(v) >= max_per_class for v in buckets.values()):
            break
    return {c: torch.stack(buckets[c], dim=0) for c in classes if buckets[c]}


def load_external_classifier_if_requested(dataset, device, enabled):
    if not enabled:
        return None
    from external_classifiers import load_external_classifier

    return load_external_classifier(dataset, device)


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean())


def run_class_conditional(args, run_dir, sb_model, classifier_fn, device, kid_features_fn=None):
    from evaluation import compute_classification_metrics, compute_kid

    classes = args.classes or list(range(10))
    y_dummy = torch.tensor(0, device=device)
    external_classifier_fn = load_external_classifier_if_requested(args.dataset, device, args.external_classifier)
    real_images = collect_real_images(args.dataset, classes, args.K) if args.compute_kid else {}

    results = {"per_class": {str(c): [] for c in classes}, "summary": {}}
    sample_dir = None
    if args.sample_grid_4x4:
        sample_dir = os.path.join(run_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)

    for repeat in range(args.n_repeats):
        seed = args.seed_base + repeat
        torch.manual_seed(seed)
        np.random.seed(seed)
        for target_class in classes:
            print(
                f"  [class_cond] repeat {repeat + 1}/{args.n_repeats} "
                f"class {target_class}: sampling K={args.K} N={args.N}...",
                flush=True,
            )
            likelihood = ImageClassifierLikelihood(classifier_fn, target_class, sb_model.image_shape)
            t0 = time.time()
            result = image_smc_sb_sample(
                sb_model,
                likelihood,
                y_dummy,
                K=args.K,
                N=args.N,
                g_init=args.g_init,
                g_schedule=args.g_schedule,
                tempering=args.tempering,
                grad_clip_norm=args.grad_clip_norm,
                resample_start_frac=args.resample_start_frac,
                ess_threshold=args.ess_threshold,
                proposal_grad_scale=args.proposal_grad_scale,
                grad_batch_size=args.grad_batch_size,
                return_ess_history=args.save_ess,
            )
            elapsed = time.time() - t0
            samples = result["samples"]
            weights = result["weights"]
            acc_info = compute_classification_metrics(
                samples,
                weights,
                classifier_fn,
                target_class,
                device,
                external_classifier_fn=external_classifier_fn,
            )

            kid_mean, kid_std = None, None
            if args.compute_kid and target_class in real_images:
                kid_mean, kid_std = compute_kid(
                    real_images[target_class],
                    acc_info["resampled_samples"].detach().cpu(),
                    device,
                    dataset=args.dataset,
                    features_fn=kid_features_fn,
                    subset_size=min(args.K, real_images[target_class].size(0)),
                    subsets=args.kid_subsets,
                )

            row = {
                "dataset": args.dataset,
                "task": "class_cond",
                "repeat": repeat,
                "target_class": target_class,
                "K": args.K,
                "N": args.N,
                "likelihood_acc": acc_info["likelihood_acc"],
                "external_acc": acc_info["external_acc"],
                "kid_mean": kid_mean,
                "kid_std": kid_std,
                "ess": float(result["ess"]),
                "time": elapsed,
            }
            if args.save_ess:
                row["ess_history"] = result["ess_history"].detach().cpu().tolist()
            results["per_class"][str(target_class)].append(row)
            print(
                f"  [class_cond] repeat {repeat + 1}/{args.n_repeats} "
                f"class {target_class}: acc={acc_info['likelihood_acc']:.3f} "
                f"ess={float(result['ess']):.1f} time={elapsed:.1f}s",
                flush=True,
            )

            if args.sample_grid_4x4 and samples.size(0) >= 16:
                grid_generator = torch.Generator(device="cpu")
                grid_seed = int(args.seed_base) + int(repeat) * 1000 + int(target_class)
                grid_generator.manual_seed(grid_seed)
                vis_indices = torch.randperm(
                    acc_info["resampled_samples"].size(0),
                    generator=grid_generator,
                )[:16].to(acc_info["resampled_samples"].device)
                grid = acc_info["resampled_samples"].index_select(0, vis_indices).detach().cpu()
                save_image((grid + 1.0) / 2.0, os.path.join(sample_dir, f"class{target_class}_rep{repeat}.png"), nrow=4)

    all_rows = [row for rows in results["per_class"].values() for row in rows]
    for key in ["likelihood_acc", "external_acc", "kid_mean", "ess", "time"]:
        vals = [r[key] for r in all_rows if r[key] is not None]
        if vals:
            results["summary"][f"{key}_mean"] = summarize(vals)
    return results


def collect_inpainting_images(dataset, classes, per_class):
    data = load_dataset(dataset, train=False)
    buckets = {c: [] for c in classes}
    for image, label in data:
        label = int(label)
        if label in buckets and len(buckets[label]) < per_class:
            buckets[label].append(image)
        if all(len(v) >= per_class for v in buckets.values()):
            break
    items = []
    for c in classes:
        for i, image in enumerate(buckets[c]):
            items.append((c, i, image))
    return items


def run_inpainting(args, run_dir, sb_model, device, explicit_keys):
    from evaluation import compute_rmse, compute_ssim_batch

    classes = args.classes or list(range(10))
    test_images = collect_inpainting_images(args.dataset, classes, args.n_test_images)
    results = {"per_mask": {m: [] for m in args.mask_types}, "summary": {}}
    sample_dir = None
    if args.save_inpainting_examples:
        sample_dir = os.path.join(run_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)

    native_shape = get_dataset_image_shape(args.dataset)
    y_dummy = torch.tensor(0, device=device)

    for mask_type in args.mask_types:
        apply_task_defaults(args, explicit_keys, mask_type=mask_type)
        for repeat in range(args.n_repeats):
            seed = args.seed_base + repeat
            torch.manual_seed(seed)
            np.random.seed(seed)
            for label, image_idx, x_native in test_images:
                print(
                    f"  [inpainting/{mask_type}] repeat {repeat + 1}/{args.n_repeats} "
                    f"label {label} img {image_idx}: sampling K={args.K} N={args.N}...",
                    flush=True,
                )
                x_model = adapt_image_batch_to_shape(x_native.unsqueeze(0), sb_model.image_shape, pad_value=-1.0).squeeze(0).to(device)
                if mask_type == "half":
                    mask = create_half_mask(sb_model.image_shape).to(device)
                elif mask_type == "random50":
                    mask = create_random_mask(sb_model.image_shape, keep_ratio=0.5, seed=seed + image_idx).to(device)
                else:
                    raise ValueError(f"Unknown mask type: {mask_type}")

                likelihood = InpaintingLikelihood(
                    y_obs=x_model * mask,
                    mask=mask,
                    noise_std=args.sigma_y,
                    image_shape=sb_model.image_shape,
                )
                t0 = time.time()
                result = image_smc_sb_sample(
                    sb_model,
                    likelihood,
                    y_dummy,
                    K=args.K,
                    N=args.N,
                    g_init=args.g_init,
                    g_schedule=args.g_schedule,
                    tempering=args.tempering,
                    grad_clip_norm=args.grad_clip_norm,
                    resample_start_frac=args.resample_start_frac,
                    ess_threshold=args.ess_threshold,
                    proposal_grad_scale=args.proposal_grad_scale,
                    grad_batch_size=args.grad_batch_size,
                    return_ess_history=args.save_ess,
                )
                elapsed = time.time() - t0
                samples = result["samples"]
                weights = result["weights"]
                best_idx = weights.argmax().item()
                resampled = samples[systematic_resample(weights)]

                eval_samples = adapt_image_batch_to_shape(samples.detach().cpu(), native_shape, pad_value=-1.0)
                eval_resampled = adapt_image_batch_to_shape(resampled.detach().cpu(), native_shape, pad_value=-1.0)
                eval_mask = adapt_image_batch_to_shape(mask.detach().cpu(), native_shape, pad_value=1.0)
                x_target = x_native.detach().cpu()

                rmse_mean = compute_rmse(eval_resampled, x_target, eval_mask).mean().item()
                rmse_best = compute_rmse(eval_samples[best_idx : best_idx + 1], x_target, eval_mask).item()
                ssim_mean = compute_ssim_batch(eval_resampled, x_target.unsqueeze(0)).mean().item()
                ssim_best = compute_ssim_batch(eval_samples[best_idx : best_idx + 1], x_target.unsqueeze(0)).item()

                row = {
                    "dataset": args.dataset,
                    "task": "inpainting",
                    "mask_type": mask_type,
                    "repeat": repeat,
                    "label": label,
                    "image_idx": image_idx,
                    "K": args.K,
                    "N": args.N,
                    "sigma_y": args.sigma_y,
                    "rmse_mean": rmse_mean,
                    "rmse_best": rmse_best,
                    "ssim_mean": ssim_mean,
                    "ssim_best": ssim_best,
                    "ess": float(result["ess"]),
                    "time": elapsed,
                }
                if args.save_ess:
                    row["ess_history"] = result["ess_history"].detach().cpu().tolist()
                results["per_mask"][mask_type].append(row)
                print(
                    f"  [inpainting/{mask_type}] repeat {repeat + 1}/{args.n_repeats} "
                    f"label {label} img {image_idx}: rmse={rmse_mean:.4f} "
                    f"ssim={ssim_mean:.3f} ess={float(result['ess']):.1f} time={elapsed:.1f}s",
                    flush=True,
                )

                if args.save_inpainting_examples and repeat == 0 and image_idx == 0:
                    observed = (x_model * mask + (-1.0) * (1.0 - mask)).detach().cpu()
                    best = samples[best_idx].detach().cpu()
                    panel = torch.stack([
                        adapt_image_batch_to_shape(observed, native_shape, pad_value=-1.0),
                        adapt_image_batch_to_shape(best, native_shape, pad_value=-1.0),
                        x_target,
                    ])
                    save_image((panel + 1.0) / 2.0, os.path.join(sample_dir, f"{mask_type}_label{label}.png"), nrow=3)

    for mask_type, rows in results["per_mask"].items():
        mask_summary = {}
        for key in ["rmse_mean", "rmse_best", "ssim_mean", "ssim_best", "ess", "time"]:
            vals = [r[key] for r in rows]
            mask_summary[f"{key}_mean"] = summarize(vals)
        results["summary"][mask_type] = mask_summary
    return results


def load_prompts(path):
    prompts = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("The prompts file must contain a nonempty list")
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            raise ValueError(f"Prompt {index} must be an object")
        positive = prompt.get("positive_text")
        negative = prompt.get("negative_texts")
        if isinstance(positive, str):
            positive = [positive]
        if isinstance(negative, str):
            negative = [negative]
        if not isinstance(positive, list) or not positive or any(
            not isinstance(text, str) or not text.strip() for text in positive
        ):
            raise ValueError(f"Prompt {index} requires nonempty positive_text descriptions")
        if not isinstance(negative, list) or not negative or any(
            not isinstance(text, str) or not text.strip() for text in negative
        ):
            raise ValueError(f"Prompt {index} requires nonempty negative_texts")
        prompt["positive_text"] = positive
        prompt["negative_texts"] = negative
    return prompts


def run_text_trial(sb_model, likelihood, args):
    result = image_smc_sb_sample(
        sb_model, likelihood, torch.tensor(0, device=sb_model.device),
        K=args.K, N=args.N,
        g_init=args.g_init,
        ess_threshold=args.ess_threshold,
        grad_clip_norm=args.grad_clip_norm,
        g_schedule=args.g_schedule,
        resample_start_frac=args.resample_start_frac,
        tempering=args.tempering,
        grad_batch_size=args.grad_batch_size,
        proposal_grad_scale=args.proposal_grad_scale,
        delta_g=args.delta_g,
        proposal_cov_epsilon=args.proposal_cov_epsilon,
    )
    samples = result["samples"]
    weights = result["weights"]
    if not torch.isfinite(samples).all() or not torch.isfinite(weights).all():
        raise RuntimeError("Non-finite endpoint samples or weights")
    # Resample on the CPU, then transfer indices to the sample device.
    indices = systematic_resample(weights.cpu()).to(samples.device)
    resampled = samples[indices]
    clip_l = likelihood.clip_l_values(resampled)
    if not torch.isfinite(clip_l).all():
        raise RuntimeError("Non-finite CLIP_L values")
    return result, indices, resampled, clip_l


def text_result_metadata(args, sigma):
    """Record the model files and parameters used for text generation."""
    parameter_names = (
        "K", "N", "n_repeats", "seed_base", "prompt_idx", "grad_batch_size",
        "g_init", "g_schedule", "tempering", "grad_clip_norm",
        "resample_start_frac", "ess_threshold", "proposal_grad_scale",
        "clip_temperature", "delta_g", "proposal_cov_epsilon",
    )
    return {
        "method": "bridge_twist", "dataset": "celebahq", "task": "text_cond",
        "checkpoint": Path(args.checkpoint).name,
        "clip_model": Path(args.clip_model_path).name,
        "prompts_file": Path(args.prompts_file).name,
        "sigma": sigma,
        "args": {name: getattr(args, name) for name in parameter_names},
    }


def write_results(path, payload):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_text_conditional(args, parser):
    from celebahq_clip import CLIPTextLikelihood

    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run this experiment in a CUDA-enabled environment")
    for path in (args.checkpoint, args.clip_model_path):
        if not path.is_file():
            parser.error(f"Missing checkpoint: {path}. Download the required checkpoint group first.")
    if not args.prompts_file.is_file():
        parser.error(f"Missing prompts file: {args.prompts_file}. Set --prompts_file to a JSON prompt file.")
    prompts = load_prompts(args.prompts_file)
    if args.prompt_idx < -1 or args.prompt_idx >= len(prompts):
        parser.error(f"--prompt_idx must be -1 or between 0 and {len(prompts) - 1}")
    indices = list(range(len(prompts))) if args.prompt_idx == -1 else [args.prompt_idx]

    sb_model = load_sb_image_model(str(args.checkpoint), device=args.device)
    if sb_model.checkpoint_info["dataset"] != "celebahq" or sb_model.image_shape != (3, 256, 256):
        raise ValueError("This runner requires a CelebA-HQ 256x256 SB checkpoint")
    if not math.isclose(sb_model.sigma, 0.25, abs_tol=1e-12):
        raise ValueError("This release uses the CelebA-HQ checkpoint with Brownian sigma=0.25")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prompt = prompts[indices[0]]
    likelihood = CLIPTextLikelihood(
        positive_texts=prompt["positive_text"], negative_texts=prompt["negative_texts"],
        image_shape=sb_model.image_shape, clip_model_path=str(args.clip_model_path),
        device=args.device, temperature=args.clip_temperature,
    )
    likelihood._load_clip()

    output_dir = args.output_dir or _PROJECT_ROOT / "results/celebahq_t2i" / datetime.now(timezone.utc).strftime("bridge_twist_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    results = {
        "per_prompt": {str(index): [] for index in indices},
        "summary": {},
        "metadata": text_result_metadata(args, sb_model.sigma),
    }
    result_path = output_dir / "results.json"
    write_results(result_path, results)
    for prompt_index in indices:
        prompt = prompts[prompt_index]
        likelihood.set_prompts(prompt["positive_text"], prompt["negative_texts"])
        for repeat in range(args.n_repeats):
            seed = args.seed_base + prompt_index * 1000 + repeat
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            result, selected, resampled, values = run_text_trial(sb_model, likelihood, args)
            trial = {
                "repeat": repeat, "seed": seed,
                "CLIP_L": float(values.mean().item()),
            }
            if args.save_samples:
                sample_dir = output_dir / f"prompt_{prompt_index:02d}"
                sample_dir.mkdir(exist_ok=True)
                tensor_path = sample_dir / f"repeat_{repeat:02d}.pt"
                torch.save({
                    "samples": result["samples"].cpu(), "weights": result["weights"].cpu(),
                    "resampled_indices": selected.cpu(), "CLIP_L_values": values.cpu(),
                    "seed": seed, "prompt_idx": prompt_index, "repeat_idx": repeat,
                }, tensor_path)
                save_image((resampled.cpu().clamp(-1, 1) + 1) / 2,
                           sample_dir / f"repeat_{repeat:02d}.png", nrow=math.ceil(math.sqrt(args.K)))
                trial["samples_path"] = str(tensor_path.relative_to(output_dir))
            results["per_prompt"][str(prompt_index)].append(trial)
            write_results(result_path, results)
            print(f"prompt={prompt_index:02d} repeat={repeat} seed={seed} CLIP_L={trial['CLIP_L']:.6f}", flush=True)
    trial_means = [trial["CLIP_L"] for rows in results["per_prompt"].values() for trial in rows]
    results["summary"]["CLIP_L_mean"] = float(np.mean(trial_means))
    if len(trial_means) > 1:
        results["summary"]["CLIP_L_std_across_trials"] = float(np.std(trial_means, ddof=1))
    write_results(result_path, results)
    print(f"Results: {result_path}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="BridgeTwist image experiments", allow_abbrev=False,
    )
    general = parser.add_argument_group("Experiment and device")
    general.add_argument("--dataset", choices=["mnist", "cifar10", "celebahq"], default="cifar10")
    general.add_argument("--task", choices=["class_cond", "inpainting", "text_cond"], default="class_cond")
    general.add_argument("--gpu", type=int, default=0)
    general.add_argument("--device", default=None, help="Explicit device, such as cuda:0 or cpu; overrides --gpu")
    general.add_argument("--checkpoint", type=str, default=None)
    general.add_argument("--n_repeats", type=int, default=1, help="Repetitions: default 1")
    general.add_argument("--seed_base", type=int, default=2026)

    sampling = parser.add_argument_group("BridgeTwist sampling")
    sampling.add_argument("--K", type=int, default=16, help="Number of particles: default 16, or 25 for CelebA-HQ")
    sampling.add_argument("--N", type=int, default=800, help="Number of time steps")
    sampling.add_argument("--grad_batch_size", type=int, default=160, help="Gradient batch size: default 160, or 4 for CelebA-HQ")
    sampling.add_argument("--g_init", type=float, default=4.0)
    sampling.add_argument("--g_schedule", choices=["constant", "cosine", "sigma_t"], default="sigma_t")
    sampling.add_argument("--tempering", choices=["none", "linear", "quadratic", "cubic"], default="linear")
    sampling.add_argument("--grad_clip_norm", type=float, default=100.0)
    sampling.add_argument("--resample_start_frac", type=float, default=0.1)
    sampling.add_argument("--ess_threshold", type=float, default=0.5)
    sampling.add_argument("--proposal_grad_scale", type=float, default=1.0)

    image_options = parser.add_argument_group("MNIST/CIFAR-10 options")
    image_options.add_argument("--classes", type=int, nargs="*", default=None)
    image_options.add_argument("--save_ess", action="store_true", help="Save per-step ESS values in each trial's result")

    class_options = parser.add_argument_group("Class-conditional sampling")
    class_options.add_argument("--classifier_checkpoint", type=str, default=None)
    class_options.add_argument("--compute_kid", action="store_true")
    class_options.add_argument("--kid_subsets", type=int, default=20)
    class_options.add_argument("--external_classifier", action="store_true")
    class_options.add_argument("--sample_grid_4x4", action="store_true")

    inpainting_options = parser.add_argument_group("Inpainting")
    inpainting_options.add_argument("--sigma_y", type=float, default=0.1)
    inpainting_options.add_argument("--n_test_images", type=int, default=1)
    inpainting_options.add_argument("--mask_types", nargs="+", choices=["half", "random50"], default=["random50"])
    inpainting_options.add_argument("--save_inpainting_examples", action="store_true")

    text_options = parser.add_argument_group("CelebA-HQ text_cond options")
    text_options.add_argument("--clip_model_path", default=str(_PROJECT_ROOT / "checkpoints/open_clip_pytorch_model.bin"))
    text_options.add_argument(
        "--prompts_file", default=str(_PROJECT_ROOT / "configs" / "celebahq_prompts.json"),
        help="Text prompts JSON (default: configs/celebahq_prompts.json)",
    )
    text_options.add_argument("--prompt_idx", type=int, default=-1, help="Zero-based prompt index; -1 runs all prompts")
    text_options.add_argument("--clip_temperature", type=float, default=100.0)
    text_options.add_argument("--delta_g", type=float, default=1e-6)
    text_options.add_argument("--proposal_cov_epsilon", type=float, default=1e-6)
    text_options.add_argument("--save_samples", action="store_true", help="Save endpoint tensors, resampling indices, and PNG grids")
    text_options.add_argument("--output_dir", default=None, help="New output directory; an existing directory is never overwritten")
    return parser


def validate_args(parser, args):
    """Reject invalid sampling settings before loading data or models."""
    for name in ("K", "N", "n_repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.grad_batch_size < 0:
        parser.error("--grad_batch_size must be nonnegative")
    for name in ("g_init", "grad_clip_norm", "delta_g", "proposal_cov_epsilon", "clip_temperature"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name} must be finite and positive")
    if not math.isfinite(args.proposal_grad_scale) or args.proposal_grad_scale < 0:
        parser.error("--proposal_grad_scale must be finite and nonnegative")
    if not 0 <= args.ess_threshold <= 1 or not 0 <= args.resample_start_frac <= 1:
        parser.error("ESS threshold and resampling start fraction must be in [0, 1]")
    if args.task == "class_cond" and args.kid_subsets < 1:
        parser.error("--kid_subsets must be positive")
    if args.task == "inpainting":
        if args.n_test_images < 1:
            parser.error("--n_test_images must be positive")
        if not math.isfinite(args.sigma_y) or args.sigma_y <= 0:
            parser.error("--sigma_y must be finite and positive")
    if args.classes is not None and any(label < 0 or label > 9 for label in args.classes):
        parser.error("--classes must contain labels from 0 to 9")


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    explicit_keys = collect_explicit_cli_keys(["run_full_image_experiments.py", *argv])
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.dataset == "celebahq") != (args.task == "text_cond"):
        parser.error("CelebA-HQ requires --task text_cond; MNIST/CIFAR-10 support class_cond and inpainting")

    task_options = {
        "class_cond": {
            "classifier_checkpoint", "classes", "compute_kid", "kid_subsets",
            "external_classifier", "sample_grid_4x4", "save_ess",
        },
        "inpainting": {
            "classes", "sigma_y", "n_test_images", "mask_types",
            "save_inpainting_examples", "save_ess",
        },
        "text_cond": {
            "clip_model_path", "prompts_file", "prompt_idx", "clip_temperature",
            "delta_g", "proposal_cov_epsilon", "save_samples", "output_dir",
        },
    }
    unsupported = explicit_keys & (set().union(*task_options.values()) - task_options[args.task])
    if unsupported:
        options = ", ".join("--" + name for name in sorted(unsupported))
        parser.error(f"Unsupported options for --task {args.task}: {options}")

    apply_task_defaults(args, explicit_keys)
    validate_args(parser, args)
    if args.task == "text_cond":
        args.checkpoint = Path(get_checkpoint_path(args))
        args.clip_model_path = Path(args.clip_model_path)
        args.prompts_file = Path(args.prompts_file)
        if args.output_dir is not None:
            args.output_dir = Path(args.output_dir)
        args.device = args.device or f"cuda:{args.gpu}"
    return parser, args, explicit_keys


def main():
    parser, args, explicit_keys = parse_args()
    if args.task == "text_cond":
        run_text_conditional(args, parser)
        return

    device = args.device or (f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    run_name, run_dir = make_run_dirs(args.dataset, args.task)
    ckpt_path = get_checkpoint_path(args)

    print(f"Run name: {run_name}")
    print(f"Result dir: {run_dir}")
    print(f"Device: {device}")
    print(f"SB checkpoint: {ckpt_path}")

    sb_model = load_sb_image_model(ckpt_path, device)
    classifier_path = None
    classifier_fn = None
    kid_features_fn = None
    has_mnist_classifier_features = args.task == "class_cond" and args.dataset == "mnist"
    if args.task == "class_cond":
        classifier_path = get_classifier_path(args.dataset, args.classifier_checkpoint)
        print(f"Classifier checkpoint: {classifier_path}")
        if has_mnist_classifier_features:
            classifier_fn, kid_features_fn = load_image_classifier(
                classifier_path,
                args.dataset,
                device,
                return_features=True,
            )
        else:
            classifier_fn = load_image_classifier(classifier_path, args.dataset, device)

    metadata = {
        "run_name": run_name,
        "method": "bridge_twist",
        "dataset": args.dataset,
        "task": args.task,
        "checkpoint": os.path.abspath(ckpt_path),
        "classifier": os.path.abspath(classifier_path) if classifier_path else None,
        "kid_features": "likelihood_classifier_pre_linear" if args.compute_kid and has_mnist_classifier_features else None,
        "bridge_twist_defaults": {
            "delta_g": 1e-6,
            "proposal_cov_epsilon": 1e-6,
        },
        "args": vars(args),
    }
    save_json(os.path.join(run_dir, "config.json"), metadata)

    if args.task == "class_cond":
        results = run_class_conditional(args, run_dir, sb_model, classifier_fn, device, kid_features_fn=kid_features_fn)
    else:
        results = run_inpainting(args, run_dir, sb_model, device, explicit_keys)

    results["metadata"] = metadata
    save_json(os.path.join(run_dir, "results.json"), results)
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
