"""Deterministic statistics for the frozen-versus-fresh path audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from util.mass_shell import parallel_transport
from util.minkowski_utils import dotsq4, normsq4


AUDIT_FORMAT_VERSION = 1


def deterministic_indices(population: int, count: int, seed: int) -> np.ndarray:
    if count <= 0 or count > population:
        raise ValueError("count must be in [1, population]")
    values = np.random.Generator(np.random.PCG64(seed)).choice(
        population, size=count, replace=False
    )
    return np.sort(values.astype(np.int64))


def latin_hypercube_times(count: int, seed: int) -> np.ndarray:
    """One fixed time per target, with exactly uniform occupancy over (0, 1)."""
    order = np.random.Generator(np.random.PCG64(seed)).permutation(count)
    return (order.astype(np.float64) + 0.5) / float(count)


def per_jet_field_metrics(prediction: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor, eps: float = 1e-12) -> dict[str, torch.Tensor]:
    """Invariant per-jet tangent-field diagnostics, averaged over real particles."""
    pred = prediction.to(torch.float64)
    truth = target.to(torch.float64)
    weight = mask.to(torch.float64)
    denom = weight.sum(dim=1).clamp(min=1.0)
    error_sq = (-normsq4(pred - truth)).clamp(min=0.0)
    target_sq = (-normsq4(truth)).clamp(min=0.0)
    prediction_sq = (-normsq4(pred)).clamp(min=0.0)
    inner = -dotsq4(pred, truth)

    def mean_real(value):
        return (value * weight).sum(dim=1) / denom

    loss = mean_real(error_sq)
    target_norm_sq = mean_real(target_sq)
    prediction_norm_sq = mean_real(prediction_sq)
    cosine_particle = inner / torch.sqrt(
        (target_sq * prediction_sq).clamp(min=eps)
    )
    active = (target_sq > eps) & (prediction_sq > eps) & (weight > 0)
    active_count = active.sum(dim=1).clamp(min=1)
    alignment = (cosine_particle * active).sum(dim=1) / active_count
    alignment = torch.where(active.any(dim=1), alignment, torch.zeros_like(alignment))
    return {
        "loss": loss,
        "target_norm_sq": target_norm_sq,
        "prediction_norm_sq": prediction_norm_sq,
        "relative_error": loss / target_norm_sq.clamp(min=eps),
        "alignment": alignment,
    }


def transported_draw_dispersion(states: torch.Tensor, predictions: torch.Tensor,
                                targets: torch.Tensor, mask: torch.Tensor,
                                regulator_mass: float, eps: float = 1e-12) -> torch.Tensor:
    """Per-jet prediction dispersion after transporting K draws to target tangent spaces.

    Shapes are ``states,predictions=(K,B,P,4)``, ``targets=(B,P,4)``.
    """
    if states.shape != predictions.shape or states.ndim != 4:
        raise ValueError("states and predictions must have matching (K,B,P,4) shapes")
    destination = targets.to(torch.float64).unsqueeze(0).expand_as(states)
    moved = parallel_transport(states, destination, predictions, regulator_mass)
    mean = moved.mean(dim=0, keepdim=True)
    variance = (-normsq4(moved - mean)).clamp(min=0.0).mean(dim=0)
    magnitude = (-normsq4(mean.squeeze(0))).clamp(min=0.0)
    weight = mask.to(torch.float64)
    denom = weight.sum(dim=1).clamp(min=1.0)
    variance_jet = (variance * weight).sum(dim=1) / denom
    magnitude_jet = (magnitude * weight).sum(dim=1) / denom
    return variance_jet / magnitude_jet.clamp(min=eps)


def log_gap(fresh: np.ndarray, frozen: np.ndarray, eps: float = 1e-12) -> float:
    fresh = np.asarray(fresh, dtype=np.float64)
    frozen = np.asarray(frozen, dtype=np.float64)
    if fresh.ndim != 2 or frozen.shape != (fresh.shape[0],):
        raise ValueError("fresh must be (N,K) and frozen must be (N,)")
    return float(np.median(np.log((fresh + eps) / (frozen[:, None] + eps))))


def h1_difference(train_fresh: np.ndarray, train_cached: np.ndarray,
                  valid_fresh: np.ndarray, valid_frozen: np.ndarray) -> float:
    return log_gap(train_fresh, train_cached) - log_gap(valid_fresh, valid_frozen)


def cluster_bootstrap_h1(train_fresh: np.ndarray, train_cached: np.ndarray,
                         valid_fresh: np.ndarray, valid_frozen: np.ndarray, *,
                         seed: int, samples: int = 2000) -> dict[str, float]:
    """Bootstrap target jets while retaining all fresh draws inside each cluster."""
    train_fresh, train_cached = np.asarray(train_fresh), np.asarray(train_cached)
    valid_fresh, valid_frozen = np.asarray(valid_fresh), np.asarray(valid_frozen)
    if train_fresh.shape[0] != train_cached.shape[0]:
        raise ValueError("training target counts disagree")
    if valid_fresh.shape[0] != valid_frozen.shape[0]:
        raise ValueError("validation target counts disagree")
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        ti = rng.integers(0, len(train_cached), size=len(train_cached))
        vi = rng.integers(0, len(valid_frozen), size=len(valid_frozen))
        values[index] = h1_difference(
            train_fresh[ti], train_cached[ti], valid_fresh[vi], valid_frozen[vi]
        )
    estimate = h1_difference(train_fresh, train_cached, valid_fresh, valid_frozen)
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "log_effect": float(estimate),
        "ratio": float(np.exp(estimate)),
        "ci95_log_low": float(low),
        "ci95_log_high": float(high),
        "ci95_ratio_low": float(np.exp(low)),
        "ci95_ratio_high": float(np.exp(high)),
        "bootstrap_samples": int(samples),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
