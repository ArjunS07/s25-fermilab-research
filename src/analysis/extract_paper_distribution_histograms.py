#!/usr/bin/env python3
"""Reduce large saved sample tensors to small, local-plot-ready histograms."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/mnt/data/output")
OUT = Path("/mnt/data/paper_figures/leftjen_2026-09-11/data/distribution_histograms.json")
TEST_DATA = ROOT / "2026-08-21_10-04-25--cf87d826-9bab-45fb-b52e-02c9a60af5b8-gqt30-lnet-h-cfg-994k/data/x_test.pkl"
SELECTED = {
    "g": (0, 0.0, ROOT / "2026-08-25_02-02-13--46b37f78-4a56-4b6d-bd1c-651dfaac0c27-paper30-gqt994k-cfg-w000-eval/eval"),
    "q": (1, 0.25, ROOT / "2026-08-25_22-18-31--a31c196a-5009-42a1-82e9-a9696e6146ee-paper30-gqt994k-cfg-w025-eval/eval"),
    "t": (2, 0.5, ROOT / "2026-08-25_20-59-00--3a48a5f7-5493-4335-be28-c301daee932c-paper30-gqt994k-cfg-w050-eval/eval"),
}


def cartesian_to_relative(p4: torch.Tensor) -> torch.Tensor:
    """Convert (E, px, py, pz) particles to JetNet relative eta/phi/pT."""
    energy, px, py, pz = p4.unbind(dim=-1)
    pt = torch.sqrt(px.square() + py.square())
    particle_eta = torch.asinh(pz / pt.clamp_min(1e-12))
    particle_phi = torch.atan2(py, px)

    total = p4.sum(dim=1)
    jet_px, jet_py, jet_pz = total[:, 1], total[:, 2], total[:, 3]
    jet_pt = torch.sqrt(jet_px.square() + jet_py.square()).clamp_min(1e-12)
    jet_eta = torch.asinh(jet_pz / jet_pt)
    jet_phi = torch.atan2(jet_py, jet_px)

    eta_rel = particle_eta - jet_eta[:, None]
    phi_rel = torch.atan2(
        torch.sin(particle_phi - jet_phi[:, None]),
        torch.cos(particle_phi - jet_phi[:, None]),
    )
    pt_rel = pt / jet_pt[:, None]
    relative = torch.stack((eta_rel, phi_rel, pt_rel), dim=-1)
    return relative * (p4.abs().sum(dim=-1) > 0).unsqueeze(-1)


def relative_mass_from_rel(rel: torch.Tensor) -> np.ndarray:
    eta, phi, pt = rel[..., 0], rel[..., 1], rel[..., 2]
    real = pt > 0
    px = pt * torch.cos(phi) * real
    py = pt * torch.sin(phi) * real
    pz = pt * torch.sinh(eta) * real
    energy = pt * torch.cosh(eta) * real
    total = torch.stack((energy.sum(1), px.sum(1), py.sum(1), pz.sum(1)), dim=-1)
    mass = torch.sqrt(torch.clamp(total[:, 0].square() - total[:, 1:].square().sum(-1), min=0))
    jet_pt = torch.sqrt(total[:, 1].square() + total[:, 2].square()).clamp_min(1e-12)
    return (mass / jet_pt).numpy()


def flatten_real(rel: torch.Tensor, feature: int) -> np.ndarray:
    real = rel[..., 2] > 0
    values = rel[..., feature][real]
    values = values[torch.isfinite(values)]
    return values.numpy()


def histogram_pair(test: np.ndarray, generated: np.ndarray, feature: str) -> dict:
    test = test[np.isfinite(test)]
    generated = generated[np.isfinite(generated)]
    joined = np.concatenate((test, generated))
    if feature in {"eta_rel", "phi_rel"}:
        bound = float(np.quantile(np.abs(joined), 0.999))
        lo, hi = -bound, bound
    else:
        lo, hi = 0.0, float(np.quantile(joined, 0.999))
    edges = np.linspace(lo, hi, 71)
    test_density, _ = np.histogram(test, bins=edges, density=True)
    generated_density, _ = np.histogram(generated, bins=edges, density=True)
    return {
        "edges": edges.tolist(),
        "test_density": test_density.tolist(),
        "generated_density": generated_density.tolist(),
        "n_test": int(test.size),
        "n_generated": int(generated.size),
        "display_quantile": 0.999,
    }


def main() -> None:
    with TEST_DATA.open("rb") as handle:
        test_data = pickle.load(handle)
    test_particles, test_jets = test_data[:]
    test_rel_all = test_particles[..., :3].cpu()
    test_rel_all *= test_particles[..., 3:4].cpu()
    test_types = test_jets[:, 4].long().cpu()

    payload = {"format_version": 1, "class_order": ["g", "q", "t"], "classes": {}}
    for cls, (class_index, guidance, run) in SELECTED.items():
        samples = torch.load(run / "samples.pt", map_location="cpu", weights_only=False)
        bundle = torch.load(run / "replay_bundle.pt", map_location="cpu", weights_only=False)
        generated_types = bundle["generated_jet_attrs"][:, :5].argmax(dim=-1).long()
        finite = torch.isfinite(samples).all(dim=-1).all(dim=-1)
        selected = finite & (generated_types == class_index)
        generated = samples[selected]
        generated_rel = cartesian_to_relative(generated)

        test_rel = test_rel_all[test_types == class_index]
        features = {
            "eta_rel": (flatten_real(test_rel, 0), flatten_real(generated_rel, 0)),
            "phi_rel": (flatten_real(test_rel, 1), flatten_real(generated_rel, 1)),
            "pt_rel": (flatten_real(test_rel, 2), flatten_real(generated_rel, 2)),
            "relative_mass": (relative_mass_from_rel(test_rel), relative_mass_from_rel(generated_rel)),
        }
        payload["classes"][cls] = {
            "guidance_weight": guidance,
            "source_run": str(run),
            "histograms": {name: histogram_pair(*values, feature=name)
                           for name, values in features.items()},
        }
        del samples, bundle, generated, generated_rel

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Distribution histograms written to {OUT}")


if __name__ == "__main__":
    main()
