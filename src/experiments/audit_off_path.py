#!/usr/bin/env python3
"""Controlled frozen-versus-fresh exact-ICP path audit for the qualified model."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cache_icp import (_icp_permute_worker, _mass_shell_transport_costs,
                       source_dataset_fingerprint, validate_cache_metadata)
from models.LEFT_JeN import LEFTJeN
from util import jet_attributes
from util.coordinates import (build_reference_vectors, deterministic_jet_phi,
                              transform_rel_particle_coordinates_to_cartesian)
from util.distributions import gen_initial_distribution
from util.mass_shell import (conditional_vector_field, geodesic_interpolant,
                             parallel_transport, project_to_shell,
                             pushforward_to_tangent)
from util.minkowski_utils import normsq4
from util.off_path_audit import (AUDIT_FORMAT_VERSION, cluster_bootstrap_h1,
                                 deterministic_indices, latin_hypercube_times,
                                 per_jet_field_metrics, sha256_file, write_json)
from util.rng import keyed_torch_rng


DEFAULT_SOURCE = ("/mnt/data/output/2026-07-23_04-40-41--"
                  "36429f69-a6a5-49a0-81fd-8f351f5a9c11-g30-massshell-v2")
DEFAULT_CACHE = "/mnt/data/caches/g_p30_v2_massshell_m01_phi-v1-d2/icp_cache.pkl"
CHECKPOINTS = {
    "step_000000_raw": "train/stability_probes/step_000000/checkpoint.pth",
    "step_002000_raw": "train/stability_probes/step_002000/checkpoint.pth",
    "step_080000_raw": "train/stability_probes/step_080000/checkpoint.pth",
    "step_100000_raw": "train/stability_probes/step_100000/checkpoint.pth",
    "step_100000_ema": "train/stability_probes/step_100000/checkpoint.pth",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=DEFAULT_SOURCE)
    parser.add_argument("--cache-path", default=DEFAULT_CACHE)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-targets", type=int, default=4096)
    parser.add_argument("--fresh-draws", type=int, default=4)
    parser.add_argument("--audit-seed", type=int, default=17029)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def _model_from_checkpoint(checkpoint, device, use_ema=False):
    full = checkpoint["full_config"]
    model_cfg = full["model"]
    data_cfg = full["data"]
    model = LEFTJeN(
        max_num_jet_types=jet_attributes.NUM_CLASSES,
        max_particles=data_cfg["num_particles"],
        num_layers=model_cfg["n_layers"],
        hidden_dim=model_cfg["n_hidden"],
        use_residual_update=model_cfg.get("use_residual", False),
        include_pt=True,
        use_reference_vectors=model_cfg.get("use_reference_vectors", False),
        use_node_scalars=model_cfg.get("use_node_scalars", False),
        node_scalar_seed=model_cfg.get("node_scalar_seed", "physics"),
        use_adaln=model_cfg.get("use_adaln", False),
        use_attention=model_cfg.get("use_attention", False),
        backbone=model_cfg["backbone"],
        include_mass_condition=model_cfg.get("include_mass_condition", False),
        num_attention_heads=model_cfg.get("num_attention_heads", 8),
        vector_channels=model_cfg.get("vector_channels", 16),
        regulator_mass=model_cfg["regulator_mass"],
        velocity_readout_init=model_cfg.get("velocity_readout_init", "small_normal"),
    ).to(device)
    key = "ema_state_dict" if use_ema else "model_state_dict"
    model.load_state_dict(checkpoint[key], strict=True)
    model.eval()
    return model


def _select_split(dataset, indices, final_scale, num_particles):
    all_phi = deterministic_jet_phi(len(dataset), seed=42)
    transformed = transform_rel_particle_coordinates_to_cartesian(
        dataset, jet_phi=all_phi
    )[:, :num_particles].clone()
    transformed[..., :4] /= final_scale
    selected = torch.as_tensor(indices, dtype=torch.long)
    particles = transformed[selected]
    jet_info = dataset[:][1][selected].clone()
    jet_info[:, 3] = jet_info[:, 3].clamp(max=num_particles)
    return {
        "x1": particles[..., :4].float(),
        "mask": particles[..., 4].float(),
        "jet_info": jet_info.float(),
        "jet_phi": all_phi[selected].float(),
    }


def _align_exact(prior, target, mask, workers):
    x0 = prior.detach().cpu().numpy().astype(np.float32)
    x1 = target.detach().cpu().numpy().astype(np.float32)
    n_real = mask.sum(dim=1).long().cpu().numpy()
    tasks = [
        (i, x0[i], x1[i], int(n_real[i]), 1, "mass_shell", 0.1,
         "squared_geodesic", "exact_geodesic_icp", 5042)
        for i in range(len(x0))
    ]
    perms = np.empty((len(x0), x0.shape[1]), dtype=np.int32)
    with Pool(processes=workers) as pool:
        for index, perm, _ in pool.imap_unordered(_icp_permute_worker, tasks, chunksize=64):
            perms[index] = perm
    paired = np.take_along_axis(x0, perms[..., None], axis=1)
    paired *= mask.cpu().numpy()[..., None]
    costs, means = _mass_shell_transport_costs(paired, x1, n_real, 0.1)
    return torch.from_numpy(paired).float(), torch.from_numpy(perms), means


def _fresh_bundles(split, draws, audit_seed, split_id, final_scale, workers):
    bundles = []
    for draw in range(draws):
        with keyed_torch_rng(audit_seed, split_id, draw, 0, "cpu"):
            prior = gen_initial_distribution(
                x_1=split["x1"], prior_dist="axis_aligned_per_jet",
                jet_features=split["jet_info"], jet_phi=split["jet_phi"],
                device="cpu", model_scale=final_scale,
            ).float()
        prior *= split["mask"].unsqueeze(-1)
        paired, perm, cost = _align_exact(prior, split["x1"], split["mask"], workers)
        bundles.append({"x0": paired, "perm": perm, "transport_cost": cost})
    return bundles


def _conditions_and_references(split, final_scale, include_mass, device):
    info = split["jet_info"].to(device)
    one_hot = jet_attributes.one_hot_enc_jet_type(info[:, 4].long()).to(device)
    parts = [one_hot, info[:, 3:4], info[:, 1:2] / final_scale]
    if include_mass:
        parts.append(info[:, 2:3] / final_scale)
    conditions = torch.cat(parts, dim=-1)
    references = build_reference_vectors(
        info[:, 0], info[:, 1], final_scale, device,
        jet_phi=split["jet_phi"].to(device),
        jet_mass=(info[:, 2] if include_mass else None),
    )
    return conditions, references


@torch.no_grad()
def _evaluate_bundle(model, split, x0, times, final_scale, regulator_mass,
                     include_mass, batch_size, device):
    conditions, references = _conditions_and_references(
        split, final_scale, include_mass, device
    )
    output = {key: [] for key in (
        "loss", "target_norm_sq", "prediction_norm_sq", "relative_error", "alignment"
    )}
    transported = []
    for start in range(0, len(x0), batch_size):
        end = min(start + batch_size, len(x0))
        mask = split["mask"][start:end].to(device)
        p0 = project_to_shell(x0[start:end].to(device), regulator_mass)
        p1 = project_to_shell(split["x1"][start:end].to(device), regulator_mass)
        t = times[start:end].to(device)
        state = geodesic_interpolant(p0, p1, t, regulator_mass)
        target = conditional_vector_field(state, p1, t.to(torch.float64), regulator_mass)
        target *= mask.to(torch.float64).unsqueeze(-1)
        dtype = next(model.parameters()).dtype
        prediction = model(
            state, t.to(dtype), conditions[start:end].to(dtype), mask,
            ref_vectors=references[start:end].to(dtype),
        )
        prediction = pushforward_to_tangent(
            state, prediction.to(torch.float64), regulator_mass
        ) * mask.to(torch.float64).unsqueeze(-1)
        metrics = per_jet_field_metrics(prediction, target, mask)
        for key, value in metrics.items():
            output[key].append(value.cpu())
        transported.append(parallel_transport(state, p1, prediction, regulator_mass).cpu())
    return ({key: torch.cat(value).numpy() for key, value in output.items()},
            torch.cat(transported))


def _dispersion(transported_draws, mask):
    values = torch.stack(transported_draws).to(torch.float64)
    mean = values.mean(dim=0, keepdim=True)
    variance = (-normsq4(values - mean)).clamp(min=0.0).mean(dim=0)
    magnitude = (-normsq4(mean[0])).clamp(min=0.0)
    weight = mask.to(torch.float64)
    denom = weight.sum(1).clamp(min=1)
    return (((variance * weight).sum(1) / denom) /
            ((magnitude * weight).sum(1) / denom).clamp(min=1e-12)).numpy()


def _write_records(path, records):
    fields = list(records[0])
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _report_markdown(summary):
    lines = ["# H1 off-path audit", "", f"Verdict: **{summary['verdict']}**", "",
             "| checkpoint | excess ratio | 95% CI | train gap | valid gap |",
             "|---|---:|---:|---:|---:|"]
    for label, result in summary["checkpoints"].items():
        effect = result["h1"]
        lines.append(
            f"| {label} | {effect['ratio']:.4g} | "
            f"[{effect['ci95_ratio_low']:.4g}, {effect['ci95_ratio_high']:.4g}] | "
            f"{result['train_log_gap']:.4g} | {result['valid_log_gap']:.4g} |"
        )
    lines.extend(["", "The primary statistic is the train-minus-validation difference of "
                  "median log fresh/frozen loss ratios. Raw weights are primary; 100k EMA is "
                  "a deployed-model corroboration."])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = Path(args.source_run)
    required = [source / "data/x_train.pkl", source / "data/x_test.pkl",
                source / "train/scale.txt", Path(args.cache_path)]
    required += [source / path for path in sorted(set(CHECKPOINTS.values()))]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required audit artifacts: {missing}")

    checkpoints = {
        label: torch.load(source / path, map_location="cpu", weights_only=False)
        for label, path in CHECKPOINTS.items()
    }
    reference_cfg = checkpoints["step_100000_raw"]["full_config"]
    num_particles = reference_cfg["data"]["num_particles"]
    model_cfg = reference_cfg["model"]
    mass = float(model_cfg["regulator_mass"])
    if model_cfg["backbone"] != "tangent_attention" or mass != 0.1:
        raise ValueError("T0a is frozen to the qualified tangent-attention m=0.1 model")
    with open(source / "train/scale.txt") as handle:
        final_scale = float(handle.read().strip())
    with open(source / "data/x_train.pkl", "rb") as handle:
        train_data = pickle.load(handle)
    with open(source / "data/x_test.pkl", "rb") as handle:
        valid_data = pickle.load(handle)
    with open(args.cache_path, "rb") as handle:
        cache = pickle.load(handle)

    n_train = min(reference_cfg["training"]["n_train_samples"], len(train_data))
    expected = {
        "source_dataset_fingerprint": source_dataset_fingerprint(
            train_data, n_train, num_particles),
        "dataset_indices": list(range(n_train)), "jet_types": ["g"],
        "prior_dist": "axis_aligned_per_jet", "seed": 42,
        "jet_phi_convention": "index_seeded_v1", "jet_phi_seed": 42,
        "num_particles": num_particles, "final_scale": final_scale,
        "geometry": "mass_shell", "regulator_mass": mass,
        "assignment_cost": "squared_geodesic",
    }
    validate_cache_metadata(cache, expected)

    train_indices = deterministic_indices(n_train, args.n_targets, args.audit_seed)
    valid_indices = deterministic_indices(len(valid_data), args.n_targets, args.audit_seed + 1)
    times_np = latin_hypercube_times(args.n_targets, args.audit_seed + 2)
    times = torch.from_numpy(times_np)
    train = _select_split(train_data, train_indices, final_scale, num_particles)
    valid = _select_split(valid_data, valid_indices, final_scale, num_particles)
    train_cached = torch.from_numpy(cache["paired_x0"][train_indices]).float()
    train_cached *= train["mask"].unsqueeze(-1)
    cached_cost = _mass_shell_transport_costs(
        train_cached.numpy(), train["x1"].numpy(),
        train["mask"].sum(1).long().numpy(), mass)[1]

    train_fresh = _fresh_bundles(
        train, args.fresh_draws, args.audit_seed, 10, final_scale, args.workers)
    valid_all = _fresh_bundles(
        valid, args.fresh_draws + 1, args.audit_seed, 20, final_scale, args.workers)
    valid_frozen, valid_fresh = valid_all[0], valid_all[1:]

    bundle = {
        "format_version": AUDIT_FORMAT_VERSION,
        "audit_seed": args.audit_seed,
        "train_indices": torch.from_numpy(train_indices),
        "valid_indices": torch.from_numpy(valid_indices),
        "times": times,
        "train_cached_x0": train_cached,
        "train_fresh_x0": torch.stack([item["x0"] for item in train_fresh]),
        "valid_frozen_x0": valid_frozen["x0"],
        "valid_fresh_x0": torch.stack([item["x0"] for item in valid_fresh]),
        "metadata": {"source_run": str(source), "cache_path": args.cache_path,
                     "n_targets": args.n_targets, "fresh_draws": args.fresh_draws,
                     "coupling": "exact_geodesic_icp", "assignment_cost": "squared_geodesic"},
    }
    torch.save(bundle, out / "audit_bundle.pt")

    records = []
    summary = {"format_version": AUDIT_FORMAT_VERSION, "checkpoints": {},
               "source_run": str(source), "cache_path": args.cache_path,
               "n_targets_per_split": args.n_targets, "fresh_draws": args.fresh_draws}
    aggregates = []
    all_effects = {}
    for label, checkpoint in checkpoints.items():
        use_ema = label.endswith("ema")
        model = _model_from_checkpoint(checkpoint, device, use_ema=use_ema)
        train_base, _ = _evaluate_bundle(
            model, train, train_cached, times, final_scale, mass,
            model_cfg.get("include_mass_condition", False), args.batch_size, device)
        valid_base, _ = _evaluate_bundle(
            model, valid, valid_frozen["x0"], times, final_scale, mass,
            model_cfg.get("include_mass_condition", False), args.batch_size, device)
        train_draw_metrics, train_moved = [], []
        valid_draw_metrics, valid_moved = [], []
        for item in train_fresh:
            metrics, moved = _evaluate_bundle(
                model, train, item["x0"], times, final_scale, mass,
                model_cfg.get("include_mass_condition", False), args.batch_size, device)
            train_draw_metrics.append(metrics); train_moved.append(moved)
        for item in valid_fresh:
            metrics, moved = _evaluate_bundle(
                model, valid, item["x0"], times, final_scale, mass,
                model_cfg.get("include_mass_condition", False), args.batch_size, device)
            valid_draw_metrics.append(metrics); valid_moved.append(moved)
        train_loss = np.stack([item["loss"] for item in train_draw_metrics], axis=1)
        valid_loss = np.stack([item["loss"] for item in valid_draw_metrics], axis=1)
        h1 = cluster_bootstrap_h1(
            train_loss, train_base["loss"], valid_loss, valid_base["loss"],
            seed=args.audit_seed + 100, samples=args.bootstrap_samples,
        )
        target_h1 = cluster_bootstrap_h1(
            np.stack([item["target_norm_sq"] for item in train_draw_metrics], axis=1),
            train_base["target_norm_sq"],
            np.stack([item["target_norm_sq"] for item in valid_draw_metrics], axis=1),
            valid_base["target_norm_sq"],
            seed=args.audit_seed + 101, samples=args.bootstrap_samples,
        )
        train_gap = float(np.median(np.log((train_loss + 1e-12) /
                                           (train_base["loss"][:, None] + 1e-12))))
        valid_gap = float(np.median(np.log((valid_loss + 1e-12) /
                                           (valid_base["loss"][:, None] + 1e-12))))
        result = {
            "h1": h1, "train_log_gap": train_gap, "valid_log_gap": valid_gap,
            "target_field_control": target_h1,
            "train_draw_dispersion_median": float(np.median(_dispersion(train_moved, train["mask"]))),
            "valid_draw_dispersion_median": float(np.median(_dispersion(valid_moved, valid["mask"]))),
        }
        summary["checkpoints"][label] = result
        all_effects[label] = h1
        for split_name, split, base, draws, costs0, cost_draws in (
            ("train", train, train_base, train_draw_metrics, cached_cost,
             [item["transport_cost"] for item in train_fresh]),
            ("valid", valid, valid_base, valid_draw_metrics, valid_frozen["transport_cost"],
             [item["transport_cost"] for item in valid_fresh]),
        ):
            for draw_index in range(-1, args.fresh_draws):
                metric = base if draw_index == -1 else draws[draw_index]
                costs = costs0 if draw_index == -1 else cost_draws[draw_index]
                cell = "frozen" if draw_index == -1 else "fresh"
                for i in range(args.n_targets):
                    info = split["jet_info"][i]
                    records.append({
                        "checkpoint": label, "split": split_name, "cell": cell,
                        "draw": draw_index, "dataset_index": int(
                            train_indices[i] if split_name == "train" else valid_indices[i]),
                        "time": times_np[i], "time_bin": min(7, int(times_np[i] * 8)),
                        "n_particles": int(info[3]), "jet_pt": float(info[1]),
                        "jet_mass": float(info[2]), "transport_cost": float(costs[i]),
                        **{key: float(value[i]) for key, value in metric.items()},
                    })
            aggregates.append({"checkpoint": label, "split": split_name,
                               "stratum": "all", "time_bin": "all",
                               "frozen_loss_median": float(np.median(base["loss"])),
                               "fresh_loss_median": float(np.median(np.stack(
                                   [item["loss"] for item in draws]))),
                               "fresh_frozen_log_gap": train_gap if split_name == "train" else valid_gap})
            fresh_losses = np.stack([item["loss"] for item in draws], axis=1)
            for time_bin in range(8):
                chosen = np.floor(times_np * 8).astype(int) == time_bin
                bin_gap = float(np.median(np.log(
                    (fresh_losses[chosen] + 1e-12) /
                    (base["loss"][chosen, None] + 1e-12)
                )))
                aggregates.append({
                    "checkpoint": label, "split": split_name,
                    "stratum": "time", "time_bin": time_bin,
                    "frozen_loss_median": float(np.median(base["loss"][chosen])),
                    "fresh_loss_median": float(np.median(fresh_losses[chosen])),
                    "fresh_frozen_log_gap": bin_gap,
                })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    step0 = all_effects["step_000000_raw"]
    control_ok = max(step0["ratio"], 1.0 / step0["ratio"]) < 1.1
    target_controls = [result["target_field_control"]
                       for result in summary["checkpoints"].values()]
    target_control_ok = all(
        max(item["ratio"], 1.0 / item["ratio"]) < 1.1 for item in target_controls
    )
    raw100 = all_effects["step_100000_raw"]
    ema100 = all_effects["step_100000_ema"]
    raw2 = all_effects["step_002000_raw"]
    raw80 = all_effects["step_080000_raw"]
    strong = (control_ok and target_control_ok and raw100["ci95_ratio_low"] > 1.5
              and raw100["ratio"] > raw2["ratio"] and raw80["ratio"] > raw2["ratio"]
              and ema100["ratio"] > 1.0)
    weak = (control_ok and target_control_ok and raw80["ci95_ratio_high"] < 1.2
            and raw100["ci95_ratio_high"] < 1.2)
    summary["step0_control_ok"] = control_ok
    summary["target_field_control_ok"] = target_control_ok
    summary["verdict"] = ("strong_h1_support" if strong else
                          "h1_weakened" if weak else "inconclusive")
    _write_records(out / "off_path_records.csv.gz", records)
    with open(out / "off_path_aggregate.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader(); writer.writerows(aggregates)
    write_json(out / "off_path_summary.json", summary)
    (out / "off_path_report.md").write_text(_report_markdown(summary))

    labels = list(summary["checkpoints"])
    ratios = [summary["checkpoints"][label]["h1"]["ratio"] for label in labels]
    lows = [summary["checkpoints"][label]["h1"]["ci95_ratio_low"] for label in labels]
    highs = [summary["checkpoints"][label]["h1"]["ci95_ratio_high"] for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(labels))
    axes[0].errorbar(x, ratios, yerr=[np.array(ratios)-lows, np.array(highs)-ratios], fmt="o-")
    axes[0].axhline(1, color="black", lw=1); axes[0].axhline(1.5, color="tab:red", ls="--")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("excess off-path loss ratio")
    # Time decomposition uses the final raw records.
    final = [row for row in records if row["checkpoint"] == "step_100000_raw"]
    for split_name in ("train", "valid"):
        ys = []
        for time_bin in range(8):
            frozen = [row["loss"] for row in final if row["split"] == split_name
                      and row["cell"] == "frozen" and row["time_bin"] == time_bin]
            fresh = [row["loss"] for row in final if row["split"] == split_name
                     and row["cell"] == "fresh" and row["time_bin"] == time_bin]
            # Records are appended in draw-major order, so tile the target-major
            # frozen values rather than repeating each frozen target K times.
            ys.append(float(np.exp(np.median(np.log(
                (np.asarray(fresh)+1e-12) /
                (np.tile(frozen, args.fresh_draws)+1e-12)
            )))))
        axes[1].plot((np.arange(8)+0.5)/8, ys, marker="o", label=split_name)
    axes[1].axhline(1, color="black", lw=1); axes[1].set_xlabel("t bin center")
    axes[1].set_ylabel("fresh/frozen loss ratio"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "off_path_gap.png", dpi=180); plt.close(fig)

    provenance = {
        "source_run": str(source), "cache_path": args.cache_path,
        "inputs": {str(path): sha256_file(path) for path in required},
        "outputs": {name: sha256_file(out / name) for name in (
            "audit_bundle.pt", "off_path_records.csv.gz", "off_path_aggregate.csv",
            "off_path_summary.json", "off_path_report.md", "off_path_gap.png")},
    }
    write_json(out / "provenance.json", provenance)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
