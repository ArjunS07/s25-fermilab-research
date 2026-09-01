import seaborn as sns
import torch
import matplotlib.pyplot as plt
import os
import time

from jetnet.utils import EtaPhiPtE_to_cartesian

from util.data import jet_attributes
from util.infra.file_management import make_clear_folder
from util.data.distributions import gen_initial_distribution
from util.geometry.coordinates import build_reference_vectors
from util.geometry.conditioning import scale_condition_pt

plt.rc("mathtext", fontset="cm")
sns.set_style("whitegrid")

features = [r"e_c", r"$p_x$", r"$p_y$", r"$p_z$"]


def generate_samples(
        model,
        jet_attr_model,
        device,
        root_output_path,
        max_particles_per_jet,
        final_scale,
        integration_steps,
        integration_end_time,
        n_samples,
        batch_size,
        jet_types=("g", "q", "t"),
        samples_per_jet_type=None,
        use_cfg=False,
        cfg_guidance_weight=2.0,
        regulator_mass=0.5,
        prior_dist='axis_aligned_per_jet',
        replay_bundle_path=None,
):
    sampling_start = time.perf_counter()

    if samples_per_jet_type is not None:
        expected_samples = len(jet_types) * samples_per_jet_type
        if n_samples != expected_samples:
            raise ValueError(
                "balanced generation requires n_samples == "
                f"len(jet_types) * samples_per_jet_type; got {n_samples} != "
                f"{len(jet_types)} * {samples_per_jet_type}"
            )
        balanced_global_types = torch.tensor(
            jet_attributes.global_jet_type_indices(jet_types), dtype=torch.long
        ).repeat_interleave(samples_per_jet_type)
    else:
        balanced_global_types = None

    # make folder
    make_clear_folder(f"{root_output_path}/samples")

    all_samples = []
    all_prior_samples = []
    all_pt_cond = []      # conditioning pT for each jet (physical GeV)
    all_gen_eta = []      # conditioning jet eta for each jet (for the canonical FPND axis)
    all_jet_types = []    # per-jet class index in configured local class order
    all_failure_steps = []
    all_explosion_steps = []
    all_generated_jet_attrs = []
    all_jet_phi = []
    all_masks = []
    all_internal_prior = []
    all_failure_records = []

    replay_bundle = None
    if replay_bundle_path:
        # Replay bundles are deliberately restricted to tensors and primitive metadata.
        replay_bundle = torch.load(replay_bundle_path, map_location="cpu", weights_only=False)
        if replay_bundle.get("format_version") != 1:
            raise ValueError("unsupported replay bundle format")
        metadata = replay_bundle.get("metadata", {})
        expected = {
            "n_samples": int(n_samples),
            "num_particles": int(max_particles_per_jet),
            "prior_dist": prior_dist,
            "final_scale": float(final_scale),
            "regulator_mass": float(regulator_mass),
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"incompatible replay bundle metadata: {mismatches}")

    with torch.inference_mode():
        model.eval()
        jet_attr_model.eval()

        times = torch.linspace(0, integration_end_time, integration_steps + 1).to(device)

        for start_idx in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - start_idx)

            if replay_bundle is None:
                one_hot_types = None
                if balanced_global_types is not None:
                    one_hot_types = jet_attributes.one_hot_enc_jet_type(
                        balanced_global_types[start_idx:start_idx + current_batch_size].to(device)
                    )
                generated_jet_attrs, _ = jet_attributes.generate_jets(
                    jet_attr_model, device, jet_types=jet_types,
                    num_jets=current_batch_size, one_hot_types=one_hot_types,
                )
            else:
                generated_jet_attrs = replay_bundle["generated_jet_attrs"][
                    start_idx:start_idx + current_batch_size
                ].to(device)
            # generated_jet_attrs layout: [one_hot(5), eta, pt, mass, n_particles]
            jet_one_hot_enc = generated_jet_attrs[:, :5].to(device)
            gen_pt = generated_jet_attrs[:, 6].to(device)   # normalized pT from NF
            gen_mass = generated_jet_attrs[:, 7].to(device)
            gen_n_particles = generated_jet_attrs[:, -1].long().to(device)
            gen_n_particles = gen_n_particles.clamp(max=max_particles_per_jet)
            masks = jet_attributes.generate_masks(
                gen_n_particles,
                max_particles_per_jet=max_particles_per_jet,
                device=device,
            )

            # All supported priors are conditioned on the sampled jet axis.
            jet_phi = (
                (2 * torch.pi) * torch.rand(current_batch_size, device=device)
                if replay_bundle is None else
                replay_bundle["jet_phi"][start_idx:start_idx + current_batch_size].to(device)
            )
            gen_eta = generated_jet_attrs[:, 5].to(device)
            jet_features = torch.stack([gen_eta, gen_pt], dim=-1)

            if replay_bundle is None:
                x = gen_initial_distribution(
                    batch_size=current_batch_size,
                    num_particles=max_particles_per_jet,
                    prior_dist=prior_dist,
                    jet_features=jet_features,
                    jet_phi=jet_phi,
                    device=device,
                    model_scale=final_scale,
                    particle_mask=masks,
                ).to(device)
            if replay_bundle is not None:
                saved_masks = replay_bundle["masks"][
                    start_idx:start_idx + current_batch_size
                ].to(device)
                if not torch.equal(masks.cpu(), saved_masks.cpu()):
                    raise ValueError("replay bundle mask disagrees with Stage-1 multiplicity")
                x = replay_bundle["internal_prior"][
                    start_idx:start_idx + current_batch_size
                ].to(device)
            # Preserve the exact integration start, including sampled attributes,
            # multiplicity mask, and orientation.
            prior_x = x * masks.unsqueeze(-1)
            from util.geometry.mass_shell import project_to_shell
            if replay_bundle is None:
                prior_x = project_to_shell(prior_x, regulator_mass) * masks.unsqueeze(-1)
            else:
                prior_x = x * masks.unsqueeze(-1)
            cond_pt = scale_condition_pt(gen_pt, final_scale)
            condition_parts = [
                jet_one_hot_enc,
                gen_n_particles.unsqueeze(-1).float(),
                cond_pt.unsqueeze(-1),
            ]
            condition_parts.append((gen_mass / final_scale).unsqueeze(-1))
            cond = torch.cat(condition_parts, dim=-1).to(device)

            # H uses typed virtual references reconstructed from sampled jet attributes.
            ref_vectors = build_reference_vectors(
                gen_eta, gen_pt, final_scale, device, jet_phi=jet_phi, jet_mass=gen_mass,
            )

            # Integrate geodesically on the mass shell; the transport state remains on H_m.
            y = project_to_shell(x * masks.unsqueeze(-1), regulator_mass)
            failure_step = torch.full(
                (current_batch_size,), -1, dtype=torch.int64, device=device)
            explosion_step = torch.full_like(failure_step, -1)
            failure_records = [None] * current_batch_size
            for i in range(integration_steps):
                active = failure_step < 0
                if not active.any():
                    break
                stepped = model.step_hyperbolic(
                    y_t=y[active], jet_conditions=cond[active], mask=masks[active],
                    t_start=times[i], t_end=times[i + 1], use_cfg=use_cfg,
                    guidance_weight=cfg_guidance_weight, ref_vectors=ref_vectors[active],
                )
                active_indices = active.nonzero(as_tuple=False).flatten()
                finite = torch.isfinite(stepped).all(dim=(1, 2))
                y = y.clone()
                y[active_indices[finite]] = stepped[finite]
                if (~finite).any():
                    failed_indices = active_indices[~finite]
                    y[failed_indices] = float("nan")
                    failure_step[failed_indices] = i
                    for global_idx in failed_indices.tolist():
                        failure_records[global_idx] = {
                            "integration_step": i,
                            "reason": "nonfinite_state",
                            "message": "mass-shell Euler step produced a non-finite state",
                        }
                max_abs = stepped.abs().amax(dim=(1, 2))
                new_explosive = finite & (max_abs > 1e6)
                unset = explosion_step[active_indices] < 0
                if (new_explosive & unset).any():
                    explosion_step[active_indices[new_explosive & unset]] = i
            from util.geometry.mass_shell import massless_energy_view
            shell_x = project_to_shell(y, regulator_mass)
            x = massless_energy_view(shell_x, masks)
            prior_x = massless_energy_view(prior_x, masks)
            scaled_x = final_scale * x * masks.unsqueeze(-1)
            # torch.save(scaled_x, f"{root_output_path}/samples/batch_{start_idx//batch_size:04d}.pt")

            all_samples.append(scaled_x.cpu())
            all_prior_samples.append((final_scale * prior_x).cpu())
            all_pt_cond.append(gen_pt.cpu())
            all_gen_eta.append(generated_jet_attrs[:, 5].cpu())
            global_types = jet_one_hot_enc.argmax(dim=-1)
            all_jet_types.append(
                jet_attributes.local_jet_type_indices(global_types, jet_types).cpu()
            )
            all_generated_jet_attrs.append(generated_jet_attrs.cpu())
            all_jet_phi.append(jet_phi.cpu())
            all_masks.append(masks.cpu())
            all_internal_prior.append(prior_x.cpu())
            all_failure_steps.append(failure_step.cpu())
            all_explosion_steps.append(explosion_step.cpu())
            all_failure_records.extend(failure_records)

    all_samples_cat   = torch.cat(all_samples, dim=0)
    all_prior_samples_cat = torch.cat(all_prior_samples, dim=0)
    all_pt_cond_cat   = torch.cat(all_pt_cond, dim=0)
    all_gen_eta_cat   = torch.cat(all_gen_eta, dim=0)
    all_jet_types_cat = torch.cat(all_jet_types, dim=0)

    if replay_bundle is None:
        bundle = {
            "format_version": 1,
            "metadata": {
                "n_samples": int(n_samples),
                "num_particles": int(max_particles_per_jet),
                "prior_dist": prior_dist,
                "final_scale": float(final_scale),
                "regulator_mass": float(regulator_mass),
                "samples_per_jet_type": samples_per_jet_type,
            },
            "generated_jet_attrs": torch.cat(all_generated_jet_attrs),
            "jet_phi": torch.cat(all_jet_phi),
            "masks": torch.cat(all_masks),
            "internal_prior": torch.cat(all_internal_prior),
        }
        torch.save(bundle, os.path.join(root_output_path, "replay_bundle.pt"))

    if all_failure_steps:
        import json
        failures = torch.cat(all_failure_steps)
        explosions = torch.cat(all_explosion_steps)
        generated_attrs = torch.cat(all_generated_jet_attrs)
        prior_max_abs = all_prior_samples_cat.abs().amax(dim=(1, 2))
        endpoint_max_abs = all_samples_cat.abs().amax(dim=(1, 2))
        noteworthy = (failures >= 0) | (explosions >= 0)
        def quantiles(values):
            q = torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0], dtype=torch.float64)
            return {name: float(value) for name, value in zip(
                ("p50", "p90", "p99", "p999", "max"),
                torch.quantile(values.to(torch.float64), q))}
        trajectory_records = []
        for idx in noteworthy.nonzero(as_tuple=False).flatten().tolist():
            attrs = generated_attrs[idx]
            trajectory_records.append({
                "index": idx,
                "failure_step": int(failures[idx]),
                "explosion_step": int(explosions[idx]),
                "jet_type": int(attrs[:5].argmax()),
                "eta": float(attrs[5]),
                "pt_condition": float(attrs[6]),
                "mass_condition": float(attrs[7]),
                "n_particles": int(attrs[-1]),
                "prior_max_abs": float(prior_max_abs[idx]),
                "endpoint_max_abs": float(endpoint_max_abs[idx]),
                "failure": all_failure_records[idx],
            })
        failure_reason_counts = {}
        for record in all_failure_records:
            if record is not None:
                reason = record.get("reason", "unknown")
                failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
        generation_report = {
            "n_total": int(failures.numel()),
            "n_failed": int((failures >= 0).sum()),
            "n_crossed_max_abs_1e6": int((explosions >= 0).sum()),
            "n_unstable": int(((failures >= 0) | (explosions >= 0)).sum()),
            "first_failure_step_counts": {
                str(int(step)): int((failures == step).sum())
                for step in torch.unique(failures[failures >= 0]).tolist()
            },
            "first_explosion_step_counts": {
                str(int(step)): int((explosions == step).sum())
                for step in torch.unique(explosions[explosions >= 0]).tolist()
            },
            "failure_reason_counts": failure_reason_counts,
            "integration_steps": int(integration_steps),
            "integration_end_time": float(integration_end_time),
            "flow_geometry": "mass_shell",
            "sampling_seconds": time.perf_counter() - sampling_start,
            "all_trajectory_quantiles": {
                "abs_eta": quantiles(generated_attrs[:, 5].abs()),
                "pt_condition": quantiles(generated_attrs[:, 6]),
                "mass_condition": quantiles(generated_attrs[:, 7]),
                "n_particles": quantiles(generated_attrs[:, -1]),
                "prior_max_abs": quantiles(prior_max_abs),
            },
            "noteworthy_trajectories": trajectory_records,
        }
        with open(f"{root_output_path}/generation_diagnostics.json", "w") as handle:
            json.dump(generation_report, handle, indent=2)

    return (all_samples_cat.to(device), all_jet_types_cat, all_pt_cond_cat,
            all_prior_samples_cat, all_gen_eta_cat)
