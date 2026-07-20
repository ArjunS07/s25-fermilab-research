import json
import os
import pickle
import torch

from jetnet.utils import EtaPhiPtE_to_relEtaPhiPt, cartesian_to_EtaPhiPtE
import jetnet.evaluation as jetnet_eval

from util.eval_report import eval_report, physicality_isotropy_scalars
from util.tail_diagnostics import endpoint_tail_diagnostics

# Fixed seed so the random jet-phi assignment (and hence the isotropy reference) is
# reproducible across runs — the harness is a fixed yardstick (experiment plan 0.4).
EVAL_SEED = 42


def __x_test_to_abs(X_test, device='cpu'):
    """Assemble absolute (eta, phi, pt, E) per particle from the JetNet test set.

    JetNet's jet_features layout is [eta, pt, mass, num_particles, type]. Phi is
    not tracked per-jet in JetNet, so we draw a uniform reference phi per jet —
    that random assignment is what the fixed EVAL_SEED reproduces.
    """
    jet_eta = (X_test[:][1][:, 0]).unsqueeze(1)
    jet_phi_vals = (2 * torch.pi) * torch.rand(len(X_test)).unsqueeze(1)
    jet_pt_ec = X_test[:][1][:, 1:3]
    jet_features = torch.concat([jet_eta, jet_phi_vals, jet_pt_ec], dim=-1)
    eta_rel, phi_rel, pt_rel = torch.unbind(X_test[:][0][:, :, :3], axis=-1)
    Eta, Phi, Pt, _ = torch.unbind(jet_features, axis=-1)

    pt = pt_rel * Pt.unsqueeze(1)
    eta = eta_rel + Eta.unsqueeze(1)
    phi = phi_rel + Phi.unsqueeze(1)
    # Wrap to (-pi, pi] to match the atan2 convention used for generated phi
    # (cartesian_to_EtaPhiPtE). The synthetic jet phi is drawn on [0, 2pi), so
    # without this the absolute-phi figure shows test and gen on different
    # branches of the circle — a cosmetic ~pi offset. Relative coords and the
    # atan2-based isotropy stats are unaffected either way.
    phi = torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi
    p0 = pt * torch.cosh(eta)

    return torch.stack([eta, phi, pt, p0], dim=-1).to(device)


def run_save_metrics(
    X_test,
    gen_samples,
    jet_types,
    output_path,
    device='cpu',
    gen_jet_types=None,
    gen_pt_cond=None,
    prior_samples=None,
):
    """Compute scalar metrics + call eval_report to write the 14 eval figures.

    Args:
        jet_types: list of class-name strings (e.g. ["g", "q", "t"]). Used both
            for FPND per class and passed to eval_report for panel labels.
        gen_jet_types: (N_gen,) long class-index tensor; if None, defaults to
            zeros (per-class figures degenerate to a single-class overlay).
        gen_pt_cond: (N_gen,) normalized conditioning pT; if None the slope
            annotation on jet_pt_total is skipped.
        prior_samples: exact pre-transport Cartesian state paired with gen_samples.
    """
    torch.manual_seed(EVAL_SEED)  # fixed yardstick across runs
    if prior_samples is None:
        print("Prior provenance unavailable: exact paired prior tensor was not supplied; "
              "prior overlays will be omitted (not reconstructed).")

    # A handful of non-finite jets must not erase every metric. Drop whole invalid jets,
    # report the exact count, and compare the remaining generated sample to an equally-sized
    # prefix of the frozen test set.
    tail_report = endpoint_tail_diagnostics(gen_samples)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "endpoint_tail_diagnostics.json"), "w") as handle:
        json.dump(tail_report, handle, indent=2)

    finite_jet = torch.isfinite(gen_samples).all(dim=-1).all(dim=-1)
    n_invalid = int((~finite_jet).sum().item())
    n_total = int(gen_samples.shape[0])
    if not finite_jet.any():
        raise ValueError("all generated jets contain non-finite values")
    if n_invalid:
        print(f"WARNING: dropping {n_invalid}/{n_total} non-finite generated jets before metrics")
        gen_samples = gen_samples[finite_jet]
        if gen_jet_types is not None:
            gen_jet_types = gen_jet_types[finite_jet.to(gen_jet_types.device)]
        if gen_pt_cond is not None:
            gen_pt_cond = gen_pt_cond[finite_jet.to(gen_pt_cond.device)]
        if prior_samples is not None:
            prior_samples = prior_samples[finite_jet.to(prior_samples.device)]

    gen_polar_abs = cartesian_to_EtaPhiPtE(gen_samples.to(device))
    prior_polar_abs = (cartesian_to_EtaPhiPtE(prior_samples.to(device))
                       if prior_samples is not None else None)

    test_polar_abs = __x_test_to_abs(X_test=X_test, device=device)
    test_polar_rel = EtaPhiPtE_to_relEtaPhiPt(test_polar_abs)
    gen_polar_rel  = EtaPhiPtE_to_relEtaPhiPt(gen_polar_abs)
    prior_polar_rel = (EtaPhiPtE_to_relEtaPhiPt(prior_polar_abs)
                       if prior_polar_abs is not None else None)

    # Padded rows must land exactly at (0, 0, 0) in rel coords on both sides so
    # jetnet's marginal-based W1s aren't polluted by phantom values. Gen padding is
    # all-zero Cartesian (masked in generate_samples); test padding comes from the
    # JetNet mask channel at column 3.
    gen_mask  = (gen_samples.abs().sum(dim=-1) > 0).to(gen_polar_rel.dtype).to(gen_polar_rel.device)
    test_particles = X_test[:][0]
    if test_particles.shape[-1] >= 4:
        test_mask = test_particles[:, :, 3].to(test_polar_rel.dtype).to(test_polar_rel.device)
    else:
        # Fallback if data.py's MASK flag was flipped off: recover from pt_rel > 0.
        test_mask = (test_particles[:, :, 2] > 0).to(test_polar_rel.dtype).to(test_polar_rel.device)
    gen_polar_rel  = gen_polar_rel  * gen_mask.unsqueeze(-1)
    if prior_polar_rel is not None:
        prior_mask = (prior_samples.abs().sum(dim=-1) > 0).to(
            prior_polar_rel.dtype).to(prior_polar_rel.device)
        prior_polar_rel = prior_polar_rel * prior_mask.unsqueeze(-1)
    test_polar_rel = test_polar_rel * test_mask.unsqueeze(-1)

    # Move to CPU once for all subsequent numpy / jetnet calls
    test_polar_abs = test_polar_abs.cpu()
    test_polar_rel = test_polar_rel.cpu()
    gen_polar_abs  = gen_polar_abs.cpu()
    gen_polar_rel  = gen_polar_rel.cpu()
    if prior_polar_abs is not None:
        prior_polar_abs = prior_polar_abs.cpu()
        prior_polar_rel = prior_polar_rel.cpu()

    # Test per-jet class index lives at X_test[:][1][:, 4] (JetNet layout).
    test_jet_types = X_test[:][1][:, 4].long().cpu()
    if gen_jet_types is None:
        gen_jet_types = torch.zeros(gen_samples.shape[0], dtype=torch.long)
    else:
        gen_jet_types = gen_jet_types.long().cpu()

    eval_info = {
        "n_generated_total": n_total,
        "n_generated_invalid": n_invalid,
        "frac_generated_invalid": n_invalid / n_total,
        "n_generated_finite_max_abs_gt_1e3": tail_report.get("n_finite_max_abs_gt_1e3", 0),
        "n_generated_finite_max_abs_gt_1e6": tail_report.get("n_finite_max_abs_gt_1e6", 0),
        "frac_generated_finite_max_abs_gt_1e6": tail_report.get(
            "fraction_finite_max_abs_gt_1e6", 0.0),
    }
    try:
        eval_info.update(physicality_isotropy_scalars(gen_samples.cpu(), test_polar_abs))
        print(f"Physicality: E<0 {eval_info['frac_negative_energy']:.2%}, "
              f"spacelike {eval_info['frac_spacelike']:.2%}")
    except Exception as e:
        print(f"Error computing physicality/isotropy scalars: {e}")

    try:
        eval_report(
            gen_cartesian=gen_samples.cpu(),
            test_polar_abs=test_polar_abs,
            gen_polar_abs=gen_polar_abs,
            test_polar_rel=test_polar_rel,
            gen_polar_rel=gen_polar_rel,
            gen_jet_types=gen_jet_types,
            test_jet_types=test_jet_types,
            jet_type_names=jet_types,
            gen_pt_cond=gen_pt_cond.cpu() if gen_pt_cond is not None else None,
            prior_cartesian=prior_samples.cpu() if prior_samples is not None else None,
            prior_polar_abs=prior_polar_abs,
            prior_polar_rel=prior_polar_rel,
            output_path=output_path,
        )
    except Exception as e:
        print(f"Error occurred while writing eval_report: {e}")

    try:
        eval_info["cov_mmd"] = jetnet_eval.cov_mmd(
            real_jets=test_polar_rel[:gen_polar_rel.shape[0], :, :3],
            gen_jets=gen_polar_rel[:, :, :3]
        )
        print(f"Cov MMD: {eval_info['cov_mmd']}")
    except Exception as e:
        print(f"Error computing cov_mmd: {e}")

    test_polar_rel = test_polar_rel[:gen_polar_rel.shape[0]]
    assert test_polar_rel.shape == gen_polar_rel.shape

    # Jetnet W1 metrics expect 3-channel (eta_rel, phi_rel, pt_rel) jets
    gen_rel3  = gen_polar_rel[:, :, :3]
    test_rel3 = test_polar_rel[:, :, :3]

    try:
        eval_info["w1efp"] = jetnet_eval.w1efp(jets1=gen_rel3, jets2=test_rel3)
        print(f"W1EFP: {eval_info['w1efp']}")
    except Exception as e:
        print(f"Error computing w1efp: {e}")

    try:
        eval_info["w1m"] = jetnet_eval.w1m(jets1=gen_rel3, jets2=test_rel3)
        print(f"W1M: {eval_info['w1m']}")
    except Exception as e:
        print(f"Error computing w1m: {e}")

    try:
        eval_info["w1p"] = jetnet_eval.w1p(jets1=gen_rel3, jets2=test_rel3)
        print(f"W1P: {eval_info['w1p']}")
    except Exception as e:
        print(f"Error computing w1p: {e}")

    try:
        # fpd expects one flat feature vector per jet: (N_jets, max_particles * 3)
        N_test = test_rel3.shape[0]
        N_gen  = gen_rel3.shape[0]
        eval_info["fpd"] = jetnet_eval.fpd(
            real_features=test_rel3.reshape(N_test, -1),
            gen_features=gen_rel3.reshape(N_gen, -1),
            seed=42,
        )
        print(f"FPD: {eval_info['fpd']}")
    except Exception as e:
        print(f"Error computing fpd: {e}")

    for jet_type in jet_types:
        try:
            eval_info[f"fpnd_{jet_type}"] = jetnet_eval.fpnd(
                # JetNet's pretrained ParticleNet weights are float32.  The
                # mass-shell pipeline deliberately retains float64 through
                # generation, so cast only at this external model boundary.
                jets=gen_polar_rel.float(),
                jet_type=jet_type,
                use_tqdm=False
            )
        except Exception as e:
            print(f"Error occurred while computing fpnd for {jet_type}: {e}")

    with open(f"{output_path}/metrics.csv", "w") as f:
        f.write("Metric,Value\n")
        for key, value in eval_info.items():
            f.write(f"{key},{value}\n")

    with open(f"{output_path}/eval_info.pkl", "wb") as f:
        pickle.dump(eval_info, f)
    print(f"Metrics saved to {output_path}/metrics.csv and {output_path}/eval_info.pkl")
    return eval_info
