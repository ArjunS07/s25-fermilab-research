import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from jetnet.utils import EtaPhiPtE_to_relEtaPhiPt, cartesian_to_EtaPhiPtE
import jetnet.evaluation as jetnet_eval

from util.minkowski_utils import normsq4
from util.ks import ks_statistic_vs_uniform, ks_two_sample

# Fixed seed so the random jet-phi assignment (and hence the isotropy reference) is
# reproducible across runs — the harness is a fixed yardstick (experiment plan 0.4).
EVAL_SEED = 42


def _total_momentum_direction(p3):
    """cos(theta) and phi of a batch of 3-momenta (N, 3)."""
    norm = p3.norm(dim=-1).clamp(min=1e-12)
    cos_theta = p3[:, 2] / norm
    phi = torch.atan2(p3[:, 1], p3[:, 0])
    return cos_theta, phi


def _plot_physicality_and_isotropy(gen_cartesian, test_polar_abs, output_path):
    """Physicality diagnostics + isotropy figure (experiment plan 0.1, 0.3).

    gen_cartesian : (N, P, 4) generated 4-vectors (E, px, py, pz), padding zeroed.
    test_polar_abs: (N_test, P, 4) absolute (eta, phi, pt, E) for the test set.
    Returns a dict of scalar diagnostics to fold into eval_info / metrics.csv.
    """
    gen = gen_cartesian.cpu()
    real = gen.abs().sum(dim=-1) > 1e-12  # per-particle real mask

    # --- Physicality: E<0 and spacelike (m^2 = <x,x> < 0) fractions over real particles ---
    E = gen[..., 0]
    msq = normsq4(gen)  # (N, P)
    n_real = real.sum().clamp(min=1).item()
    frac_neg_E = (E < 0)[real].float().mean().item() if real.any() else float("nan")
    frac_spacelike = (msq < 0)[real].float().mean().item() if real.any() else float("nan")
    msq_real = msq[real]

    diag = {
        "frac_negative_energy": frac_neg_E,
        "frac_spacelike": frac_spacelike,
        "msq_mean": msq_real.mean().item() if msq_real.numel() else float("nan"),
        "msq_median": msq_real.median().item() if msq_real.numel() else float("nan"),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(msq_real.numpy(), bins=100, density=True, color="tab:purple", alpha=0.8)
    ax.axvline(0.0, color="k", ls="--", lw=1, label="m² = 0 (lightlike)")
    ax.set_xlabel(r"$m^2 = \langle x, x\rangle$ (generated particles)")
    ax.set_ylabel("Density")
    ax.set_title(f"Physicality: E<0 {frac_neg_E:.2%}, spacelike {frac_spacelike:.2%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_path}/physicality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Isotropy: cos(theta), phi of the per-jet total 3-momentum, generated vs test ---
    gen_total = gen[..., 1:4].sum(dim=1)  # (N, 3), padding already zero
    gen_cos, gen_phi = _total_momentum_direction(gen_total)

    # KS of the generated jet-axis direction vs the isotropic (uniform) reference: cos(theta)
    # ~ U[-1,1], phi ~ U[-pi, pi]. Small p => departs from isotropy (symmetry successfully
    # broken). This is the plan's per-run abort signal — recorded in metrics.csv / summary.json.
    ks_cos_d, ks_cos_p = ks_statistic_vs_uniform(gen_cos.numpy(), -1.0, 1.0)
    ks_phi_d, ks_phi_p = ks_statistic_vs_uniform(gen_phi.numpy(), -np.pi, np.pi)
    diag.update({
        "isotropy_ks_costheta": ks_cos_d,
        "isotropy_ks_costheta_p": ks_cos_p,
        "isotropy_ks_phi": ks_phi_d,
        "isotropy_ks_phi_p": ks_phi_p,
    })

    eta, phi_t, pt = test_polar_abs[..., 0], test_polar_abs[..., 1], test_polar_abs[..., 2]
    test_real = pt > 0
    px = (pt * torch.cos(phi_t)) * test_real
    py = (pt * torch.sin(phi_t)) * test_real
    pz = (pt * torch.sinh(eta)) * test_real
    test_total = torch.stack([px.sum(1), py.sum(1), pz.sum(1)], dim=-1)
    test_cos, test_phi = _total_momentum_direction(test_total)

    # Gen-vs-data two-sample KS (more discriminating than vs-uniform).
    gvd_cos_d, gvd_cos_p = ks_two_sample(gen_cos.numpy(), test_cos.numpy())
    gvd_phi_d, gvd_phi_p = ks_two_sample(gen_phi.numpy(), test_phi.numpy())
    diag.update({
        "isotropy_gen_vs_data_cos_ks": gvd_cos_d,
        "isotropy_gen_vs_data_cos_p": gvd_cos_p,
        "isotropy_gen_vs_data_phi_ks": gvd_phi_d,
        "isotropy_gen_vs_data_phi_p": gvd_phi_p,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Isotropy of generated jet axis vs. data", fontsize=14)
    axes[0].hist(test_cos.numpy(), bins=60, density=True, alpha=0.5, label="Real")
    axes[0].hist(gen_cos.numpy(), bins=60, density=True, alpha=0.5, label="Generated")
    axes[0].axhline(0.5, color="gray", ls=":", lw=1, label="Isotropic (uniform)")
    axes[0].set_xlabel(r"$\cos\theta$ of total 3-momentum")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[1].hist(test_phi.numpy(), bins=60, density=True, alpha=0.5, label="Real")
    axes[1].hist(gen_phi.numpy(), bins=60, density=True, alpha=0.5, label="Generated")
    axes[1].set_xlabel(r"$\phi$ of total 3-momentum")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{output_path}/isotropy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return diag

def __x_test_to_abs(X_test, device='cpu'):
    jet_eta = (X_test[:][1][:, 0]).unsqueeze(1)
    jet_phi_vals = (2 * torch.pi) * torch.rand(len(X_test)).unsqueeze(1)
    jet_pt_ec = X_test[:][1][:, 1:3]
    jet_features = torch.concat([jet_eta, jet_phi_vals, jet_pt_ec], dim=-1)
    eta_rel, phi_rel, pt_rel = torch.unbind(X_test[:][0][:, :, :3], axis=-1)
    Eta, Phi, Pt, _ = torch.unbind(jet_features, axis=-1)

    pt = pt_rel * Pt.unsqueeze(1)
    eta = eta_rel + Eta.unsqueeze(1)
    phi = phi_rel + Phi.unsqueeze(1)
    p0 = pt * torch.cosh(eta)

    return torch.stack([eta, phi, pt, p0], dim=-1).to(device)

def run_save_metrics(X_test, gen_samples, jet_types, output_path, device='cpu'):
    torch.manual_seed(EVAL_SEED)  # fixed yardstick across runs
    gen_polar_abs = cartesian_to_EtaPhiPtE(gen_samples.to(device))
    gen_polar_abs[:, :, 1] += torch.pi

    test_polar_abs = __x_test_to_abs(X_test=X_test, device=device)
    test_polar_rel = EtaPhiPtE_to_relEtaPhiPt(test_polar_abs)
    gen_polar_rel = EtaPhiPtE_to_relEtaPhiPt(gen_polar_abs)

    # Move to CPU once for all subsequent numpy / jetnet calls
    test_polar_abs = test_polar_abs.cpu()
    test_polar_rel = test_polar_rel.cpu()
    gen_polar_abs  = gen_polar_abs.cpu()
    gen_polar_rel  = gen_polar_rel.cpu()
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Real vs Generated Jet Distributions', fontsize=16)

    # Relative coordinates (what metrics use)
    features = ['η_rel', 'φ_rel', 'p_T_rel']
    for idx, (ax, feature) in enumerate(zip(axes[0], features)):
        ax.hist(test_polar_rel[:, :, idx].flatten().numpy(), bins=100, 
                alpha=0.5, label='Real', density=True)
        ax.hist(gen_polar_rel[:, :, idx].flatten().numpy(), bins=100, 
                alpha=0.5, label='Generated', density=True)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'{feature} (Relative)')

    # Absolute coordinates
    abs_features = ['η', 'φ', 'p_T']
    for idx, (ax, feature) in enumerate(zip(axes[1], abs_features)):
        ax.hist(test_polar_abs[:, :, idx].flatten().numpy(), bins=100, 
                alpha=0.5, label='Real', density=True)
        ax.hist(gen_polar_abs[:, :, idx].flatten().numpy(), bins=100, 
                alpha=0.5, label='Generated', density=True)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'{feature} (Absolute)')

    plt.tight_layout()
    plt.savefig(f"{output_path}/distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Jet-level features comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Jet-Level Features', fontsize=16)

    # Number of particles per jet
    real_mask = (test_polar_rel[:, :, 2] > 0).sum(dim=1)
    gen_mask = (gen_polar_rel[:, :, 2] > 0).sum(dim=1)

    axes[0].hist(real_mask.numpy(), bins=30, alpha=0.5, label='Real', density=True)
    axes[0].hist(gen_mask.numpy(), bins=30, alpha=0.5, label='Generated', density=True)
    axes[0].set_xlabel('Number of Particles')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].set_title('Particle Multiplicity')

    # Total jet pT
    real_jet_pt = test_polar_abs[:, :, 2].sum(dim=1)
    gen_jet_pt = gen_polar_abs[:, :, 2].sum(dim=1)

    axes[1].hist(real_jet_pt.numpy(), bins=50, alpha=0.5, label='Real', density=True)
    axes[1].hist(gen_jet_pt.numpy(), bins=50, alpha=0.5, label='Generated', density=True)
    axes[1].set_xlabel('Total Jet p_T')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].set_title('Total Jet Transverse Momentum')

    plt.tight_layout()
    plt.savefig(f"{output_path}/jet_features.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_path}/")

    eval_info = {}

    # Physicality diagnostics + isotropy figure (the before/after yardstick).
    try:
        eval_info.update(_plot_physicality_and_isotropy(gen_samples, test_polar_abs, output_path))
        print(f"Physicality: E<0 {eval_info['frac_negative_energy']:.2%}, "
              f"spacelike {eval_info['frac_spacelike']:.2%}")
    except Exception as e:
        print(f"Error computing physicality/isotropy diagnostics: {e}")

    eval_info["cov_mmd"] = jetnet_eval.cov_mmd(
        real_jets=test_polar_rel[:, :, :3],
        gen_jets=gen_polar_rel[:, :, :3]
    )
    print(f"Cov MMD: {eval_info['cov_mmd']}")

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
                jets=gen_polar_rel,
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