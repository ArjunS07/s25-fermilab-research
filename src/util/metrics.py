import pickle
import torch
import matplotlib.pyplot as plt

from jetnet.utils import EtaPhiPtE_to_relEtaPhiPt, cartesian_to_EtaPhiPtE
import jetnet.evaluation as jetnet_eval

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
    eval_info["cov_mmd"] = jetnet_eval.cov_mmd(
        real_jets=test_polar_rel[:, :, :3],
        gen_jets=gen_polar_rel[:, :, :3]
    )
    print(f"Cov MMD: {eval_info['cov_mmd']}")

    test_polar_rel = test_polar_rel[:gen_polar_rel.shape[0]]
    assert test_polar_rel.shape == gen_polar_rel.shape

    try:
        eval_info["w1efp"] = jetnet_eval.w1efp(
            jets1=gen_polar_rel,
            jets2=test_polar_rel,
        )
        eval_info["w1m"] = jetnet_eval.w1m(
            jets1=gen_polar_rel,
            jets2=test_polar_rel,
        )
        eval_info["w1p"] = jetnet_eval.w1p(
            jets1=gen_polar_rel,
            jets2=test_polar_rel,
        )
    except Exception as e:
        print(f"Error occurred while computing W1 metrics: {e}")

    try:
        eval_info["fpd"] = jetnet_eval.fpd(
            real_features=test_polar_rel.reshape((-1, 4)),
            gen_features=gen_polar_rel.reshape((-1, 4)),
            seed=42
        )
    except Exception as e:
        print(f"Error occurred while computing fpd: {e}")

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