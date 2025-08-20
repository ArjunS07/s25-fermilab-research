import torch

from jetnet.utils import EtaPhiPtE_to_relEtaPhiPt, cartesian_to_EtaPhiPtE
import jetnet.evaluation as jetnet_eval

def __x_test_to_abs(X_test):
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

    return torch.stack([eta, phi, pt, p0], dim=-1)

def run_save_metrics(X_test, gen_samples, scale, jet_types, output_path):
    gen_polar_abs = cartesian_to_EtaPhiPtE(scale * gen_samples)
    gen_polar_abs[:, :, 1] += torch.pi

    test_polar_abs = __x_test_to_abs(X_test=X_test)
    test_polar_rel = EtaPhiPtE_to_relEtaPhiPt(test_polar_abs)
    gen_polar_rel = EtaPhiPtE_to_relEtaPhiPt(gen_polar_abs)
    
    eval_info = {}
    eval_info["cov_mmd"] = jetnet_eval.cov_mmd(
        real_jets=test_polar_rel[:, :, :3],
        gen_jets=gen_polar_rel[:, :, :3]
    )
    eval_info["fpd"] = jetnet_eval.fpd(
        real_features=test_polar_rel.reshape((-1, 4)),
        gen_features=gen_polar_rel.reshape((-1, 4)),
        seed=42
    )
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

    for jet_type in jet_types:
        try:
            eval_info[f"fpnd_{jet_type}"] = jetnet_eval.fpnd(
                jets=gen_polar_rel[:, :, :3],
                jet_type=jet_type,
                use_tqdm=False
            )
        except Exception as e:
            print(f"Error occurred while computing fpnd for {jet_type}: {e}")

    with open(f"{output_path}/metrics.csv", "w") as f:
        f.write("Metric,Value\n")
        for key, value in eval_info.items():
            f.write(f"{key},{value}\n")