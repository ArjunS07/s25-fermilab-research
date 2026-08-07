"""Canonical relative-coordinate input for FPND.

FPND is the one metric compared against jetnet's *external* cached ParticleNet
reference, so its input must match JetNet's canonical convention:

    pt_rel  = pt_i / true_clustered_jet_pt      (Σ over constituents ≈ 0.91 for g30,
                                                  because the 30-particle truncation
                                                  drops ~9% of soft-radiation pT)
    eta_rel = eta_i − jet_eta
    phi_rel = wrap(phi_i − jet_phi)

jetnet's convenience `EtaPhiPtE_to_relEtaPhiPt` instead normalizes by the *vector
sum of the 30 retained constituents* (Σ pt_rel ≈ 1.0) and centers on their
vector-sum axis. That ~10% pt_rel inflation pushes every jet past ParticleNet's
bounded input range (fpnd_norm pt_rel max = 0.8935), inflating FPND to ~20 even
for real jets. This helper rebuilds the input from the *conditioning* jet pT and
η (physical GeV, identity-normalized), which are the true clustered quantities.

Kept jetnet-free on purpose so it can be unit-tested without importing
`util.metrics.metrics` (which imports jetnet at module load). The W1/FPD/Cov paths are
gen-vs-test in the vector-sum convention and must keep using it — do NOT route
them through this helper.
"""
import torch


def build_fpnd_input(polar_abs, jet_eta, jet_pt, mask, eps=1e-8):
    """Canonical (eta_rel, phi_rel, pt_rel) FPND input from absolute polar coords.

    Args:
        polar_abs: (N, P, 4) absolute per-particle (eta, phi, pt, E).
        jet_eta:   (N,) true/conditioning jet eta (physical).
        jet_pt:    (N,) true/conditioning jet pT (physical GeV, same units as pt).
        mask:      (N, P) float/bool, 1 = real particle, 0 = padding.

    Returns:
        (N, P, 3) tensor of (eta_rel, phi_rel, pt_rel) with padded rows exactly 0.
    """
    # Padded rows can be non-finite (cartesian_to_EtaPhiPtE of a zero 4-vector gives
    # eta = nan). Sanitize before arithmetic so nan never leaks into the jet-phi sum.
    pa = torch.nan_to_num(polar_abs)
    eta = pa[..., 0]
    phi = pa[..., 1]
    pt = pa[..., 2]

    m = mask.to(pt.dtype)
    if m.dim() == pt.dim() + 1:  # tolerate a trailing singleton
        m = m[..., 0]

    # phi axis: vector-sum of transverse momenta. FPND is invariant to the per-jet
    # phi reference (JetNet does not track jet phi), so any consistent choice works;
    # the vector sum matches jetnet's own centering for the phi channel.
    sum_py = (pt * torch.sin(phi) * m).sum(dim=1)
    sum_px = (pt * torch.cos(phi) * m).sum(dim=1)
    jet_phi = torch.atan2(sum_py, sum_px)

    eta_rel = eta - jet_eta.unsqueeze(1)
    phi_rel = phi - jet_phi.unsqueeze(1)
    phi_rel = torch.remainder(phi_rel + torch.pi, 2 * torch.pi) - torch.pi
    pt_rel = pt / (jet_pt.unsqueeze(1) + eps)

    out = torch.stack([eta_rel, phi_rel, pt_rel], dim=-1)
    # torch.where (not multiply) so padded rows are a clean 0, never 0 * nan.
    return torch.where(m.unsqueeze(-1) > 0, out, torch.zeros_like(out))
