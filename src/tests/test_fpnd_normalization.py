"""Guards for the canonical FPND-input normalization (util.fpnd_input.build_fpnd_input).

The FPND input must normalize pt_rel by the *true clustered jet pT* (from the
conditioning), not by the vector sum of the 30 retained constituents. The latter
(jetnet's EtaPhiPtE_to_relEtaPhiPt) inflates pt_rel ~10% and blew historical FPND
up to ~20 even for real jets. build_fpnd_input is deliberately jetnet-free so these
math checks run locally; the real-jet round-trip at the bottom is cluster-gated
(needs jetnet + torch_geometric + energyflow, absent locally).
"""
import math

import pytest
import torch

from util.fpnd_input import build_fpnd_input


def _polar_abs(eta, phi, pt):
    """Assemble (N,P,4) absolute (eta,phi,pt,E); E is unused by build_fpnd_input."""
    E = pt * torch.cosh(eta)
    return torch.stack([eta, phi, pt, E], dim=-1)


def _one_jet():
    # 3 real constituents (Σpt=90) + 1 padded row; true jet pT=100 (soft radiation
    # dropped by the 30-particle truncation lives in the 100 vs 90 gap).
    eta = torch.tensor([[1.10, 0.90, 1.05, 0.0]])
    phi = torch.tensor([[0.10, -0.10, 0.05, 0.0]])
    pt = torch.tensor([[40.0, 30.0, 20.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    jet_eta = torch.tensor([1.0])
    jet_pt = torch.tensor([100.0])
    return _polar_abs(eta, phi, pt), jet_eta, jet_pt, mask


def test_ptrel_normalized_by_true_jet_pt():
    polar_abs, jet_eta, jet_pt, mask = _one_jet()
    out = build_fpnd_input(polar_abs, jet_eta, jet_pt, mask)
    pt_rel = out[..., 2]
    # exact per-particle pt_i / jet_pt
    assert torch.allclose(pt_rel[0, :3], torch.tensor([0.40, 0.30, 0.20]), atol=1e-5)
    # Σ pt_rel = Σ pt_i / jet_pt = 90/100 = 0.9  (< 1, the canonical hallmark)
    assert abs(pt_rel[0, :3].sum().item() - 0.9) < 1e-5
    assert pt_rel[0, :3].sum().item() < 0.95


def test_corrected_sum_below_vector_sum_convention():
    """The fix deflates pt_rel relative to jetnet's vector-sum denominator."""
    polar_abs, jet_eta, jet_pt, mask = _one_jet()
    out = build_fpnd_input(polar_abs, jet_eta, jet_pt, mask)
    our_sum = out[0, :3, 2].sum().item()

    # vector-sum jet pT (what EtaPhiPtE_to_relEtaPhiPt would divide by)
    eta, phi, pt = polar_abs[..., 0], polar_abs[..., 1], polar_abs[..., 2]
    px = (pt * torch.cos(phi) * mask).sum()
    py = (pt * torch.sin(phi) * mask).sum()
    vec_sum_pt = torch.sqrt(px * px + py * py).item()
    vector_sum_convention_sum = (pt * mask).sum().item() / vec_sum_pt

    assert vector_sum_convention_sum > 0.99          # ~1.0, the inflated convention
    assert our_sum < vector_sum_convention_sum        # our fix is lower (toward canonical)


def test_eta_rel_about_true_axis():
    polar_abs, jet_eta, jet_pt, mask = _one_jet()
    out = build_fpnd_input(polar_abs, jet_eta, jet_pt, mask)
    eta_rel = out[..., 0]
    expected = torch.tensor([1.10, 0.90, 1.05]) - 1.0
    assert torch.allclose(eta_rel[0, :3], expected, atol=1e-5)


def test_padded_rows_zero_and_finite_even_with_nan():
    polar_abs, jet_eta, jet_pt, mask = _one_jet()
    # cartesian_to_EtaPhiPtE of a zero 4-vector yields eta = nan on padded rows.
    polar_abs = polar_abs.clone()
    polar_abs[0, 3, 0] = float("nan")
    out = build_fpnd_input(polar_abs, jet_eta, jet_pt, mask)
    assert torch.isfinite(out).all()
    assert torch.all(out[0, 3] == 0.0)


def test_phi_rel_wrapped_to_pi():
    eta = torch.tensor([[0.0, 0.0]])
    phi = torch.tensor([[3.0, -3.0]])          # near ±pi
    pt = torch.tensor([[10.0, 10.0]])
    mask = torch.tensor([[1.0, 1.0]])
    out = build_fpnd_input(_polar_abs(eta, phi, pt), torch.tensor([0.0]),
                           torch.tensor([20.0]), mask)
    phi_rel = out[..., 1]
    assert torch.all(phi_rel.abs() <= math.pi + 1e-6)


def test_batch_shapes_and_alignment():
    polar_abs, jet_eta, jet_pt, mask = _one_jet()
    polar_abs = polar_abs.repeat(5, 1, 1)
    out = build_fpnd_input(polar_abs, jet_eta.repeat(5), jet_pt.repeat(5), mask.repeat(5, 1))
    assert out.shape == (5, 4, 3)
    assert torch.all(out[:, 3] == 0.0)          # padded row zero for every jet


# ── Cluster-gated: real gluon jets round-tripped through the FPND input stay low ──
def test_real_jets_roundtrip_low_fpnd_cluster_only():
    """Real JetNet g30 jets → build_fpnd_input → FPND should be ≪ the ~20 artifact.

    Needs jetnet + torch_geometric (ParticleNet) + energyflow; all absent locally,
    so this skips off-cluster.
    """
    pytest.importorskip("jetnet")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("energyflow")
    from jetnet.datasets import JetNet
    import jetnet.evaluation as jetnet_eval

    data = JetNet(
        jet_type=["g"],
        data_dir="/mnt/data/caches/jetnet",
        particle_features=JetNet.ALL_PARTICLE_FEATURES,   # [etarel, phirel, ptrel, mask]
        jet_features=["eta", "pt", "mass", "num_particles", "type"],
        num_particles=30,
        split="valid",
        download=True,
    )
    pf, jf = data[:]
    pf, jf = torch.as_tensor(pf).double(), torch.as_tensor(jf).double()
    etarel, phirel, ptrel, mask = pf[..., 0], pf[..., 1], pf[..., 2], pf[..., 3]
    jet_eta, jet_pt = jf[:, 0], jf[:, 1]
    jet_phi = (2 * math.pi) * torch.rand(len(jf), dtype=torch.float64)

    eta = etarel + jet_eta.unsqueeze(1)
    phi = torch.remainder(phirel + jet_phi.unsqueeze(1) + math.pi, 2 * math.pi) - math.pi
    pt = ptrel * jet_pt.unsqueeze(1)
    polar_abs = _polar_abs(eta, phi, pt)

    fpnd_input = build_fpnd_input(polar_abs, jet_eta, jet_pt, mask)
    val = jetnet_eval.fpnd(jets=fpnd_input.float(), jet_type="g", use_tqdm=False)
    assert val < 1.0, f"real-jet round-trip FPND {val} not < 1.0 — normalization still off"
