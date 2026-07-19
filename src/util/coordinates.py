import torch
from jetnet.utils import EtaPhiPtE_to_cartesian


def transform_rel_particle_coordinates_to_cartesian(X, jet_phi=None):
    """Convert JetNet relative particle coordinates to absolute Cartesian vectors.

    ``X`` is ``(particle_features, jet_features)``. Particle features begin with
    ``(eta_rel, phi_rel, pt_rel, mask)`` and jet features begin with
    ``(eta, pt, mass)``. JetNet does not store the global jet azimuth, so callers
    can supply the orientation used elsewhere in the same training/inference pair.
    """
    particle_polarrel_features = X[:][0][:, :, :3]
    masks = X[:][0][:, :, 3]

    jet_eta = X[:][1][:, 0].unsqueeze(1)
    jet_phi_vals = (
        (2 * torch.pi) * torch.rand(len(X), device=jet_eta.device)
        if jet_phi is None else jet_phi.to(jet_eta.device)
    ).unsqueeze(1)
    jet_pt_ec = X[:][1][:, 1:3]
    jet_features = torch.concat([jet_eta, jet_phi_vals, jet_pt_ec], dim=-1)

    eta_rel, phi_rel, pt_rel = torch.unbind(particle_polarrel_features, dim=-1)
    jet_eta, jet_phi, jet_pt, _ = torch.unbind(jet_features, dim=-1)
    pt = pt_rel * jet_pt.unsqueeze(1)
    eta = eta_rel + jet_eta.unsqueeze(1)
    phi = phi_rel + jet_phi.unsqueeze(1)
    energy = pt * torch.cosh(eta)

    polar_abs = torch.stack([eta, phi, pt, energy], dim=-1)
    cartesian = EtaPhiPtE_to_cartesian(polar_abs)
    return torch.cat([cartesian, masks.unsqueeze(-1)], dim=-1)


def build_reference_vectors(jet_eta, jet_pt, final_scale, device, jet_phi=None):
    """Build ``(e_t, jet_p4)`` from the physical massless conditioning jet.

    Pass the same ``jet_phi`` used to orient particles and the aligned prior. If
    omitted, a uniform orientation is sampled. Returns shape ``(B, 2, 4)``.
    """
    batch = jet_eta.shape[0]
    phi = (
        (2 * torch.pi) * torch.rand(batch, device=device)
        if jet_phi is None else jet_phi.to(device)
    )
    energy = jet_pt * torch.cosh(jet_eta)
    polar = torch.stack([jet_eta, phi, jet_pt, energy], dim=-1)
    jet_p4 = EtaPhiPtE_to_cartesian(polar) / final_scale
    e_t = torch.zeros(batch, 4, device=device, dtype=jet_p4.dtype)
    e_t[:, 0] = 1.0
    return torch.stack([e_t, jet_p4], dim=1)
