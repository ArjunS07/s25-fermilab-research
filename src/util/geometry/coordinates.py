import torch


def eta_phi_pt_e_to_cartesian(values):
    """Convert ``(..., eta, phi, pT, E)`` to ``(..., E, px, py, pz)``."""
    eta, phi, pt, energy = values.unbind(dim=-1)
    return torch.stack([
        energy,
        pt * torch.cos(phi),
        pt * torch.sin(phi),
        pt * torch.sinh(eta),
    ], dim=-1)


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
    cartesian = eta_phi_pt_e_to_cartesian(polar_abs)
    return torch.cat([cartesian, masks.unsqueeze(-1)], dim=-1)


def build_reference_vectors(jet_eta, jet_pt, final_scale, device, jet_phi=None,
                            jet_mass=None):
    """Build typed ``(e_t, jet_p4)`` references from conditioning attributes.

    Pass the same ``jet_phi`` used to orient particles and the aligned prior. If
    omitted, a uniform orientation is sampled. ``jet_mass=None`` preserves the legacy
    massless reference. Supplying mass constructs the physical massive jet four-vector.
    Returns shape ``(B, 2, 4)`` in model-scaled units for the jet reference.
    """
    batch = jet_eta.shape[0]
    phi = (
        (2 * torch.pi) * torch.rand(batch, device=device)
        if jet_phi is None else jet_phi.to(device)
    )
    mass = (torch.zeros_like(jet_pt) if jet_mass is None else jet_mass.to(device))
    transverse_mass = torch.sqrt(jet_pt.square() + mass.square())
    energy = transverse_mass * torch.cosh(jet_eta)
    pz = transverse_mass * torch.sinh(jet_eta)
    jet_p4 = torch.stack([
        energy,
        jet_pt * torch.cos(phi),
        jet_pt * torch.sin(phi),
        pz,
    ], dim=-1) / final_scale
    e_t = torch.zeros(batch, 4, device=device, dtype=jet_p4.dtype)
    e_t[:, 0] = 1.0
    return torch.stack([e_t, jet_p4], dim=1)

def deterministic_jet_phi(n_jets: int, seed: int = 42) -> torch.Tensor:
    """Return an index-stable azimuth assignment without touching global RNG state.

    Exact paired-prior caches are only meaningful when cache construction and
    training rotate each dataset row by the same azimuth.  A private CPU
    generator makes that contract independent of CUDA initialization and
    any random draws performed earlier in either process.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return (2 * torch.pi) * torch.rand(n_jets, generator=generator)
