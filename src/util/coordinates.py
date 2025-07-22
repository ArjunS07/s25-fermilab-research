import torch
from jetnet.utils import EtaPhiPtE_to_cartesian

def transform_rel_particle_coordinates_to_cartesian(X):
    """
    Transforms relative particle coordinates to absolute Cartesian coordinates using the JetNet relEtaPhiPt_to_cartesian utility function

    Requires X to be a list of length N_jets where each item is a tuple (particle_features, jet_features)
    where particle_features is of shape (n_particles, n_particle_features)
    and jet_features is of length n_jet_features

    Particle features need to start as etarel, phirel, ptrel
    Jet features need to start as eta, pt, mass

    The function generates random phi-values for jets taking into account the azimuthal symmetry of the collider
    """

    particle_polarrel_features = X[:][0][:, :, :3]
    masks = X[:][0][:, :, 3] 
    
    # Phi has to be the second column for the JetNet utility function
    jet_eta = (X[:][1][:, 0]).unsqueeze(1)
    jet_phi_vals = (2 * torch.pi) * torch.rand(len(X)).unsqueeze(1)
    jet_pt_ec = X[:][1][:, 1:3]
    jet_features = torch.concat([jet_eta, jet_phi_vals, jet_pt_ec], dim=-1)

    # Because of issues with the JetNet utility implementation, we do the conversion ourselves
    eta_rel, phi_rel, pt_rel = torch.unbind(particle_polarrel_features, axis=-1)
    Eta, Phi, Pt, _ = torch.unbind(jet_features, axis=-1)

    pt = pt_rel * Pt.unsqueeze(1)
    eta = eta_rel + Eta.unsqueeze(1)
    phi = phi_rel + Phi.unsqueeze(1)
    p0 = pt * torch.cosh(eta)

    stacked = torch.stack([eta, phi, pt, p0], axis=-1)

    cartesian_feats = EtaPhiPtE_to_cartesian(stacked)
    
    # Return the Cartesian coordinates and the masks
    return torch.cat([cartesian_feats, masks.unsqueeze(-1)], dim=-1)