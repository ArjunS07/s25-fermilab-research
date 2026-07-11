import torch
from jetnet.utils import EtaPhiPtE_to_cartesian, cartesian_to_EtaPhiPtE

def transform_rel_particle_coordinates_to_cartesian(X):
    """
    Transforms relative particle coordinates to absolute Cartesian coordinates using the JetNet relEtaPhiPt_to_cartesian utility function

    Requires X to be a list of length N_jets where each item is a tuple (particle_features, jet_features)
    where particle_features is of shape (n_particles, n_particle_features)
    and jet_features is of length n_jet_features

    Particle features need to start as etarel, phirel, ptrel. Jet features need to start as eta, pt, mass

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


def build_reference_vectors(jet_eta, jet_pt, final_scale, device):
    """Inference reference 4-vectors: e_t=(1,0,0,0) and a reconstructed jet 4-momentum.

    Mirrors transform_rel_particle_coordinates_to_cartesian: the jet axis is
    (jet_eta, random phi, jet_pt) with energy E = pt*cosh(eta), converted with the same jetnet
    routine used to build the training particles, then divided by final_scale to enter the
    model's scaled space. This matches train.py, where the reference is the sum of the scaled
    constituents (= physical jet 4-momentum / final_scale). The random phi is the *chosen* jet
    orientation the references then induce in the generated cloud.

    Returns (B, 2, 4).
    """
    batch = jet_eta.shape[0]
    phi = (2 * torch.pi) * torch.rand(batch, device=device)
    energy = jet_pt * torch.cosh(jet_eta)
    stacked = torch.stack([jet_eta, phi, jet_pt, energy], dim=-1)  # (B, 4) = (eta, phi, pt, E)
    jet_p4 = EtaPhiPtE_to_cartesian(stacked) / final_scale         # (B, 4) scaled (E, px, py, pz)
    e_t = torch.zeros(batch, 4, device=device, dtype=jet_p4.dtype)
    e_t[:, 0] = 1.0
    return torch.stack([e_t, jet_p4], dim=1)  # (B, 2, 4)


def jacobian_epp_etaphipte(eppp):
    """
    Returns Jacobian for converting a Cartesian vector at Cartesian cooordinates E, p_x, p_y, p_z to a polar vector eta, phi, p_T, E

    eppp: batch_size, particles_per_jet, 4
    """
    E, p_x, p_y, p_z = eppp[..., 0], eppp[..., 1], eppp[..., 2], eppp[..., 3]

    p_T = torch.sqrt(p_x**2 + p_y**2)
    p_T = p_T + torch.finfo(p_T.dtype).eps  # Avoid division by zero

    # Cache repeated values
    p_T_sq = p_T ** 2
    zero = torch.zeros_like(E)
    reciprocal_p_T = 1 / p_T
    reciprocal_p_T_sq = reciprocal_p_T ** 2
    reciprocal_p_T_cubed = reciprocal_p_T ** 3
    deta_denom = 1 / (torch.sqrt(
        1 + ((p_z**2) * reciprocal_p_T_sq)
    ))

    # First row: deta/d[E, p_x, p_y, p_z]
    deta_dE = zero
    deta_dpx = (-1 * p_x * p_z) * reciprocal_p_T_cubed * deta_denom
    deta_dpy = (-1 * p_y * p_z) * reciprocal_p_T_cubed * deta_denom
    deta_dpz = reciprocal_p_T * deta_denom

    # Second row: dphi/d[E, p_x, p_y, p_z]
    dphi_dE = zero
    dphi_dpx = -p_y * reciprocal_p_T_sq
    dphi_dpy = p_x * reciprocal_p_T_sq
    dphi_dpz = zero

    # Third row: dp_t/d[E, p_x, p_y, p_z]
    dpT_dE = zero
    dpT_dpx = p_x * reciprocal_p_T
    dpT_dpy = p_y * reciprocal_p_T
    dpT_dpz = zero

    # Fourth row: dE/d[E, p_x, p_y, p_z]
    dE_dE = torch.ones_like(E)
    dE_dpx = zero
    dE_dpy = zero
    dE_dpz = zero

    return torch.stack([
        torch.stack([deta_dE, deta_dpx, deta_dpy, deta_dpz], dim=-1),  
        torch.stack([dphi_dE, dphi_dpx, dphi_dpy, dphi_dpz], dim=-1),  
        torch.stack([dpT_dE, dpT_dpx, dpT_dpy, dpT_dpz], dim=-1),      
        torch.stack([dE_dE, dE_dpx, dE_dpy, dE_dpz], dim=-1)           
    ], dim=-2)
    

if __name__=="__main__":
    import time

    def __jacobian_naive(eppp, etaphipte):
        """
        Returns Jacobian for converting Cartesian cooordinates E, p_x, p_y, p_z to polar eta, phi, p_T, E

        epp: batch_size, particles_per_jet, 4
        etaphipte: batch_size, particles_per_jet, 4
        """
        E, p_x, p_y, p_z = eppp[..., 0], eppp[..., 1], eppp[..., 2], eppp[..., 3]
        eta, phi, p_T, E = etaphipte[..., 0], etaphipte[..., 1], etaphipte[..., 2], etaphipte[..., 3]

        p_T = torch.clamp(p_T, 1e-10)
        zero, one = torch.zeros_like(E), torch.ones_like(E)

        jac_E_col = torch.stack([zero, zero, zero, one], dim=-1)
        jac_p_x_col = torch.stack([
            (-1 * p_x * p_z) / (p_T**3 * torch.sqrt((p_z**2)/(p_T**2) + 1)),
            -p_y / (p_T**2),
            p_x / p_T,
            zero,
        ], dim=-1)
        jac_p_y_col = torch.stack([
            (-1 * p_y * p_z) / (p_T**3 * torch.sqrt((p_z**2)/(p_T**2) + 1)),
            -p_x / (p_T**2),
            p_y / p_T,
            zero
        ], dim=-1)
        jac_p_z_col = torch.stack([
            (1) / (p_T * torch.sqrt((p_z**2 / p_T**2) + 1)),
            zero,
            zero,
            zero
        ], dim=-1)

        return torch.stack([
            jac_E_col,
            jac_p_x_col,
            jac_p_y_col,
            jac_p_z_col
        ], dim=-2)

    batch_size, particles_per_jet = 50, 150
    pred_cartesian = torch.randn(batch_size, particles_per_jet, 4)
    pred_cartesian[..., 0] = torch.abs(pred_cartesian[..., 0]) + 1.0
    pred_polar = cartesian_to_EtaPhiPtE(pred_cartesian)

    start_time = time.process_time()
    jac = jacobian_epp_etaphipte(pred_cartesian, pred_polar)
    optimized_time = time.process_time() - start_time

    start_time = time.process_time()
    jac = __jacobian_naive(pred_cartesian, pred_polar)
    naive_time = time.process_time() - start_time

    diff = naive_time - optimized_time
    speedup_pct = ((diff) / naive_time) * 100
    print(f"{optimized_time=} s {naive_time=} s")
    print(f"Speedup: {speedup_pct}% ({diff} (s))")