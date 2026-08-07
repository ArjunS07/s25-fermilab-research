import torch
from util.coordinates import eta_phi_pt_e_to_cartesian


def gen_initial_distribution(x_1 = None, batch_size = None, num_particles=None, prior_dist='isotropic_com', jet_features=None, jet_phi=None, device='cpu', model_scale=None):

    if x_1 is not None:
        batch_size, num_particles = x_1.shape[:2]
        device = x_1.device

    if prior_dist == 'isotropic_com':
        n_pairs = num_particles // 2
    
        # random directions and magnitudes
        E_c = 0.5 + 0.5 * torch.exp(0.5 * torch.randn(batch_size, n_pairs))
        theta = torch.arccos(1 - 2*torch.rand(batch_size, n_pairs))
        phi = 2*torch.pi*torch.rand(batch_size, n_pairs)
        
        p_x = E_c * torch.sin(theta) * torch.cos(phi)
        p_y = E_c * torch.sin(theta) * torch.sin(phi)
        p_z = E_c * torch.cos(theta)
        
        # pair opposite momentum particles
        p_x_opp = -p_x
        p_y_opp = -p_y  
        p_z_opp = -p_z
        E_c_opp = E_c
        
        # interleave pairs consecutively to minimize the damage of the mask to the zero CoM structure
        particles = torch.zeros(batch_size, num_particles, 4, device=device)
        particles[:, 0:2*n_pairs:2] = torch.stack([E_c, p_x, p_y, p_z], dim=-1)
        particles[:, 1:2*n_pairs:2] = torch.stack([E_c_opp, p_x_opp, p_y_opp, p_z_opp], dim=-1)

        current_std = particles.std()
        target_std = 1.0
        particles = particles * (target_std / current_std)
        
        return particles

    elif prior_dist == 'isotropic_lognorm':
        E_c = torch.exp(torch.randn(batch_size, num_particles, device=device))
        theta = torch.arccos(1 - 2*torch.rand(batch_size, num_particles, device=device))
        phi = 2*torch.pi*torch.rand(batch_size, num_particles, device=device)
        p_x = E_c * torch.sin(theta) * torch.cos(phi)
        p_y = E_c * torch.sin(theta) * torch.sin(phi)
        p_z = E_c * torch.cos(theta)
        return torch.stack([E_c, p_x, p_y, p_z], dim=2)

    elif prior_dist == 'jet_ref_frame':
        assert jet_features is not None, "jet_features must be provided for jet_ref_frame method"
        device = jet_features.device
        eta_rel = torch.normal(0, 0.8, size=(batch_size, num_particles), device=device)
        phi_rel = torch.normal(0, 0.8, size=(batch_size, num_particles), device=device)
        pt_rel = torch.exp(torch.randn(batch_size, num_particles, device=device))

        # jet features are ["eta", "pt", "mass", "num_particles", "type"],
        jet_eta = jet_features[:, 0]
        jet_phi = ((2 * torch.pi) * torch.rand(batch_size, device=device) if jet_phi is None
                   else jet_phi.to(device))
        jet_pt = jet_features[:, 1]

        pt = pt_rel * jet_pt.unsqueeze(1)
        eta = eta_rel + jet_eta.unsqueeze(1)
        phi = phi_rel + jet_phi.unsqueeze(1)
        p0 = pt * torch.cosh(eta)

        stacked = torch.stack([eta, phi, pt, p0], axis=-1)
        return eta_phi_pt_e_to_cartesian(stacked)

    elif prior_dist == 'axis_aligned':
        # Collimated, positive-energy, (near-)lightlike prior around the jet axis. Shortens
        # the transport path vs. an isotropic prior. Like jet_ref_frame but with a small
        # angular spread, then normalized to unit std so it lives in the scaled particle
        # space (matching isotropic_com and the final_scale normalization of x_1).
        assert jet_features is not None, "jet_features must be provided for axis_aligned prior"
        device = jet_features.device
        angular_spread = 0.15  # radians/eta units; << jet_ref_frame's 0.8 => collimated
        eta_rel = torch.normal(0, angular_spread, size=(batch_size, num_particles), device=device)
        phi_rel = torch.normal(0, angular_spread, size=(batch_size, num_particles), device=device)
        pt_rel = torch.exp(torch.randn(batch_size, num_particles, device=device))

        jet_eta = jet_features[:, 0]
        jet_phi = ((2 * torch.pi) * torch.rand(batch_size, device=device) if jet_phi is None
                   else jet_phi.to(device))
        jet_pt = jet_features[:, 1]

        pt = pt_rel * jet_pt.unsqueeze(1)
        eta = eta_rel + jet_eta.unsqueeze(1)
        phi = phi_rel + jet_phi.unsqueeze(1)
        p0 = pt * torch.cosh(eta)  # massless => E = |p| > 0 (positive-energy, lightlike)

        stacked = torch.stack([eta, phi, pt, p0], axis=-1)
        particles = eta_phi_pt_e_to_cartesian(stacked)

        # Normalize to unit std (scalar, direction-preserving) to match the scaled space.
        current_std = particles.std().clamp(min=1e-8)
        return particles * (1.0 / current_std)

    elif prior_dist == 'axis_aligned_per_jet':
        # Batch-independent physical prior. Positive lognormal weights are normalized within
        # each jet, and one scalar correction makes the transverse vector sum match the
        # conditioning pT exactly. The result is then put directly in model-scaled units.
        if jet_features is None or model_scale is None:
            raise ValueError("axis_aligned_per_jet requires jet_features and model_scale")
        device = jet_features.device
        jet_eta = jet_features[:, 0]
        jet_pt = jet_features[:, 1]
        phi0 = ((2 * torch.pi) * torch.rand(batch_size, device=device)
                if jet_phi is None else jet_phi.to(device))
        eta = jet_eta.unsqueeze(1) + 0.15 * torch.randn(batch_size, num_particles, device=device)
        phi = phi0.unsqueeze(1) + 0.15 * torch.randn(batch_size, num_particles, device=device)
        weights = torch.softmax(torch.randn(batch_size, num_particles, device=device), dim=1)
        ux, uy = torch.cos(phi), torch.sin(phi)
        resultant = torch.sqrt((weights * ux).sum(1).square() + (weights * uy).sum(1).square())
        pt = weights * (jet_pt / resultant.clamp(min=1e-4)).unsqueeze(1)
        energy = pt * torch.cosh(eta)
        particles = eta_phi_pt_e_to_cartesian(torch.stack([eta, phi, pt, energy], dim=-1))
        return particles / float(model_scale)

    else:
        raise ValueError(f"Unknown prior_dist: {prior_dist}")



def time_dist(batch_size, device='cpu', mode='power_law', **kwargs):
    if mode == 'uniform':
        return torch.rand(batch_size, device=device)
    elif mode == 'power_law':
        # pdf = (1+a) * x^a
        # cdf = x^(a+1)
        # https://math.stackexchange.com/questions/3499892/sampling-from-a-distribution-with-given-pdf
        a = kwargs.get('a', -0.2)
        # sample x uniformly over 0, 1
        u = torch.rand(batch_size, device=device)
        return u ** (1 / (a + 1))
    elif mode == 'lognorm':
        mu = kwargs.get('mu', -0.5)
        sigma = kwargs.get('sigma', 1.0)
        dist = torch.distributions.LogNormal(mu, sigma)
        samples = dist.sample((batch_size,)).to(device)
        # Normalize using the theoretical 95th percentile of this LogNormal so the
        # scaling is batch-independent. p95 = exp(mu + sigma * 1.645).
        p95 = torch.exp(torch.tensor(mu + sigma * 1.645, device=device))
        samples = (samples / p95).clamp(max=1.0)
        return samples
    else:
        raise ValueError(f"Unknown time distribution mode: {mode}")
