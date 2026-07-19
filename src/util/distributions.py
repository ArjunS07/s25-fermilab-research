import torch
from jetnet.utils import EtaPhiPtE_to_cartesian
from util.hyperbolic import (
    to_poincare_ball, from_poincare_ball,
    geodesic_interpolant, conditional_vector_field, pushforward,
)


def generate_x_0_com_frame(x_1=None, batch_size=None, num_particles=None, device='cpu'):
    """Generate noise that naturally has zero total momentum"""

    if x_1 is not None:
        batch_size, num_particles = x_1.shape[:2]
        device = x_1.device
    elif (batch_size is None) or (num_particles is None):
        raise ValueError("Either x_1 or (batch_size and num_particles) must be provided.")
    raise NotImplementedError(
        "generate_x_0_com_frame is not yet implemented. "
        "Use gen_initial_distribution with prior_dist='isotropic_com' instead."
    )

    

def gen_initial_distribution(x_1 = None, batch_size = None, num_particles=None, prior_dist='isotropic_com', jet_features=None, jet_phi=None, device='cpu'):

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
        return EtaPhiPtE_to_cartesian(stacked)

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
        particles = EtaPhiPtE_to_cartesian(stacked)

        # Normalize to unit std (scalar, direction-preserving) to match the scaled space.
        current_std = particles.std().clamp(min=1e-8)
        return particles * (1.0 / current_std)

    else:
        raise ValueError(f"Unknown prior_dist: {prior_dist}")



def hyperbolic_interpolant(x_0, x_1, t, c=1.0):
    """
    Compute (x_t, y_t, u_t_ball) for Riemannian flow matching in the Poincaré ball.

    Maps Cartesian 4-momenta to the ball, computes the geodesic interpolant
    and conditional vector field entirely in the ball.

    x_0, x_1 : (batch, max_particles, 4)  normalized Cartesian 4-momenta
    t         : (batch,) in [0, 1)
    c         : Poincaré ball curvature (default 1.0)

    Returns
    -------
    x_t      : (batch, max_particles, 4)  interpolated point in Cartesian space
                                          (feed this to the model)
    y_t      : (batch, max_particles, 4)  interpolated point in the Poincaré ball
                                          (used with hyperbolic_loss)
    u_t_ball : (batch, max_particles, 4)  conditional vector field as a tangent
                                          vector at y_t in the ball (loss target)
    """
    y_0 = to_poincare_ball(x_0, c=c)
    y_1 = to_poincare_ball(x_1, c=c)

    y_t = geodesic_interpolant(y_0, y_1, t, c=c)
    u_t_ball = conditional_vector_field(y_t, y_1, t, c=c)

    x_t = from_poincare_ball(y_t, c=c)
    return x_t, y_t, u_t_ball


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
