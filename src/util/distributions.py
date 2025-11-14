import torch
from jetnet.utils import EtaPhiPtE_to_cartesian


def generate_x_0_com_frame(x_1=None, batch_size=None, num_particles=None, device='cpu'):
    """Generate noise that naturally has zero total momentum"""

    if x_1 is not None:
        batch_size, num_particles = x_1.shape[:2]
        device = x_1.device
    elif (batch_size is None) or (num_particles is None):
        raise ValueError("Either x_1 or (batch_size and num_particles) must be provided.")

    

def gen_initial_distribution(x_1 = None, batch_size = None, num_particles=None, prior_dist='isotropic_com', jet_features=None, device='cpu'):

    if x_1 is not None:
        batch_size, num_particles = x_1.shape[:2]
        device = x_1.device

    if prior_dist == 'isotropic_com':
        n_pairs = num_particles // 2
    
        # Generate random directions and magnitudes
        E_c = 0.5 + 0.5 * torch.exp(0.5 * torch.randn(batch_size, n_pairs))
        theta = torch.arccos(1 - 2*torch.rand(batch_size, n_pairs))
        phi = 2*torch.pi*torch.rand(batch_size, n_pairs)
        
        p_x = E_c * torch.sin(theta) * torch.cos(phi)
        p_y = E_c * torch.sin(theta) * torch.sin(phi)
        p_z = E_c * torch.cos(theta)
        
        # pair particles with opposite momentum
        p_x_opp = -p_x
        p_y_opp = -p_y  
        p_z_opp = -p_z
        E_c_opp = E_c
        
        # Interleave pairs so that the mask doesn't destroy the zero CoM structure
        particles = torch.zeros(batch_size, num_particles, 4, device=device)
        particles[:, 0::2] = torch.stack([E_c, p_x, p_y, p_z], dim=-1)
        particles[:, 1::2] = torch.stack([E_c_opp, p_x_opp, p_y_opp, p_z_opp], dim=-1)
        
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
        jet_phi = (2 * torch.pi) * torch.rand(batch_size, device=device)
        jet_pt = jet_features[:, 1]

        pt = pt_rel * jet_pt.unsqueeze(1)
        eta = eta_rel + jet_eta.unsqueeze(1)
        phi = phi_rel + jet_phi.unsqueeze(1)
        p0 = pt * torch.cosh(eta)

        stacked = torch.stack([eta, phi, pt, p0], axis=-1)
        return EtaPhiPtE_to_cartesian(stacked)
    

def time_dist(batch_size, device='cpu', mode='power_law', *args, **kwargs):
    if mode == 'uniform':
        return torch.rand(batch_size, device=device)
    elif mode == 'power_law':
        # pdf = (1+a) * x^a
        # cdf = x^(a+1)
        # https://math.stackexchange.com/questions/3499892/sampling-from-a-distribution-with-given-pdf
        a = kwargs.get('a', 0.2)
        # sample x uniformly over 0, 1
        u = torch.rand(batch_size, device=device)
        return u ** (1 / (a + 1))
    elif mode == 'lognorm':
        #-lognorm(-0.5, 1)
        mu = kwargs.get('mu', -0.5)
        sigma = kwargs.get('sigma', 1.0)
        dist = torch.distributions.LogNormal(mu, sigma)
        samples = dist.sample((batch_size,)).to(device)
        samples = samples / samples.max()
        return samples
    else:
        raise ValueError(f"Unknown time distribution mode: {mode}")