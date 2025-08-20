import torch
from jetnet.utils import EtaPhiPtE_to_cartesian

def gen_initial_distribution(x_1 = None, batch_size = None, num_particles=None, prior_dist='isotropic_lognorm', jet_features=None):

    if x_1 is not None:
        batch_size, num_particles = x_1.shape[:2]

    if prior_dist == 'isotropic_lognorm':
        E_c = torch.exp(torch.randn(batch_size, num_particles))
        theta = torch.arccos(1 - 2*torch.rand(batch_size, num_particles))
        phi = 2*torch.pi*torch.rand(batch_size, num_particles)
        p_x = E_c * torch.sin(theta) * torch.cos(phi)
        p_y = E_c * torch.sin(theta) * torch.sin(phi)
        p_z = E_c * torch.cos(theta)
        p = torch.stack([E_c, p_x, p_y, p_z], dim=2)

    elif prior_dist == 'jet_ref_frame':
        assert jet_features is not None, "jet_features must be provided for jet_ref_frame method"
        eta_rel = torch.normal(0, 0.8, size=(batch_size, num_particles))
        phi_rel = torch.normal(0, 0.8, size=(batch_size, num_particles))
        # sample pt_rel log-normally
        pt_rel = torch.exp(torch.randn(batch_size, num_particles))

        # jet features are ["eta", "pt", "mass", "num_particles", "type"],
        jet_eta = jet_features[:, 0]
        jet_phi = (2 * torch.pi) * torch.rand(batch_size)
        jet_pt = jet_features[:, 1]

        pt = pt_rel * jet_pt.unsqueeze(1)
        eta = eta_rel + jet_eta.unsqueeze(1)
        phi = phi_rel + jet_phi.unsqueeze(1)
        p0 = pt * torch.cosh(eta)

        stacked = torch.stack([eta, phi, pt, p0], axis=-1)
        p = EtaPhiPtE_to_cartesian(stacked)

    return p