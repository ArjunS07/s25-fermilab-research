import torch
from jetnet.utils import relEtaPhiPt_to_EtaPhiPt, EtaPhiPtE_to_cartesian

def gen_initial_distribution(x_1 = None, batch_size = None, num_particles=None, method='isotropic_lognorm', jet_features=None):

    if x_1 is not None:
        batch_size, num_particles = x_1.shape[:2]

    if method == 'isotropic_lognorm':
        E_c = torch.exp(torch.randn(batch_size, num_particles))
        theta = torch.arccos(1 - 2*torch.rand(batch_size, num_particles))
        phi = 2*torch.pi*torch.rand(batch_size, num_particles)
        p_x = E_c * torch.sin(theta) * torch.cos(phi)
        p_y = E_c * torch.sin(theta) * torch.sin(phi)
        p_z = E_c * torch.cos(theta)
        p = torch.stack([E_c, p_x, p_y, p_z], dim=2)

    elif method == 'jet_ref_frame':
        assert jet_features is not None, "jet_features must be provided for jet_ref_frame method"
        eta_rel = torch.normal(0, 0.8, size=(batch_size, num_particles))
        phi_rel = torch.normal(0, 0.8, size=(batch_size, num_particles))
        # sample pt_rel log-normally
        pt_rel = torch.exp(torch.randn(batch_size, num_particles))

        # jet features need to be in the order pt, eta, mass
        jet_eta = jet_features[:, 1].unsqueeze(1)
        jet_phi = (2 * torch.pi) * torch.rand(batch_size).unsqueeze(1)
        jet_pt = jet_features[:, 0].unsqueeze(1)
        jet_mass = jet_features[:, 2].unsqueeze(1)

        relEtaPhiPt = torch.stack([eta_rel, phi_rel, pt_rel], dim=2)
        jet_features = torch.stack([jet_eta, jet_phi, jet_pt, jet_mass], dim=1)
        p = relEtaPhiPt_to_EtaPhiPt(p_polarrel=relEtaPhiPt, jet_features=jet_features, jet_coord='polar')
        p = EtaPhiPtE_to_cartesian(p)

    return p