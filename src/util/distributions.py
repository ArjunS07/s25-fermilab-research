import torch
import numpy as np

def sample_massless_4momentum_clouds(n_clouds, cloud_size, device):
    E_c = torch.exp(torch.randn(n_clouds, cloud_size, device=device))  # Sample energy from appropriate distribution
    theta = np.arccos(1 - 2*torch.rand(n_clouds, cloud_size, device=device))
    phi = 2*np.pi*torch.rand(n_clouds, cloud_size, device=device)
    p_x = E_c * np.sin(theta) * np.cos(phi)
    p_y = E_c * np.sin(theta) * np.sin(phi)
    p_z = E_c * np.cos(theta)
    p = torch.stack([E_c, p_x, p_y, p_z], dim=2)
    return p