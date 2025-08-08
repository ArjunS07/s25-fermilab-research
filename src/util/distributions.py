import torch
import numpy as np

def gen_initial_distribution(x_1 = None, current_batch_size = None, num_particles=None, num_particle_features=None, std=0.1, clamp_min=None, clamp_max=None):

    if x_1 is not None:
        dist = torch.randn_like(x_1) * std
    elif current_batch_size is not None and num_particles is not None and num_particle_features is not None:
        dist = torch.randn((current_batch_size, num_particles, num_particle_features)) * std
    else:
        raise ValueError("Either x_1 or current_batch_size, num_particles, and num_particle_features must be provided.")

    if clamp_min is not None:
        dist = torch.clamp(dist, min=clamp_min)
    if clamp_max is not None:
        dist = torch.clamp(dist, max=clamp_max)

    return dist

def sample_massless_4momentum_clouds(n_clouds, cloud_size):
    E_c = torch.exp(torch.randn(n_clouds, cloud_size))  # Sample energy from appropriate distribution
    theta = np.arccos(1 - 2*torch.rand(n_clouds, cloud_size))
    phi = 2*np.pi*torch.rand(n_clouds, cloud_size)
    p_x = E_c * np.sin(theta) * np.cos(phi)
    p_y = E_c * np.sin(theta) * np.sin(phi)
    p_z = E_c * np.cos(theta)
    p = torch.stack([E_c, p_x, p_y, p_z], dim=2)
    return p