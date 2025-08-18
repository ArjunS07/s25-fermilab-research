import torch
import numpy as np

def gen_initial_distribution(x_1 = None, current_batch_size = None, num_particles=None, num_particle_features=None, std=1, clamp_stddevs=None):

    # if x_1 is not None:
    #     dist = torch.randn_like(x_1) * std
    # elif current_batch_size is not None and num_particles is not None and num_particle_features is not None:
    #     dist = torch.randn((current_batch_size, num_particles, num_particle_features)) * std
    # else:
    #     raise ValueError("Either x_1 or current_batch_size, num_particles, and num_particle_features must be provided.")

    # if clamp_stddevs is not None:
    #     dist = torch.clamp(dist, -clamp_stddevs * std, clamp_stddevs * std)
    
    # # First column is energy, generate physical particles
    # dist[:, :, 0] = torch.abs(dist[:, :, 0])  # Ensure energy is non-negative
    # return dist

    if x_1 is not None:
        current_batch_size, num_particles = x_1.shape[:2]

    E_c = torch.exp(torch.randn(current_batch_size, num_particles))  # Sample energy from appropriate distribution
    theta = torch.arccos(1 - 2*torch.rand(current_batch_size, num_particles))
    phi = 2*torch.pi*torch.rand(current_batch_size, num_particles)
    p_x = E_c * torch.sin(theta) * torch.cos(phi)
    p_y = E_c * torch.sin(theta) * torch.sin(phi)
    p_z = E_c * torch.cos(theta)
    p = torch.stack([E_c, p_x, p_y, p_z], dim=2)
    return p