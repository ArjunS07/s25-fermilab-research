import torch
import numpy as np

def gen_initial_distribution(x_1 = None, current_batch_size = None, num_particles=None, num_particle_features=None, std=1, clamp_stddevs=None):

    if x_1 is not None:
        current_batch_size, num_particles = x_1.shape[:2]

    E_c = torch.exp(torch.randn(current_batch_size, num_particles)) 
    theta = torch.arccos(1 - 2*torch.rand(current_batch_size, num_particles))
    phi = 2*torch.pi*torch.rand(current_batch_size, num_particles)
    p_x = E_c * torch.sin(theta) * torch.cos(phi)
    p_y = E_c * torch.sin(theta) * torch.sin(phi)
    p_z = E_c * torch.cos(theta)
    p = torch.stack([E_c, p_x, p_y, p_z], dim=2)
    return p