import torch
import torch.nn as nn
import torch.nn.functional as F

from models.NewLEFM import TimeEmbedding

class FlowMatchingMLP(nn.Module):
    def __init__(self, 
                 n_particles=150,
                 particle_dim=1,  # 4-momentum
                 global_dim=64,
                 n_layers=6,
                 hidden_dim=128,
                 n_jet_types=5,  # Maximum number of jet types used during training of NF model
                 time_embed_dim=64,
                 gradient_clip_val=1.0):
        super().__init__()
        
        self.conditioning_dim = n_jet_types + 4
        self.time_embedding = TimeEmbedding(embed_dim=time_embed_dim)
        self.global_init = nn.Sequential(
            nn.Linear(time_embed_dim + self.conditioning_dim, global_dim),
            nn.BatchNorm1d(global_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        )

        self.particle_dim = 1
        # Input: particle features + time embedding
        input_dim = (global_dim) + (n_particles * self.particle_dim)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            d_in = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(d_in, hidden_dim))
            activation = nn.LeakyReLU(0.2, inplace=True) if i < n_layers - 1 else nn.Tanh()
            self.layers.append(activation)
            self.layers.append(nn.Dropout(0.1))
        self.final_layer = nn.Linear(hidden_dim, n_particles * self.particle_dim)

        self.gradient_clip_val = gradient_clip_val


    def forward(self, x, t, jet_conditions, mask):
        """
        Args:
            x: (batch_size, n_particles, particle_dim) - current particle states
            t: (batch_size,) - time values
            jet_conditions: (batch_size, conditioning_dim) - jet conditioning info
            mask: (batch_size, n_particles) - binary mask for valid particles
            
        Returns:
            velocity: (batch_size, n_particles, particle_dim) - velocity field v_theta(t, x)
        """

        t_embed = self.time_embedding(t)  # (batch_size, time_embed_dim)
        
        # Initial global embedding g^0
        global_input = torch.cat([t_embed, jet_conditions], dim=-1)
        g_0 = self.global_init(global_input)  # (batch_size, global_dim)

        # TODO: Only use E/c
        x_current = x[:, :, 0]
        x_current = x_current.unsqueeze(-1)

        x_current = x_current.flatten(start_dim=1)  # Start with the current particle states
        x_current = x_current * mask.repeat_interleave(self.particle_dim, dim=-1)
        x_current = torch.cat([
            g_0, 
            x_current
        ], dim=-1)  

        for i, layer in enumerate(self.layers):
            x_current = layer(x_current)
        
        x_final = self.final_layer(x_current)
        out = torch.zeros_like(x)
        out[:, :, 0] = x_final
        return out
    
    def step(self, x_t, jet_conditions, mask, t_start, t_end, method='euler'):
        """
        Calculate the probability density at a particular time step
        """
        if method == 'euler':
            batch_size = x_t.shape[0]
            update = self.forward(x=x_t, t=t_start.unsqueeze(0).repeat(batch_size), jet_conditions=jet_conditions, mask=mask)
            x_next = x_t + update * (t_end - t_start)

            return x_next
        else:
            raise NotImplementedError(f"Method {method} not implemented")
    
    def clip_gradients(self):
        """Clip gradients for training stability"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)