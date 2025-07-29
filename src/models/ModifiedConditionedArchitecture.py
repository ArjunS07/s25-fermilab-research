import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.minkowski_utils import normsq4, dotsq4

from util.jet_attributes import load_model, one_hot_enc_jet_type, generate_jets, generate_masks

def psi(p):
    ''' `\psi(p) = Sgn(p) \cdot \log(|p| + 1)` '''
    # Clamp inputs
    p = torch.clamp(p, -1e4, 1e4)
    return torch.sign(p) * torch.log1p(torch.abs(p))

def minkowski_features(x, mask):
    x_i = x.unsqueeze(-2)  # second-last dimension - N
    x_j = x.unsqueeze(-3)  # third-last dimension - B
    x_diffs = x_i - x_j  # (batch_size, n_particles, n_particles, 4)

    masks = mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    expanded_mask = masks.unsqueeze(-3).expand(-1, x.shape[1], -1, -1)
    diff_masked = x_diffs * expanded_mask
    norms = normsq4(diff_masked)
    dots = dotsq4(x_i, x_j)
    norms, dots = psi(norms), psi(dots)

#     print(f"{norms.mean()=} {norms.std()=} {norms.max()=} {norms.min()=}")
#     print(f"{dots.mean()=} {dots.std()=} {dots.max()=} {dots.min()=}")
    return norms, dots, x_diffs

class TimeEmbedding(nn.Module):
    """
    Random Fourier Features for time embedding following FPCD (Mikuni et al 2023)

    1. Apply a Gaussian  projection to the time input, with fixed weights, up to embed_dim/2.
    2. Apply sin and cos to the projection.
    3. Concatenate the sin and cos outputs.
    4. Pass the embedding through a fully connected layer to 32 nodes
    5. Pass through another fully connected layer to embed_dim.

    Weights for the FC layers are initialized from a Xavier uniform distribution.
    """
    
    def __init__(self, embed_dim=64, scale=16.0, seed=42):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        
        # Randomly sample weights during initialization
        # Fixed weights, non-trainable
        # Shape: (1, embed_dim//2) to match the TF implementation
        torch.manual_seed(seed)
        projection = (torch.randn(1, embed_dim//2) * scale).detach()
        self.register_buffer('projection', projection, persistent=False)
        

        # Fully connected layers: first 32 nodes, then 64 nodes
        # Input will be embed_dim features (sin + cos concatenated)
        self.fc1 = nn.Linear(embed_dim, 32)
        self.fc2 = nn.Linear(32, embed_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, t):
        # t shape: (batch_size, 1) or (batch_size,)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch_size, 1)
        

        projection = self.projection.to(t.device)  
        time_projection = t * projection * 2 * np.pi     # (batch_size, embed_dim//2)

        # Apply sin and cos, then concatenate
        sin_proj = torch.sin(time_projection)
        cos_proj = torch.cos(time_projection)
        time_embed = torch.cat([sin_proj, cos_proj], dim=-1)  # (batch_size, embed_dim)
        
        # Pass through fully connected layers
        x = self.fc1(time_embed)  # (batch_size, 32)
        x = self.fc2(x)          # (batch_size, embed_dim)
        
        return x

class LorentzEquivariantLayer(nn.Module):
    """Single Lorentz-equivariant layer with message passing"""
    def __init__(self, particle_dim, global_dim, hidden_dim=128, gamma_init=1.0):
        super().__init__()
        self.particle_dim = particle_dim
        self.global_dim = global_dim
        
        # Message computation: phi_e
        # Applied to psi of Minkowski features between particles
        self.phi_e = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Input: [norms, dots]
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        )

        self.phi_m = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Input: [norms, dots]
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        )
        
        # Global embedding update phi_g
        self.phi_g = nn.Sequential(
            nn.Linear(2*global_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, global_dim),
            nn.BatchNorm1d(global_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0)) 

        self.phi_x = nn.Sequential(
            nn.Linear(2 * global_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, particle_dim),
            nn.BatchNorm1d(particle_dim),
            nn.Linear(particle_dim, particle_dim),
            nn.BatchNorm1d(particle_dim),
            nn.Linear(particle_dim, 1),
            nn.Tanh(),
            nn.Dropout(0.1)
        )
        self.gamma = nn.Parameter(torch.tensor(gamma_init)) 
        # self.node_norm = nn.LayerNorm(particle_dim)
        self.global_norm = nn.LayerNorm(global_dim)
        
    def forward(self, x, g, g_0, mask):
        """
        Args:
            x: (batch_size, n_particles, particle_dim) - particle features
            g: (batch_size, global_dim) - current global embedding
            g_0: (batch_size, global_dim) - initial global embedding
            mask: (batch_size, n_particles) - binary mask where 1=valid particle, 0=padded
        """
        batch_size, n_particles, _ = x.shape

        mask_expanded = mask.unsqueeze(-1)  # (batch, n_particles, 1)
        edge_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (batch, n_particles, n_particles)
        
        # Minkowski edge features
        norms, dots, _ = minkowski_features(x, mask) 

        # Zero out features for masked edges
        norms = norms * edge_mask
        dots = dots * edge_mask

        # Compute messages m_ij = phi_e(psi(norms), psi(dots))
        # Stack norms and dots for phi_e input
        edge_features = torch.stack([norms, dots], dim=-1)  # (batch, n_particles, n_particles, 2)
    #     print(f"{edge_features.mean()=} {edge_features.std()=} {edge_features.max()=} {edge_features.min()=}")
        
        # Reshape for batch processing through phi_e
        edge_features_flat = edge_features.view(-1, 2)  # (batch * n_particles * n_particles, 2)
        messages_flat = self.phi_e(edge_features_flat)  # (batch * n_particles * n_particles, hidden_dim//2)
        messages = messages_flat.view(batch_size, n_particles, n_particles, -1)  # (batch, n_particles, n_particles, hidden_dim//2)
    #     print(f"{messages.mean()=} {messages.std()=} {messages.max()=} {messages.min()=}")
        # Apply edge mask to messages
        messages = messages * edge_mask.unsqueeze(-1)
        
        # Compute phi_m for global update
        phi_m_flat = self.phi_m(edge_features_flat)  # (batch * n_particles * n_particles, hidden_dim)
        phi_m_values = phi_m_flat.view(batch_size, n_particles, n_particles, -1)  # (batch, n_particles, n_particles, hidden_dim)
        phi_m_values = phi_m_values * edge_mask.unsqueeze(-1)
    #     print(f"{phi_m_values.shape=} {phi_m_values.mean()=} {phi_m_values.std()=} {phi_m_values.max()=} {phi_m_values.min()=}")
        
        # Global embedding update
        # Compute aggregated message features for global update
        # First term: (alpha/N) * sum(phi_m * messages)
        # weighted_messages = phi_m_values * messages  # Element-wise multiplication
        weighted_messages = phi_m_values
        # weighted_messages = phi_m_values
        sum_weighted_messages = torch.sum(weighted_messages, dim=(1, 2))  # (batch, hidden_dim//2)
    #     print(f"{sum_weighted_messages.shape=} {sum_weighted_messages.mean()=} {sum_weighted_messages.std()=} {sum_weighted_messages.max()=} {sum_weighted_messages.min()=}")

        n_valid = torch.sum(mask, dim=1)  # (batch, 1)
        n_valid = torch.clamp(n_valid, min=1)  # Avoid division by zero - edge case, should never trigger in practice since output is clamped
        # Global update input
        global_input = torch.cat([
            g_0,  # Initial global embedding
            g,  
            (self.alpha / n_valid).unsqueeze(-1) * sum_weighted_messages, 
        ], dim=-1)
        
        g_updated = self.phi_g(global_input)
        g_updated = self.global_norm(g_updated + g)  # Residual connection
        # Particle updates: x_i^{l+1} = x_i^l + gamma * sum_j phi_x(g^0, g^{l+1}, m_ij) * x_j^l
        # Vectorized implementation
        
        # Expand global embeddings to match message dimensions
        g_0_expanded = g_0.unsqueeze(1).unsqueeze(2).expand(batch_size, n_particles, n_particles, -1)
        g_updated_expanded = g_updated.unsqueeze(1).unsqueeze(2).expand(batch_size, n_particles, n_particles, -1)
    #     print(f"{g_0_expanded.mean()=} {g_0_expanded.std()=} {g_0_expanded.max()=} {g_0_expanded.min()=}")
        
        # Create phi_x input for all pairs (i,j)
        phi_x_input = torch.cat([
            g_0_expanded,           # (batch, n_particles, n_particles, global_dim)
            g_updated_expanded,     # (batch, n_particles, n_particles, global_dim)  
            messages               # (batch, n_particles, n_particles, hidden_dim//2)
        ], dim=-1)  # (batch, n_particles, n_particles, 2*global_dim + hidden_dim//2)

        
        # Reshape for batch processing through phi_x
        phi_x_input_flat = phi_x_input.view(-1, phi_x_input.shape[-1])
        phi_x_output_flat = self.phi_x(phi_x_input_flat)  # (batch*n_particles*n_particles, 1)
        phi_x_output = phi_x_output_flat.view(batch_size, n_particles, n_particles, 1)
    #     print(f"{phi_x_output.mean()=} {phi_x_output.std()=} {phi_x_output.max()=} {phi_x_output.min()=}")

        # Create mask to exclude self-connections (i != j)
        self_mask = torch.eye(n_particles, device=x.device).bool()
        self_mask = self_mask.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1)
        phi_x_output = phi_x_output * (~self_mask).float()
         
        # Apply edge mask (for padded particles)
        phi_x_output = phi_x_output * edge_mask.unsqueeze(-1)
    #     print(f"{phi_x_output.mean()=} {phi_x_output.std()=} {phi_x_output.max()=} {phi_x_output.min()=}")
        
        # Compute contributions: phi_x_output * x_j for each (i,j) pair
        x_expanded = x.unsqueeze(1).expand(-1, n_particles, -1, -1)  # (batch, n_particles, n_particles, particle_dim)
        contributions = phi_x_output * x_expanded  # (batch, n_particles, n_particles, particle_dim)
        
        # Sum over j for each i (excluding self-connections via the mask)
        particle_updates = torch.sum(contributions, dim=2)  # (batch, n_particles, particle_dim)
    #     print(f"{self.gamma=}")
    #     print(f"{particle_updates.mean()=} {particle_updates.std()=} {particle_updates.max()=} {particle_updates.min()=}")
        
        # Apply updates with learnable scaling
        x_updated = x + self.gamma * particle_updates
        
        # Apply mask to ensure padded particles remain zero
        x_updated = x_updated * mask_expanded
    #     print(f"{x_updated.mean()=} {x_updated.std()=} {x_updated.max()=} {x_updated.min()=}")

        
        # Apply layer norm with residual connection
        # x_updated = self.node_norm(x_updated)
        
        
        return x_updated, g_updated

class LEJetGeneratorFM(nn.Module):
    """Flow matching model for jet generation"""
    def __init__(self, 
                 n_particles=150,
                 particle_dim=4,  # 4-momentum
                 global_dim=64,
                 n_layers=6,
                 hidden_dim=128,
                 n_jet_types=5,  # Maximum number of jet types used during training of NF model
                 time_embed_dim=64,
                 gradient_clip_val=1.0):
        super().__init__()
        
        self.n_particles = n_particles
        self.particle_dim = particle_dim
        self.global_dim = global_dim
        self.n_layers = n_layers
        self.gradient_clip_val = gradient_clip_val
        
        # Conditioning dimension: one-hot jet type + eta + mass + pT + n_constituents
        self.conditioning_dim = n_jet_types + 4
        
        # Time embedding
        self.time_embedding = TimeEmbedding(embed_dim=time_embed_dim)
        
        # Initial global embedding from time + conditioning
        self.global_init = nn.Sequential(
            nn.Linear(time_embed_dim + self.conditioning_dim, global_dim),
            nn.BatchNorm1d(global_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        )
        
        # Lorentz equivariant layers
        self.le_layers = nn.ModuleList([
            LorentzEquivariantLayer(particle_dim, global_dim, hidden_dim, gamma_init=1.0/n_particles)
            for i in range(n_layers)
        ])
        
        # Final message passing layer
        self.final_layer = LorentzEquivariantLayer(particle_dim, global_dim, hidden_dim)
        
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
        
        # Time embedding
        t_embed = self.time_embedding(t)  # (batch_size, time_embed_dim)
        
        # Initial global embedding g^0
        global_input = torch.cat([t_embed, jet_conditions], dim=-1)
        g_0 = self.global_init(global_input)  # (batch_size, global_dim)
        
        # Initialize particle features and global state
        x_current = x.clone()
        g = g_0.clone()
        
        for layer in self.le_layers:
            x_current, g = layer(x_current, g, g_0, mask)
        
        # Final message passing round
        x_final, _ = self.final_layer(x_current, g, g_0, mask)
        
        # Compute velocity field: v_theta(t, x) = x_final - x
        velocity = x_final
        
        # Ensure masked particles have zero velocity
        velocity = velocity * mask.unsqueeze(-1)
        
        return velocity
    
    def clip_gradients(self):
        """Clip gradients for training stability"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
    
    def step(self, x_t, jet_conditions, mask, t_start, t_end, method='euler'):
        """
        Calculate the probability density at a particular time step
        """
        if method == 'euler':
            batch_size = x_t.shape[0]
        #     print("Rank:", torch.linalg.matrix_rank(x_t, atol=1e-5))
            update = self.forward(x=x_t, t=t_start.unsqueeze(0).repeat(batch_size), jet_conditions=jet_conditions, mask=mask)
            x_next = x_t + update * (t_end - t_start)

            return x_next

        else:
            raise NotImplementedError(f"Method {method} not implemented")


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    n_particles = 150
    particle_dim = 4
    global_dim = 64
    n_jet_types = 5

    x = torch.randn(batch_size, n_particles, particle_dim)
    jet_feats, _ = generate_jets(load_model(), 'cpu', n_jet_types, num_jets=batch_size)
    time = torch.rand(batch_size)

    mask = generate_masks(jet_feats[:, -1], n_particles, 'cpu')
    jet_conditions = jet_feats[:, :n_jet_types + 4]  # One-hot jet types + eta + mass + pT + n_constituents

    model = LEJetGeneratorFM(n_layers=2, n_particles=n_particles, particle_dim=particle_dim, global_dim=global_dim, n_jet_types=n_jet_types)
    velocity = model(x, time, jet_conditions, mask)