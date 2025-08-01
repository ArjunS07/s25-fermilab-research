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
    x_i = x.unsqueeze(-2)
    x_j = x.unsqueeze(-3)
    x_diffs = x_i - x_j
    
    # Compute features first, then mask
    norms = normsq4(x_diffs)
    dots = dotsq4(x_i, x_j)
    
    # Apply masking after computation
    edge_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (B, N, N)
    norms = norms * edge_mask
    dots = dots * edge_mask
    
    norms, dots = psi(norms), psi(dots)
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
            nn.Linear(particle_dim, 1),
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

        edge_features = torch.stack([norms, dots], dim=-1)  # (batch, n_particles, n_particles, 2)
        
        # Reshape for batch processing through phi_e
        edge_features_flat = edge_features.view(-1, 2)  # (batch * n_particles * n_particles, 2)
        messages_flat = self.phi_e(edge_features_flat)  # (batch * n_particles * n_particles, hidden_dim//2)
        messages = messages_flat.view(batch_size, n_particles, n_particles, -1)  # (batch, n_particles, n_particles, hidden_dim//2)
        messages = messages * edge_mask.unsqueeze(-1)
        
        # Compute phi_m for global update
        phi_m_flat = self.phi_m(edge_features_flat)  # (batch * n_particles * n_particles, hidden_dim)
        phi_m_values = phi_m_flat.view(batch_size, n_particles, n_particles, -1)  # (batch, n_particles, n_particles, hidden_dim)
        phi_m_values = phi_m_values * edge_mask.unsqueeze(-1)
        
        # weighted_messages = phi_m_values * messages  # Element-wise multiplication
        weighted_messages = phi_m_values
        sum_weighted_messages = torch.sum(weighted_messages, dim=(1, 2))  # (batch, hidden_dim//2)

        n_valid = torch.sum(mask, dim=1)  # (batch, 1)
        n_valid = torch.clamp(n_valid, min=1)  # Avoid division by zero - edge case, should never trigger in practice since output of NF is clamped to 4


        global_input = torch.cat([
            g_0,
            g,  
            (self.alpha / (n_valid**2)).unsqueeze(-1) * sum_weighted_messages
        ], dim=-1)
        
        g_updated = self.phi_g(global_input)
        g_updated = self.global_norm(g_updated + g)  # Residual connection
        
        g_0_expanded = g_0.unsqueeze(1).unsqueeze(2).expand(batch_size, n_particles, n_particles, -1)
        g_updated_expanded = g_updated.unsqueeze(1).unsqueeze(2).expand(batch_size, n_particles, n_particles, -1)
        
        phi_x_input = torch.cat([
            g_0_expanded,           # (batch, n_particles, n_particles, global_dim)
            g_updated_expanded,     # (batch, n_particles, n_particles, global_dim)  
            messages               # (batch, n_particles, n_particles, hidden_dim//2)
        ], dim=-1)  # (batch, n_particles, n_particles, 2*global_dim + hidden_dim//2)

        
        phi_x_input_flat = phi_x_input.view(-1, phi_x_input.shape[-1])
        phi_x_output_flat = self.phi_x(phi_x_input_flat)  # (batch*n_particles*n_particles, 1)
        phi_x_output = phi_x_output_flat.view(batch_size, n_particles, n_particles, 1)

        self_mask = torch.eye(n_particles, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, N)
        phi_x_output = phi_x_output * (~self_mask.bool()).unsqueeze(-1) * edge_mask.unsqueeze(-1)         

        # mask x
        x_masked = x * mask_expanded  # (B, N, D)
        x_expanded = x_masked.unsqueeze(1).expand(-1, n_particles, -1, -1)  # (B, N, N, D)
        contributions = phi_x_output * x_expanded  # (batch, n_particles, n_particles, particle_dim)
        
        particle_updates = torch.sum(contributions, dim=2)  # (batch, n_particles, particle_dim)
        
        particle_updates = particle_updates * mask_expanded  # Mask the updates first
        x_updated = x + self.gamma * particle_updates

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
#     print("Velocity shape:", velocity.shape)  # Should be (batch_size, n_particles, particle_dim) 
    
# def train_model(
#         model: LEJetGeneratorFM,
#         x_train_loaded: torch.utils.data.DataLoader,
#         device: torch.device,
#         num_epochs: int,
#         batch_size: int,
#         warmup_pct=0.1,
#         lr=1e-3,
#         weight_decay=1e-2
# ):
#     losses = []
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

#     total_steps = num_epochs * len(x_train_loaded)
#     warmup_steps = int(warmup_pct * total_steps)

#     def lr_lambda(current_step):
#         if current_step < warmup_steps:
#             # Linear warm-up
#             return float(current_step) / float(max(1, warmup_steps))
#         else:
#             # Cosine annealing after warm-up
#             progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
#             return 0.5 * (1.0 + np.cos(np.pi * progress))
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

#     current_step = 0
#     for epoch in range(num_epochs):
#         epoch_loss = []

#         for i, data in enumerate(x_train_loaded):
#             optimizer.zero_grad()

#             jet_info = data[1].to(device)
#             x_1 = data[0].to(device)[:, :, :4]
#             x_0 = torch.randn_like(x_1, device=device)  # Sample random initial state

#             t = torch.rand(x_0.shape[0], device=device)
#             t_viewed = t.view(-1, 1, 1)
#             x_t = (1 - t_viewed) * x_0 + t_viewed * x_1  # Linear interpolation
#             dx_t = x_1 - x_0

#             pred = model.forward(x_t, t, jet_info)
#             loss = nn.MSELoss()(pred, dx_t)
#             loss.backward()
#             model.clip_gradients()
#             optimizer.step()

#             scheduler.step()
#             current_step += 1

#             epoch_loss.append(loss.item())

#             if i % (batch_size * 10) == 0:
#                 current_lr = optimizer.param_groups[0]['lr']
#             #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(x_train_loaded)}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
#             #     print(f"dx_t: mean={dx_t.abs().mean()}, std={dx_t.abs().std()}")
#                 if torch.cuda.is_available():
#                     allocated = torch.cuda.memory_allocated() / 1024**2       # Tensors currently live
#                     reserved = torch.cuda.memory_reserved() / 1024**2         # Memory reserved by PyTorch's caching allocator
#                     max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # Peak allocation during program
#                 #     print(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Peak: {max_allocated:.2f} MB")

#         losses.append(np.mean(epoch_loss))
#         if epoch % 10 == 0:
#             current_lr = optimizer.param_groups[0]['lr']
#         #     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses[-1]:.4f}, LR: {current_lr:.6f}")