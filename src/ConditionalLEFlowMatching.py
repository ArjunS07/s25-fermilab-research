import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import jet_attributes

from util.minkowski_utils import normsq4, dotsq4

def psi(p):
    ''' `\psi(p) = Sgn(p) \cdot \log(|p| + 1)` '''
    # Clamp inputs
    p = torch.clamp(p, -1e6, 1e6)
    return torch.sign(p) * torch.log1p(torch.abs(p))

def minkowski_features(x, device='cpu'):
    x_i = x.unsqueeze(-2).to(device)  # second-last dimension - N
    x_j = x.unsqueeze(-3).to(device)  # third-last dimension - B
    x_diffs = x_i - x_j  # (batch_size, n_particles, n_particles, 4)

    norms = normsq4(x_diffs).to(device)
    dots = dotsq4(x_i, x_j).to(device)
    norms, dots = psi(norms), psi(dots)
    return norms, dots, x_diffs

class PhiMLP(nn.Module):
    """
    Fully-connected linear layer in the network
    """
    def __init__(self, input_dim, hidden_dims, output_dim, use_batch_norm=True, dropout=0.0):
        super().__init__()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Don't add activation/norm after final layer
            if i < len(dims) - 2:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """
        Custom weight initialization for better gradient flow
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Maintain constant variance across layers to prevent vanishing gradients
                # https://proceedings.mlr.press/v9/glorot10a.html
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

class TimeEmbedding(nn.Module):
    """
    Random Fourier Features for time embedding following FPCD (Mikuni et al 2023)

    1. Apply a Gaussian fourier projection to the time input, with fixed weights, up to embed_dim/2.
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
        
        # Apply Gaussian Fourier projection
        # t * projection * 2 * pi
        projection = self.projection.to(t.device)  # Ensure projection is on the same device as t
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
    def __init__(self, particle_dim, global_dim, hidden_dim=128, c=1.0):
        super().__init__()
        
        self.c = c
        self.particle_dim = particle_dim
        self.global_dim = global_dim
        
        # Message computation: phi_m
        # Applied to psi of Minkowski features between particles
        self.phi_m = PhiMLP(2, [hidden_dim], hidden_dim, dropout=0.1)
        
        # Edge feature processing: phi_e and phi_ex  
        self.phi_e = PhiMLP(hidden_dim, [hidden_dim], hidden_dim//2, dropout=0.1)
        self.phi_ex = PhiMLP(hidden_dim, [hidden_dim], hidden_dim//2, dropout=0.1)
        
        # Global update: phi_eg
        # Input: [g_l, sum_pooled_edges, avg_pooled_edges]
        global_input_dim = global_dim + 2 * (hidden_dim//2)
        self.phi_eg = PhiMLP(global_input_dim, [hidden_dim, hidden_dim], global_dim, dropout=0.1)
        
        # Node update: phi_x
        # Input: [g_0, g_{l+1}, sum_pooled_edges, avg_pooled_edges]
        node_input_dim = 2 * global_dim + 2 * (hidden_dim//2)
        self.phi_x = PhiMLP(node_input_dim, [hidden_dim, hidden_dim], particle_dim, dropout=0.1)
        
        # Normalization for the entire layer
        self.node_norm = nn.LayerNorm(particle_dim)
        self.global_norm = nn.LayerNorm(global_dim)
        
    def forward(self, x, g, g_0, device='cpu'):
        """
        Args:
            x: (batch_size, n_particles, particle_dim) - particle features
            g: (batch_size, global_dim) - current global embedding
            g_0: (batch_size, global_dim) - initial global embedding
        """
        batch_size, n_particles, _ = x.shape
        
        # Compute Minkowski edge features
        norms, dots, _ = minkowski_features(x[..., :4], device) 
        
        # Compute messages m_ij = phi_m(psi(||x_i - x_j||), psi(⟨x_i, x_j⟩))
        edge_features = torch.stack([norms, dots], dim=-1)  # (batch, n_particles, n_particles, 2)
        
        # Reshape for MLP processing
        edge_shape = edge_features.shape
        edge_features_flat = edge_features.view(-1, 2)
        messages_flat = self.phi_m(edge_features_flat)
        
        # Process messages for global and node updates
        edge_features_global = self.phi_e(messages_flat).view(*edge_shape[:-1], -1)
        edge_features_node = self.phi_ex(messages_flat).view(*edge_shape[:-1], -1)
        
        # Aggregate messages (sum and average pooling)
        sum_pooled_global = edge_features_global.sum(dim=(1, 2))  # (batch, hidden_dim//2)
        avg_pooled_global = edge_features_global.mean(dim=(1, 2))  # (batch, hidden_dim//2)
        
        sum_pooled_node = edge_features_node.sum(dim=2)  # (batch, n_particles, hidden_dim//2)
        avg_pooled_node = edge_features_node.mean(dim=2)  # (batch, n_particles, hidden_dim//2)
        
        # Update global embedding
        global_input = torch.cat([g, sum_pooled_global, avg_pooled_global], dim=-1)
        g_new = self.phi_eg(global_input)
        g_new = self.global_norm(g_new + g)  # Residual connection with normalization
        
        # Update node embeddings
        # Broadcast global features to match node dimensions
        g_0_broadcast = g_0.unsqueeze(1).expand(-1, n_particles, -1)
        g_new_broadcast = g_new.unsqueeze(1).expand(-1, n_particles, -1)
        
        node_input = torch.cat([
            g_0_broadcast, 
            g_new_broadcast, 
            sum_pooled_node, 
            avg_pooled_node
        ], dim=-1)
        
        # Reshape for MLP
        node_input_flat = node_input.view(-1, node_input.shape[-1])
        x_update_flat = self.phi_x(node_input_flat)
        x_update = x_update_flat.view(batch_size, n_particles, -1)
        
        # Residual connection with normalization
        x_new = self.node_norm(x + x_update)
        
        return x_new, g_new

class JetFMGenerator(nn.Module):
    """Flow matching model for jet generation"""
    def __init__(self, 
                 n_particles=30,
                 particle_dim=4,  # 4-momentum
                 global_dim=64,
                 n_layers=6,
                 hidden_dim=128,
                 n_jet_types=5,
                 c=1.0):
        super().__init__()
        
        self.n_particles = n_particles
        self.particle_dim = particle_dim
        self.global_dim = global_dim
        self.n_layers = n_layers
        
        # Time embedding
        self.time_embedding = TimeEmbedding(global_dim)
        
        # Initial global embedding
        # Input: [time_embed, jet_type_onehot, eta, jet_p_t, jet_mass, jet_n_constituents]
        conditioning_dim = global_dim + n_jet_types + 4
        self.initial_global = PhiMLP(conditioning_dim, [global_dim, global_dim], global_dim)
        
        # Lorentz-equivariant layers
        self.le_layers = nn.ModuleList([
            LorentzEquivariantLayer(particle_dim, global_dim, hidden_dim, c)
            for _ in range(n_layers)
        ])
        
        # Final message passing layer
        final_input_dim = 2 * global_dim + 2 * (hidden_dim//2)
        self.final_phi_m = PhiMLP(2, [hidden_dim], hidden_dim, dropout=0.1)
        self.final_phi_ex = PhiMLP(hidden_dim, [hidden_dim], hidden_dim//2, dropout=0.1)
        self.final_phi_x = PhiMLP(final_input_dim, [hidden_dim, hidden_dim], particle_dim, dropout=0.1)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(particle_dim)

        # Gradient clipping for stability
        self.gradient_clip_val = 1.0
        
    def forward(self, x, t, jet_conditions, device='cpu'):
        """
        Args:
            x: (batch_size, n_particles, 4) - initial particle 4-momenta
            t: (batch_size,) - time values
            jet_conditions: (batch_size, n_jet_types + 4) - jet type one-hot and other features
        """
        batch_size = x.shape[0]
        
        # Time embedding
        t_embed = self.time_embedding(t)
        conditioning = torch.cat([t_embed, jet_conditions], dim=-1)

        # Initial global embedding
        g_0 = self.initial_global(conditioning)

        # Initialize particle features (pad x if needed)
        if x.shape[-1] < self.particle_dim:
            padding = torch.zeros(batch_size, self.n_particles, 
                                self.particle_dim - x.shape[-1], device=device)
            x = torch.cat([x, padding], dim=-1)
        
        # Apply Lorentz-equivariant layers
        g = g_0
        for layer in self.le_layers:
            x, g = layer(x, g, g_0, device)
        
        # Final message passing
        norms, dots, _ = minkowski_features(x[..., :4], device)
        edge_features = torch.stack([norms, dots], dim=-1)
        
        edge_shape = edge_features.shape
        edge_features_flat = edge_features.view(-1, 2)
        messages_flat = self.final_phi_m(edge_features_flat)
        
        edge_features_node = self.final_phi_ex(messages_flat).view(*edge_shape[:-1], -1)
        sum_pooled_node = edge_features_node.sum(dim=2)
        avg_pooled_node = edge_features_node.mean(dim=2)
        
        # Final node update
        g_0_broadcast = g_0.unsqueeze(1).expand(-1, self.n_particles, -1)
        g_broadcast = g.unsqueeze(1).expand(-1, self.n_particles, -1)

        print(f"g_0_broadcast shape: {g_0_broadcast.shape}, g_broadcast shape: {g_broadcast.shape}")

        final_input = torch.cat([
            g_0_broadcast,
            g_broadcast,
            sum_pooled_node,
            avg_pooled_node
        ], dim=-1)

        final_input_flat = final_input.view(-1, final_input.shape[-1])
        x_final_flat = self.final_phi_x(final_input_flat)
        x_final = x_final_flat.view(batch_size, self.n_particles, -1)
        
        # Final residual connection
        x_out = self.final_norm(x + x_final)

        return x_out
    
    def clip_gradients(self):
        """Clip gradients for training stability"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
    
    def step(self, x_t, jet_conditions, t_start, t_end, method='euler'):
        """
        Calculate the probability density at a particular time step
        """
        if method == 'euler':
            batch_size = x_t.shape[0]
            update = self.forward(x=x_t, t=t_start.unsqueeze(0).repeat(batch_size), jet_conditions=jet_conditions)
            x_next = x_t + update * (t_end - t_start)
            return x_next
        else:
            raise NotImplementedError(f"Method {method} not implemented")

# Example usage and training utilities
# Training step example
def training_step(model: JetFMGenerator, optimizer, jet_conditions, target, device='cpu'):
    """Example training step with gradient clipping"""
    optimizer.zero_grad()

    x_0 = torch.randn_like(target, device=device)  # Initial state
    x_1 = target.to(device)[:, :, :4]
    
    t = torch.rand(x_0.shape[0], device=device)
    t_viewed = t.view(-1, 1, 1)  # Reshape for broadcasting
    x_t = (1 - t_viewed) * x_0 + t_viewed * x_1  # Linear interpolation
    dx_t = x_1 - x_0

    loss = nn.MSELoss()(model.forward(x_t, t, jet_conditions, device=device), dx_t)
    loss.backward()
    model.clip_gradients()
    optimizer.step()

    return loss.item()

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 50

    jet_attr_generator = jet_attributes.load_model().to(device)
    with torch.no_grad():
        generated_jet_attrs, _ = jet_attributes.generate_jets(
            jet_attr_generator, device, n_jet_types=3, num_jets=batch_size
        )

    model: JetFMGenerator = JetFMGenerator().to(device)
    target = torch.randn(batch_size, 30, 4, device=device) + 0.2 * torch.randn(batch_size, 30, 4, device=device)  # Placeholder target

    n_epochs = 25
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(n_epochs):
        loss = training_step(model, optimizer, generated_jet_attrs, target, device)
        print(f"Epoch {epoch + 1}/{n_epochs}, Training loss: {loss}")

    integration_steps = 8
    
    n_samples = 25
    times = torch.linspace(0, 1, integration_steps + 1).to(device)

    x = torch.randn(n_samples, 30, 4, device=device)  # Sample initial state
    generated_jet_attrs, _ = jet_attributes.generate_jets(
        jet_attr_generator, device, n_jet_types=3, num_jets=n_samples
    )
    x0 = x.clone()
    with torch.no_grad():
        model.eval()
        model.to(device)
        x = x.to(device)
        for i in range(integration_steps):
            x = model.step(x, generated_jet_attrs, times[i], times[i + 1])