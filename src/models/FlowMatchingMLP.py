import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=128, time_embed_dim=32, num_layers=12):
        super().__init__()
        self.in_features = in_features

        # Embed time using simple MLP (you could also use Sinusoidal embeddings)
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input: particle features + time embedding
        input_dim = in_features + time_embed_dim
        layers = []
        for i in range(num_layers):
            d_in = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, in_features))  # Output: velocity
        self.net = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(in_features)

    def forward(self, x, t):
        """
        x: [batch_size, num_particles, in_features]
        t: [batch_size] or [batch_size, 1] or scalar
        Returns: velocity of shape [batch_size, num_particles, in_features]
        """
        B, N, F = x.shape
        x = self.norm(x)

        if isinstance(t, float) or len(t.shape) == 0:
            t = torch.tensor([t], device=x.device).repeat(B)
        t = t.view(B, 1, 1).expand(B, N, 1)  # shape: [B, N, 1]
        t_embed = self.time_embed(t)        # shape: [B, N, time_embed_dim]

        x_input = torch.cat([x, t_embed], dim=-1)  # shape: [B, N, F + time_embed_dim]
        x_flat = x_input.view(-1, x_input.shape[-1])
        v_flat = self.net(x_flat)
        v = v_flat.view(B, N, F)
        return v
    
    def step(self, x, t0, t1):
        """
        x: [batch_size, num_particles, in_features]
        t0: scalar or [batch_size] or [batch_size, 1]
        t1: scalar or [batch_size] or [batch_size, 1]
        Returns: updated x after one step of flow matching
        """
        v = self.forward(x, t0)
        dt = t1 - t0
        return x + v * dt