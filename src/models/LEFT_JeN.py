import torch
import torch.nn as nn
import numpy as np


from util.minkowski_utils import normsq4, dotsq4

def psi(p):
    ''' `\psi(p) = Sgn(p) \cdot \log(|p| + 1)` '''
    # Clamp inputs
    p = torch.clamp(p, -1e4, 1e4)
    return torch.sign(p) * torch.log1p(torch.abs(p))

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
        
        gen = torch.Generator()
        gen.manual_seed(seed)
        projection = (torch.randn(1, embed_dim//2, generator=gen) * scale).detach()
        self.register_buffer('projection', projection, persistent=True)

        self.fc1 = nn.Linear(embed_dim, 32)
        self.fc2 = nn.Linear(32, embed_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        
    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1) 

        projection = self.projection.to(t.device)  
        time_projection = t * projection * 2 * np.pi     # (batch_size, embed_dim//2)

        # Apply sin and cos, then concatenate
        sin_proj = torch.sin(time_projection)
        cos_proj = torch.cos(time_projection)
        time_embed = torch.cat([sin_proj, cos_proj], dim=-1)  # (batch_size, embed_dim)
        
        x = self.fc1(time_embed)  # (batch_size, 32)
        x = self.fc2(x)          # (batch_size, embed_dim)
        
        return x

class PhiMLP(nn.Module):
    """Multi-layer perceptron with configurable layers and activation."""
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.LeakyReLU(), dropout=0.05, output_activation=None):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on output layer
                if i > 0:
                    layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        if output_activation is not None:
            layers.append(output_activation)
        
        self.net = nn.Sequential(*layers)
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

class GlobalEmbedding(nn.Module):
    """Embeds jet type, number of constituents, and (optionally) jet pT."""
    def __init__(self, max_num_jet_types, max_constituents=150, embed_dim=64, include_pt=False):
        super().__init__()
        self.max_constituents = max_constituents
        self.include_pt = include_pt

        # Layout: [one_hot_type (max_num_jet_types), n_particles_norm (1)] + optional [pT (1)]
        input_dim = max_num_jet_types + 1 + (1 if include_pt else 0)

        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, jet_info):
        jet_info = jet_info.to(self.fc1.weight.device)
        if self.include_pt:
            # Layout: [...one_hot..., n_particles, pT]
            n_constituents = jet_info[:, -2]
            pt = jet_info[:, -1:]  # already normalized by FeaturewiseLinear — no transform
            n_norm = (n_constituents.float() / self.max_constituents).unsqueeze(-1)
            x = torch.cat([jet_info[:, :-2], n_norm, pt], dim=-1)
        else:
            # Layout: [...one_hot..., n_particles]
            n_constituents = jet_info[:, -1]
            n_norm = (n_constituents.float() / self.max_constituents).unsqueeze(-1)
            x = torch.cat([jet_info[:, :-1], n_norm], dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.output_norm(x)

class LorentzEquivariantLayer(nn.Module):
    def __init__(self, embed_dim=64, message_dim=128, hidden_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.message_dim = message_dim
        
        # Message computation: phi_e^l
        # Input: psi(||x_i - x_j||), psi(<x_i, x_j>), g^0, g^l, t_emb
        message_input_dim = 2 + embed_dim + embed_dim + embed_dim  # 2 + 3*embed_dim
        self.phi_e = PhiMLP(message_input_dim, [hidden_dim, hidden_dim], message_dim)
        
        # Message aggregation scalar: phi_m^l — Sigmoid gives soft gate in [0,1]
        self.phi_m = PhiMLP(message_dim, [hidden_dim, hidden_dim], 1, output_activation=nn.Sigmoid())

        # Global embedding update: phi_g^l
        global_input_dim = embed_dim + embed_dim + embed_dim + message_dim  # g^0, g^l, t_emb, aggregated_msg
        self.phi_g = PhiMLP(global_input_dim, [hidden_dim, hidden_dim], embed_dim)

        # Displacement scaling: phi_x^l — no output activation; gamma controls magnitude
        displacement_input_dim = embed_dim + embed_dim + embed_dim + message_dim  # g^0, g^l, t_emb, m_ij
        self.phi_x = PhiMLP(displacement_input_dim, [hidden_dim, hidden_dim], 1)

        # Learnable scaling parameters
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1))

        self.message_sf = nn.LayerNorm(message_dim)
        self.global_sf = nn.LayerNorm(embed_dim)
        
    def forward(self, x, g0, g_prev, t_emb, mask):
        batch_size, max_particles, _ = x.shape
        device = x.device

        x = torch.clamp(x, -1e8, 1e8)  # Clamp inputs to avoid numerical issues
        
        # Expand mask for broadcasting
        mask_i = mask.unsqueeze(-1).unsqueeze(-1)  # (batch_size, max_particles, 1, 1)
        mask_j = mask.unsqueeze(-2).unsqueeze(-1)  # (batch_size, 1, max_particles, 1)
        pair_mask = (mask_i * mask_j).squeeze(-1)  # (batch_size, max_particles, max_particles)
        
        # Compute pairwise distances and inner products
        x_i = x.unsqueeze(2)  # (batch_size, max_particles, 1, 4)
        x_j = x.unsqueeze(1)  # (batch_size, 1, max_particles, 4)
        
        # Minkowski norms and inner products
        diff = x_i - x_j  # (batch_size, max_particles, max_particles, 4)
        
        diff_flat = diff.view(-1, 4)
        norm_diff = normsq4(diff_flat).view(batch_size, max_particles, max_particles)

        xi_flat = x_i.expand(-1, -1, max_particles, -1).contiguous().view(-1, 4)
        xj_flat = x_j.expand(-1, max_particles, -1, -1).contiguous().view(-1, 4)        
        inner_prod = dotsq4(xi_flat, xj_flat).view(batch_size, max_particles, max_particles)
        
        psi_norm = psi(torch.clamp(norm_diff, -1e3, 1e3))
        psi_inner = psi(torch.clamp(inner_prod, -1e3, 1e3))
        
        # Apply mask to prevent computation on padded particles
        psi_norm = psi_norm * pair_mask
        psi_inner = psi_inner * pair_mask
        
        # Prepare inputs for message computation
        # Expand embeddings for pairwise computation
        g0_exp = g0.unsqueeze(1).unsqueeze(1).expand(-1, max_particles, max_particles, -1)
        g_prev_exp = g_prev.unsqueeze(1).unsqueeze(1).expand(-1, max_particles, max_particles, -1)
        t_emb_exp = t_emb.unsqueeze(1).unsqueeze(1).expand(-1, max_particles, max_particles, -1)
        message_input = torch.cat([
            psi_norm.unsqueeze(-1),  # (batch_size, max_particles, max_particles, 1)
            psi_inner.unsqueeze(-1), # (batch_size, max_particles, max_particles, 1)
            g0_exp,                  # (batch_size, max_particles, max_particles, embed_dim)
            g_prev_exp,              # (batch_size, max_particles, max_particles, embed_dim)
            t_emb_exp                # (batch_size, max_particles, max_particles, embed_dim)
        ], dim=-1)
        
        # Compute messages: phi_e^l
        messages = self.phi_e(message_input)  # (batch_size, max_particles, max_particles, message_dim)

        # Apply pair mask before LayerNorm so masked pairs don't influence learned statistics,
        # then re-apply after to eliminate any nonzero contribution from LayerNorm's bias term.
        pair_mask_exp = pair_mask.unsqueeze(-1).expand(-1, -1, -1, self.message_dim)
        messages = messages * pair_mask_exp
        messages = self.message_sf(messages)
        messages = messages * pair_mask_exp
        
        # Compute message scalings: phi_m^l  
        message_scalings = self.phi_m(messages).squeeze(-1)  # (batch_size, max_particles, max_particles)
        message_scalings = message_scalings * pair_mask
        
        # Cache message aggregation for global update
        scaled_messages = message_scalings.unsqueeze(-1) * messages  # (batch_size, max_particles, max_particles, message_dim)
        
        # Sum over all pairs, accounting for actual number of particles
        N_actual = mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
        N_actual_sq = N_actual * N_actual  # (batch_size, 1)
        
        global_message_sum = scaled_messages.sum(dim=[1, 2])  # (batch_size, message_dim)
        global_message_sfalized = (self.alpha / N_actual_sq) * global_message_sum
        
        # Update global embedding: phi_g^l
        global_input = torch.cat([g0, g_prev, t_emb, global_message_sfalized], dim=-1)
        g_new = self.phi_g(global_input)
        g_new = self.global_sf(g_new)
        
        # Prepare inputs for phi_x^l for all pairs.
        # g0_exp, g_prev_exp, t_emb_exp were computed above for message_input and are still valid.
        displacement_input = torch.cat([g0_exp, g_prev_exp, t_emb_exp, messages], dim=-1)
        # Shape: (batch_size, max_particles, max_particles, input_dim)
        
        # Apply phi_x^l to all pairs at once
        phi_x_out = self.phi_x(displacement_input).squeeze(-1)  # (batch_size, max_particles, max_particles)
        
        # Compute normalized displacements for all pairs
        # x_i - x_j for all pairs (we already computed this as diff)
        diff_norms = torch.clamp(normsq4(diff_flat), min=1e-8).view(batch_size, max_particles, max_particles)
        normalized_diff = diff / (1 + diff_norms.unsqueeze(-1))  # (batch_size, max_particles, max_particles, 4)
        
        scaled_displacements = phi_x_out.unsqueeze(-1) * normalized_diff  # (batch_size, max_particles, max_particles, 4)

        # Create diagonal mask to exclude i=j contributions
        diag_mask = 1.0 - torch.eye(max_particles, device=device).unsqueeze(0)  # (1, max_particles, max_particles)
        
        # Apply masks: exclude diagonal, apply particle mask, apply pair mask
        full_mask = diag_mask * pair_mask  # (batch_size, max_particles, max_particles)
        scaled_displacements = scaled_displacements * full_mask.unsqueeze(-1)
        
        # Sum over j for each i to get displacement for each particle
        displacement_sum = scaled_displacements.sum(dim=2)  # (batch_size, max_particles, 4)

        displacement_term = self.gamma * displacement_sum   # (batch_size, max_particles, 4)
        
        x_new = x + displacement_term
        x_new = x_new * mask.unsqueeze(-1)
        return x_new, g_new
    
class LEFTJeN(nn.Module):
    def __init__(self, max_num_jet_types, max_particles=150, embed_dim=64,
                 num_layers=6, message_dim=128, hidden_dim=64,
                 use_residual_update=True, include_pt=False):
        super().__init__()
        self.max_particles = max_particles
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_residual_update = use_residual_update

        if use_residual_update:
            max_alpha = 0.25 + (num_layers - 1) * 0.03
            if max_alpha > 1.0:
                raise ValueError(
                    f"Residual alpha schedule exceeds 1.0 at layer {num_layers - 1} "
                    f"(max alpha={max_alpha:.2f}). Reduce num_layers or adjust the schedule."
                )

        self.time_embedding = TimeEmbedding(embed_dim=embed_dim)
        self.global_embedding = GlobalEmbedding(max_num_jet_types, max_particles, embed_dim, include_pt=include_pt)
        self.layers = nn.ModuleList([
            LorentzEquivariantLayer(embed_dim, message_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Learned null conditioning token. Replaces the all-zeros null vector so the
        # unconditional representation is clearly separated from low-value conditioning.
        # n_particles_idx marks the slot that is preserved even in the null vector
        # (it is redundant with the mask, so zeroing it provides no extra guidance signal).
        cond_dim = max_num_jet_types + 1 + (1 if include_pt else 0)
        self.n_particles_idx = max_num_jet_types  # index of n_particles in cond vector
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))
        
    
    def make_null_cond(self, jet_conditions: torch.Tensor) -> torch.Tensor:
        """
        Build null conditioning for CFG.
        - Jet type and pT slots → learned null_cond parameter.
        - n_particles slot → real value from jet_conditions (already encoded in the mask,
          so it provides no type/pT guidance and should not be nulled out).
        """
        B = jet_conditions.shape[0]
        null_base = self.null_cond.unsqueeze(0).expand(B, -1)
        # Selector: 1 at n_particles position, 0 elsewhere
        keep = torch.zeros(jet_conditions.shape[-1], device=jet_conditions.device)
        keep[self.n_particles_idx] = 1.0
        return null_base * (1 - keep) + jet_conditions * keep

    def forward(self, x, t, jet_conditions, mask):
        """
        Forward pass of the flow matching model.

        Returns:
            v_theta: (batch_size, max_particles, 4) velocity field
        """
    
        t_emb = self.time_embedding(t) 
        g0 = self.global_embedding(jet_conditions)  
        # To preserve LE, pass scalar global PT value through each layer
        # pt = jet_conditions[:, 1]
        # print(f"{pt.shape=}")
        
        x0 = x.clone()
        g = g0.clone()
        
        for i, layer in enumerate(self.layers):
            x_new, g = layer(x, g0, g, t_emb, mask)
            if self.use_residual_update:
                alpha = 0.25 + i * 0.03
                x = ((1 - alpha) * x) + (alpha * x_new)
            else:
                x = x_new
            x = x * mask.unsqueeze(-1)

        velocity = x - x0
        return velocity * mask.unsqueeze(-1)
        
    def step(self, x_t, jet_conditions, mask, t_start, t_end, method='euler', use_cfg=False, guidance_weight=2.0):
        """
        Calculate the probability density at a particular time step
        """
        batch_size = x_t.shape[0]
        if use_cfg:
            null_jet_conditions = self.make_null_cond(jet_conditions)

        if method == 'euler':
            t_batch = t_start.unsqueeze(0).expand(batch_size)
            vel = self.forward(x=x_t, t=t_batch, jet_conditions=jet_conditions, mask=mask)
            if use_cfg:
                unconditional_vel = self.forward(
                    x=x_t,
                    t=t_batch,
                    jet_conditions=null_jet_conditions,
                    mask=mask
                )
                guided_vel = vel + guidance_weight * (vel - unconditional_vel)
                vel = guided_vel
            x_next = x_t + vel * (t_end - t_start)
            # Correct x_next to CoM frame
            # x_next = enforce_com_frame(x_next, mask)
            return x_next
        else:
            raise NotImplementedError(f"Method {method} not implemented")