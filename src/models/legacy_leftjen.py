import torch
import torch.nn as nn
import numpy as np


from util.minkowski_utils import normsq4, dotsq4

def psi(p):
    r''' `\psi(p) = Sgn(p) \cdot \log(|p| + 1)` '''
    # Clamp inputs
    p = torch.clamp(p, -1e4, 1e4)
    return torch.sign(p) * torch.log1p(torch.abs(p))


def masked_neighbor_softmax(logits: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    """Softmax of per-edge ``logits`` over sending neighbors j, for each receiving node i.

    Self-loops (the diagonal) and padded pairs are removed from the softmax support, so the
    weights over the real j != i neighbours sum to 1. A node with no valid neighbour (e.g. a
    single-particle jet) softmaxes over an all-``-inf`` row → the result is set to all zeros
    so it contributes nothing and stays finite (no NaN).

    logits, pair_mask : (batch, max_particles, max_particles)
    Returns             (batch, max_particles, max_particles) attention weights.
    """
    max_particles = logits.shape[1]
    eye = torch.eye(max_particles, device=logits.device, dtype=logits.dtype).unsqueeze(0)
    support = pair_mask * (1.0 - eye)                        # real pairs with j != i
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(support == 0, neg_inf)
    attn = torch.softmax(masked_logits, dim=2)
    row_has_valid = support.sum(dim=2, keepdim=True) > 0
    return torch.where(row_has_valid, attn, torch.zeros_like(attn))

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
    
    def __init__(self, embed_dim=128, scale=16.0, seed=42):
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
    def __init__(self, max_num_jet_types, max_constituents=150, embed_dim=128, include_pt=False):
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
    def __init__(self, embed_dim=128, message_dim=128, hidden_dim=128,
                 use_node_scalars=False, node_scalar_dim=128, use_adaln=False,
                 use_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.message_dim = message_dim
        self.use_node_scalars = use_node_scalars
        self.node_scalar_dim = node_scalar_dim
        self.use_adaln = use_adaln
        self.use_attention = use_attention

        # Message computation: phi_e^l.
        # Base per-edge inputs: psi(||x_i - x_j||), psi(<x_i, x_j>) [+ h_i, h_j].
        # Without adaLN, the global/time conditioning (g^0, g^l, t_emb) is *concatenated*
        # into every edge (the (B,N,N,~3*embed) memory hog). With adaLN it modulates the
        # messages via FiLM instead, so it is dropped from the per-edge input.
        message_input_dim = 2
        if use_node_scalars:
            message_input_dim += 2 * node_scalar_dim  # h_i, h_j
        if not use_adaln:
            message_input_dim += 3 * embed_dim        # g^0, g^l, t_emb
        self.phi_e = PhiMLP(message_input_dim, [hidden_dim, hidden_dim], message_dim)

        # FiLM/adaLN modulator: invariant conditioning -> (scale, shift) for the messages.
        if use_adaln:
            self.adaln_mod = PhiMLP(3 * embed_dim, [hidden_dim], 2 * message_dim)
            # adaLN-zero init: start as identity (scale=shift=0) for stable training.
            last_linear = [m for m in self.adaln_mod.net if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

        # Per-node scalar update: phi_h^l — h_i <- h_i + phi_h([h_i, aggregated_messages_i])
        if use_node_scalars:
            self.phi_h = PhiMLP(node_scalar_dim + message_dim, [hidden_dim, hidden_dim], node_scalar_dim)

        # Message aggregation scalar: phi_m^l — Sigmoid gives soft gate in [0,1]
        self.phi_m = PhiMLP(message_dim, [hidden_dim, hidden_dim], 1, output_activation=nn.Sigmoid())

        # Attention aggregation (Phase 3.1): a Lorentz-invariant logit per edge (function of the
        # invariant message content only) that is softmax-normalized over the sending neighbors j.
        # Replaces the sigmoid soft-gate when enabled; equivariance is preserved because the
        # logits are invariant. Gated by use_attention (off => byte-for-byte the sigmoid path).
        if use_attention:
            self.phi_attn = PhiMLP(message_dim, [hidden_dim, hidden_dim], 1)

        # Global embedding update: phi_g^l (per-jet, cheap; keeps explicit conditioning)
        global_input_dim = embed_dim + embed_dim + embed_dim + message_dim  # g^0, g^l, t_emb, aggregated_msg
        self.phi_g = PhiMLP(global_input_dim, [hidden_dim, hidden_dim], embed_dim)

        # Displacement scaling: phi_x^l — no output activation; gamma controls magnitude.
        # With adaLN the messages already carry the conditioning, so the globals are dropped.
        displacement_input_dim = message_dim if use_adaln else (3 * embed_dim + message_dim)
        self.phi_x = PhiMLP(displacement_input_dim, [hidden_dim, hidden_dim], 1)

        # Learnable scaling parameters
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1))

        self.message_sf = nn.LayerNorm(message_dim)
        self.global_sf = nn.LayerNorm(embed_dim)
        
    def forward(self, x, g0, g_prev, t_emb, mask, h=None):
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
        message_parts = [
            psi_norm.unsqueeze(-1),  # (batch_size, max_particles, max_particles, 1)
            psi_inner.unsqueeze(-1), # (batch_size, max_particles, max_particles, 1)
        ]
        if not self.use_adaln:
            # Concatenate the conditioning into every edge (the memory-heavy path).
            message_parts.extend([g0_exp, g_prev_exp, t_emb_exp])
        if self.use_node_scalars:
            # Per-node scalars h_i, h_j on every edge (i receives, j sends).
            h_i_exp = h.unsqueeze(2).expand(-1, -1, max_particles, -1)  # (B, P, P, node_dim)
            h_j_exp = h.unsqueeze(1).expand(-1, max_particles, -1, -1)  # (B, P, P, node_dim)
            message_parts.extend([h_i_exp, h_j_exp])
        message_input = torch.cat(message_parts, dim=-1)

        # Compute messages: phi_e^l
        messages = self.phi_e(message_input)  # (batch_size, max_particles, max_particles, message_dim)

        # Apply pair mask before LayerNorm so masked pairs don't influence learned statistics,
        # then re-apply after to eliminate any nonzero contribution from LayerNorm's bias term.
        pair_mask_exp = pair_mask.unsqueeze(-1).expand(-1, -1, -1, self.message_dim)
        messages = messages * pair_mask_exp
        messages = self.message_sf(messages)
        messages = messages * pair_mask_exp

        if self.use_adaln:
            # FiLM/adaLN: modulate the messages by invariant (scale, shift) derived from the
            # global + time conditioning. Invariant coefficients => equivariance preserved.
            cond = torch.cat([g0, g_prev, t_emb], dim=-1)  # (batch_size, 3*embed_dim)
            scale, shift = self.adaln_mod(cond).chunk(2, dim=-1)  # each (batch_size, message_dim)
            scale = scale.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, message_dim)
            shift = shift.unsqueeze(1).unsqueeze(1)
            messages = messages * (1 + scale) + shift
            messages = messages * pair_mask_exp

        # Compute message scalings.
        if self.use_attention:
            # Invariant-logit softmax attention: normalize over the sending neighbors j per node i.
            # Logits are functions of the invariant messages only → equivariance is preserved.
            logits = self.phi_attn(messages).squeeze(-1)  # (batch_size, max_particles, max_particles)
            message_scalings = masked_neighbor_softmax(logits, pair_mask)
        else:
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
        
        # Prepare inputs for phi_x^l for all pairs. Under adaLN the messages already carry the
        # conditioning (via FiLM), so the globals are dropped from the displacement head too.
        if self.use_adaln:
            displacement_input = messages
        else:
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

        # Per-node scalar update: aggregate incoming messages per node and update h_i.
        if self.use_node_scalars:
            node_message_sum = scaled_messages.sum(dim=2)  # (batch_size, max_particles, message_dim)
            node_agg = node_message_sum / N_actual.unsqueeze(-1).clamp(min=1.0)
            h_new = h + self.phi_h(torch.cat([h, node_agg], dim=-1))
            h_new = h_new * mask.unsqueeze(-1)
        else:
            h_new = None

        return x_new, g_new, h_new
    
class LegacyLEFTJeN(nn.Module):
    def __init__(self, max_num_jet_types, max_particles=150, embed_dim=128,
                 num_layers=6, message_dim=128, hidden_dim=128,
                 use_residual_update=True, include_pt=False,
                 use_reference_vectors=False, use_node_scalars=False,
                 node_scalar_seed="physics",
                 node_scalar_dim=None, use_adaln=False, use_attention=False,
                 backbone="legacy", include_mass_condition=False,
                 **unused_compatibility_options):
        super().__init__()
        if backbone != "legacy":
            raise ValueError("LegacyLEFTJeN only implements the historical backbone")
        self.max_particles = max_particles
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_residual_update = use_residual_update
        # Symmetry-breaking / expressiveness options (Phase 1). Both default off, so with
        # both disabled the model is byte-for-byte the original (run A of the ablation grid).
        self.use_reference_vectors = use_reference_vectors
        self.use_node_scalars = use_node_scalars
        if node_scalar_seed not in ("physics", "zero", "conditions"):
            raise ValueError(f"node_scalar_seed must be one of physics/zero/conditions, got {node_scalar_seed!r}")
        self.node_scalar_seed = node_scalar_seed
        self.use_adaln = use_adaln
        self.use_attention = use_attention
        self.backbone = "legacy"
        self.include_mass_condition = include_mass_condition
        self.node_scalar_dim = node_scalar_dim if node_scalar_dim is not None else embed_dim
        # Number of reference virtual particles when enabled: e_t=(1,0,0,0) and jet 4-momentum.
        self.num_references = 2

        if use_residual_update:
            max_alpha = 0.25 + (num_layers - 1) * 0.03
            if max_alpha > 1.0:
                raise ValueError(
                    f"Residual alpha schedule exceeds 1.0 at layer {num_layers - 1} "
                    f"(max alpha={max_alpha:.2f}). Reduce num_layers or adjust the schedule."
                )

        # Learned null conditioning token. Replaces the all-zeros null vector so the
        # unconditional representation is clearly separated from low-value conditioning.
        # n_particles_idx marks the slot that is preserved even in the null vector
        # (it is redundant with the mask, so zeroing it provides no extra guidance signal).
        cond_dim = max_num_jet_types + 1 + (1 if include_pt else 0) + (1 if include_mass_condition else 0)
        self.cond_dim = cond_dim
        self.n_particles_idx = max_num_jet_types  # index of n_particles in cond vector
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))

        self.time_embedding = TimeEmbedding(embed_dim=embed_dim)
        self.global_embedding = GlobalEmbedding(
            max_num_jet_types, max_particles, embed_dim, include_pt=include_pt)
        self.layers = nn.ModuleList([
            LorentzEquivariantLayer(embed_dim, message_dim, hidden_dim,
                                    use_node_scalars=use_node_scalars,
                                    node_scalar_dim=self.node_scalar_dim,
                                    use_adaln=use_adaln,
                                    use_attention=use_attention)
            for _ in range(num_layers)
        ])

        # Per-node scalar seed input dim. "physics" and "zero" produce 3-dim invariants (m², E,
        # axis); "conditions" broadcasts jet_conditions (cond_dim) per particle so the h_i
        # channel starts from a purely learned pathway with no Minkowski off-shell hook.
        if use_node_scalars:
            seed_input_dim = 3 if node_scalar_seed in ("physics", "zero") else cond_dim
            self.node_seed = PhiMLP(seed_input_dim, [hidden_dim], self.node_scalar_dim)
        
    
    def make_null_cond(self, jet_conditions: torch.Tensor) -> torch.Tensor:
        """
        Build null conditioning for CFG.
        - Jet type and pT slots -> learned null_cond parameter.
        - n_particles slot -> real value from jet_conditions (already encoded in the mask,
          so it provides no type/pT guidance and should not be nulled out).
        """
        B = jet_conditions.shape[0]
        null_base = self.null_cond.unsqueeze(0).expand(B, -1)
        # Selector: 1 at n_particles position, 0 elsewhere
        keep = torch.zeros(jet_conditions.shape[-1], device=jet_conditions.device)
        keep[self.n_particles_idx] = 1.0
        return null_base * (1 - keep) + jet_conditions * keep

    def _node_seed_features(self, x_aug, refs, jet_conditions=None):
        """Per-node seed features for the scalar channel h_i.

        Three modes controlled by self.node_scalar_seed:
          "physics"    — (B, N, 3) Minkowski invariants
                         [psi(m_i^2), psi(E_i)=psi(<x,e_t>), psi(<x,x_jet>)].
                         With refs=None the E and axis columns are zero (run F), so h is
                         seeded from m_i^2 alone and the network stays rotation-equivariant.
          "zero"       — (B, N, 3) zeros; h_i is a purely learned per-node hidden state
                         with no physics prior (blank channel).
          "conditions" — (B, N, cond_dim) jet_conditions broadcast per particle; h_i starts
                         from Lorentz-invariant class/multiplicity/pT signal only.
        """
        if self.node_scalar_seed == "physics":
            feat_m = psi(normsq4(x_aug))
            if refs is not None:
                e_t = refs[:, 0:1, :]
                x_jet = refs[:, 1:2, :]
                feat_E = psi(dotsq4(x_aug, e_t))
                feat_axis = psi(dotsq4(x_aug, x_jet))
            else:
                feat_E = torch.zeros_like(feat_m)
                feat_axis = torch.zeros_like(feat_m)
            return torch.stack([feat_m, feat_E, feat_axis], dim=-1)
        if self.node_scalar_seed == "zero":
            B, N, _ = x_aug.shape
            return torch.zeros(B, N, 3, device=x_aug.device, dtype=x_aug.dtype)
        # "conditions": (B, cond_dim) -> (B, N, cond_dim)
        if jet_conditions is None:
            raise ValueError("node_scalar_seed='conditions' requires jet_conditions to be passed in.")
        return jet_conditions.unsqueeze(1).expand(-1, x_aug.shape[1], -1)

    def forward(self, x, t, jet_conditions, mask, ref_vectors=None):
        """
        Forward pass of the flow matching model.

        Args:
            x: (batch_size, max_particles, 4) particle 4-vectors (E, px, py, pz).
            t: (batch_size,) time in [0, 1).
            jet_conditions: (batch_size, cond_dim) conditioning vector.
            mask: (batch_size, max_particles) float mask, 1=real, 0=padding.
            ref_vectors: (batch_size, R, 4) reference 4-vectors (R=2: e_t, jet 4-momentum).
                Required iff use_reference_vectors; ignored otherwise.

        Returns:
            v_theta: (batch_size, max_particles, 4) velocity field
        """
        if self.use_reference_vectors and ref_vectors is None:
            raise ValueError("use_reference_vectors=True requires ref_vectors of shape (B, R, 4).")

        t_emb = self.time_embedding(t)
        g0 = self.global_embedding(jet_conditions)

        batch_size, num_particles, _ = x.shape

        # References enter as unmasked virtual particles; build the augmented mask.
        if self.use_reference_vectors:
            refs = ref_vectors
            num_refs = refs.shape[1]
            ref_mask = torch.ones(batch_size, num_refs, device=mask.device, dtype=mask.dtype)
            mask_aug = torch.cat([mask, ref_mask], dim=1)
        else:
            refs = None
            num_refs = 0
            mask_aug = mask

        # Seed per-node scalars from the initial (augmented) state.
        if self.use_node_scalars:
            x_aug0 = torch.cat([x, refs], dim=1) if num_refs > 0 else x
            seed_feats = self._node_seed_features(x_aug0, refs, jet_conditions)
            h = self.node_seed(seed_feats) * mask_aug.unsqueeze(-1)  # (B, P+R, node_scalar_dim)
        else:
            h = None

        x0 = x.clone()
        x_particles = x
        g = g0.clone()

        for i, layer in enumerate(self.layers):
            # References are fixed across layers: re-supply them unchanged each iteration and
            # discard their displacement outputs — only real-particle rows carry state forward.
            x_aug = torch.cat([x_particles, refs], dim=1) if num_refs > 0 else x_particles
            x_new_aug, g, h = layer(x_aug, g0, g, t_emb, mask_aug, h)
            x_new = x_new_aug[:, :num_particles]
            if self.use_residual_update:
                alpha = 0.25 + i * 0.03
                x_particles = ((1 - alpha) * x_particles) + (alpha * x_new)
            else:
                x_particles = x_new
            x_particles = x_particles * mask.unsqueeze(-1)

        velocity = x_particles - x0
        return velocity * mask.unsqueeze(-1)
        
    def step_hyperbolic(
        self,
        y_t: torch.Tensor,
        jet_conditions: torch.Tensor,
        mask: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        c: float = 1.0,
        use_cfg: bool = False,
        guidance_weight: float = 2.0,
        ref_vectors: torch.Tensor = None,
        hyperbolic_model: str = "poincare",
        regulator_mass: float = 0.5,
        max_step_rapidity: float = None,
        max_substeps: int = 64,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """
        Riemannian Euler step in the Poincaré ball (default) or on the mass shell.

        y_t  : (batch, particles, 4)  current state IN THE BALL (poincare) or ON H_m (mass_shell)
        Returns next state in the same space.
        """
        if hyperbolic_model == "mass_shell":
            return self._step_mass_shell(
                y_t, jet_conditions, mask, t_start, t_end,
                regulator_mass, use_cfg, guidance_weight, ref_vectors,
                max_step_rapidity, max_substeps,
                return_diagnostics,
            )

        from util.hyperbolic import from_poincare_ball, exp_map, pushforward, clamp_to_ball

        batch_size = y_t.shape[0]
        t_batch = t_start.unsqueeze(0).expand(batch_size)

        # Map ball → Cartesian for model input
        x_t = from_poincare_ball(y_t, c=c)

        # Model predicts velocity in Cartesian space
        vel_cartesian = self.forward(x=x_t, t=t_batch, jet_conditions=jet_conditions, mask=mask, ref_vectors=ref_vectors)

        # CFG in Cartesian space, before pushforward
        if use_cfg:
            null_cond = self.make_null_cond(jet_conditions)
            vel_uncond = self.forward(x=x_t, t=t_batch, jet_conditions=null_cond, mask=mask, ref_vectors=ref_vectors)
            vel_cartesian = vel_cartesian + guidance_weight * (vel_cartesian - vel_uncond)

        # Push Cartesian velocity to ball tangent space at y_t
        vel_ball = pushforward(x_t, vel_cartesian, c=c)
        vel_ball = vel_ball * mask.unsqueeze(-1)

        # Geodesic step: y_{t+dt} = exp_{y_t}(v_ball * dt)
        dt = t_end - t_start
        y_next = exp_map(y_t, vel_ball * dt, c=c)
        y_next = clamp_to_ball(y_next, c=c)

        return y_next * mask.unsqueeze(-1)

    def _step_mass_shell(
        self,
        y_t: torch.Tensor,
        jet_conditions: torch.Tensor,
        mask: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        m: float,
        use_cfg: bool,
        guidance_weight: float,
        ref_vectors: torch.Tensor,
        max_step_rapidity: float = None,
        max_substeps: int = 64,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """Riemannian Euler step on the mass shell H_m. The state y_t is already on the shell
        (a Cartesian on-shell 4-vector), so it is fed to the model directly; the predicted
        Cartesian velocity is projected onto the tangent space and the geodesic exp map takes
        the step. Masked/padded rows stay put (zero tangent velocity) and remain on the shell."""
        from util.mass_shell import (MassShellIntegrationError, exp_map,
                                     project_to_shell, tangent_norm)

        if max_step_rapidity is None:
            vel_tan = self._mass_shell_velocity(
                y_t, jet_conditions, mask, t_start, m, use_cfg, guidance_weight, ref_vectors)
            result = project_to_shell(exp_map(y_t, vel_tan * (t_end - t_start), m), m)
            if return_diagnostics:
                real_norms = tangent_norm(vel_tan).squeeze(-1)[mask > 0]
                max_norm = float(real_norms.max().detach().cpu()) if real_norms.numel() else 0.0
                return result, {
                    "substeps": 1,
                    "max_tangent_norm": max_norm,
                    "max_step_rapidity": max_norm * abs(float(t_end - t_start)) / float(m),
                }
            return result

        if max_step_rapidity <= 0 or max_substeps < 1:
            raise ValueError("adaptive mass-shell limits must be positive")

        state = y_t
        current_t = float(t_start.detach().cpu())
        end_t = float(t_end.detach().cpu())
        n_substeps = 0
        max_norm_seen = 0.0
        max_rapidity_seen = 0.0
        while current_t < end_t - 1e-15:
            t_scalar = torch.as_tensor(current_t, device=y_t.device, dtype=t_start.dtype)
            vel_tan = self._mass_shell_velocity(
                state, jet_conditions, mask, t_scalar, m, use_cfg, guidance_weight, ref_vectors)
            real_norms = tangent_norm(vel_tan).squeeze(-1)[mask > 0]
            if real_norms.numel() and not torch.isfinite(real_norms).all():
                raise MassShellIntegrationError(
                    "nonfinite_velocity",
                    "mass-shell tangent velocity contains non-finite values",
                    t_start=float(t_start), t_end=float(t_end), current_t=current_t,
                    completed_substeps=n_substeps,
                )
            max_norm = float(real_norms.max().detach().cpu()) if real_norms.numel() else 0.0
            max_norm_seen = max(max_norm_seen, max_norm)
            remaining = end_t - current_t
            allowed_dt = remaining if max_norm == 0.0 else min(
                remaining, float(max_step_rapidity) * float(m) / max_norm)
            if not allowed_dt > 0 or not torch.isfinite(torch.tensor(allowed_dt)):
                raise MassShellIntegrationError(
                    "invalid_step_size",
                    "adaptive mass-shell integration produced an invalid step size",
                    t_start=float(t_start), t_end=float(t_end), current_t=current_t,
                    completed_substeps=n_substeps, max_tangent_norm=max_norm,
                    allowed_dt=float(allowed_dt),
                )
            max_rapidity_seen = max(
                max_rapidity_seen, max_norm * allowed_dt / float(m)
            )
            n_substeps += 1
            if n_substeps > max_substeps:
                estimated_remaining = int(np.ceil(remaining / allowed_dt))
                raise MassShellIntegrationError(
                    "substep_limit",
                    f"adaptive mass-shell integration exceeded {max_substeps} substeps in "
                    f"[{float(t_start):.6g}, {float(t_end):.6g}]",
                    t_start=float(t_start), t_end=float(t_end), current_t=current_t,
                    completed_substeps=n_substeps - 1,
                    estimated_required_substeps=(n_substeps - 1 + estimated_remaining),
                    max_tangent_norm=max_norm_seen,
                    max_step_rapidity=max_rapidity_seen,
                )
            state = project_to_shell(exp_map(state, vel_tan * allowed_dt, m), m)
            if not torch.isfinite(state).all():
                finite_state = state[torch.isfinite(state)]
                raise MassShellIntegrationError(
                    "nonfinite_state",
                    "adaptive mass-shell integration produced a non-finite state",
                    t_start=float(t_start), t_end=float(t_end), current_t=current_t,
                    completed_substeps=n_substeps,
                    state_max_abs=(float(finite_state.abs().max().detach().cpu())
                                   if finite_state.numel() else None),
                    max_tangent_norm=max_norm_seen,
                    max_step_rapidity=max_rapidity_seen,
                )
            current_t = min(end_t, current_t + allowed_dt)
        if return_diagnostics:
            return state, {
                "substeps": n_substeps,
                "max_tangent_norm": max_norm_seen,
                "max_step_rapidity": max_rapidity_seen,
            }
        return state

    def _mass_shell_velocity(self, y_t, jet_conditions, mask, t, m, use_cfg,
                             guidance_weight, ref_vectors):
        """Evaluate the learned field and project it into the float64 tangent space."""
        from util.mass_shell import MassShellIntegrationError, pushforward_to_tangent

        batch_size = y_t.shape[0]
        t_batch = t.unsqueeze(0).expand(batch_size)

        model_dtype = next(self.parameters()).dtype
        model_refs = ref_vectors.to(model_dtype) if ref_vectors is not None else None
        model_state = y_t.to(model_dtype)
        vel_cartesian = self.forward(x=model_state, t=t_batch.to(model_dtype),
                                     jet_conditions=jet_conditions.to(model_dtype), mask=mask,
                                     ref_vectors=model_refs)
        if not torch.isfinite(vel_cartesian).all():
            finite_velocity = vel_cartesian[torch.isfinite(vel_cartesian)]
            raise MassShellIntegrationError(
                "nonfinite_velocity",
                "tangent-attention model produced a non-finite Cartesian velocity",
                current_t=float(t),
                model_output_max_abs=(float(finite_velocity.abs().max().detach().cpu())
                                      if finite_velocity.numel() else None),
            )
        if use_cfg:
            null_cond = self.make_null_cond(jet_conditions)
            vel_uncond = self.forward(x=model_state, t=t_batch.to(model_dtype),
                                      jet_conditions=null_cond.to(model_dtype), mask=mask,
                                      ref_vectors=model_refs)
            vel_cartesian = vel_cartesian + guidance_weight * (vel_cartesian - vel_uncond)

        if not torch.isfinite(vel_cartesian).all():
            finite_velocity = vel_cartesian[torch.isfinite(vel_cartesian)]
            raise MassShellIntegrationError(
                "nonfinite_velocity",
                "guided mass-shell Cartesian velocity contains non-finite values",
                current_t=float(t),
                model_output_max_abs=(float(finite_velocity.abs().max().detach().cpu())
                                      if finite_velocity.numel() else None),
                use_cfg=bool(use_cfg),
            )

        return pushforward_to_tangent(y_t, vel_cartesian, m) * mask.unsqueeze(-1)

    def _guided_velocity(self, x, t_scalar, jet_conditions, mask, use_cfg,
                         guidance_weight, ref_vectors, null_jet_conditions):
        """CFG-guided velocity at a single (scalar) time, used by the samplers."""
        t_batch = t_scalar.unsqueeze(0).expand(x.shape[0])
        vel = self.forward(x=x, t=t_batch, jet_conditions=jet_conditions, mask=mask, ref_vectors=ref_vectors)
        if use_cfg:
            vel_uncond = self.forward(x=x, t=t_batch, jet_conditions=null_jet_conditions, mask=mask, ref_vectors=ref_vectors)
            vel = vel + guidance_weight * (vel - vel_uncond)
        return vel

    def step(self, x_t, jet_conditions, mask, t_start, t_end, method='euler', use_cfg=False, guidance_weight=2.0, ref_vectors=None):
        """
        One ODE integration step. method='euler' (default) or 'heun' (2nd-order midpoint).
        """
        null_jet_conditions = self.make_null_cond(jet_conditions) if use_cfg else None
        dt = t_end - t_start

        v1 = self._guided_velocity(x_t, t_start, jet_conditions, mask, use_cfg,
                                   guidance_weight, ref_vectors, null_jet_conditions)
        if method == 'euler':
            return x_t + v1 * dt
        elif method == 'heun':
            # Trapezoidal RK2: average the slope at the start and at the Euler-predicted end.
            x_euler = x_t + v1 * dt
            v2 = self._guided_velocity(x_euler, t_end, jet_conditions, mask, use_cfg,
                                       guidance_weight, ref_vectors, null_jet_conditions)
            return x_t + 0.5 * (v1 + v2) * dt
        else:
            raise NotImplementedError(f"Method {method} not implemented")
        

if __name__ == "__main__":
    model = LegacyLEFTJeN(max_num_jet_types=5, max_particles=150, embed_dim=128, num_layers=6)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in LEFT-JeN with 6 layers: {total_params:,}")
