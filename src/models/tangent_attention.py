"""Typed scalar--tangent-vector attention backbone for mass-shell RFM.

The physical ODE point cloud stays fixed during a network evaluation. Learned scalar
features and tangent-vector channels are refined instead; the final readout is tangent by
construction. Particle permutation equivariance is retained while the two references have
explicit, non-exchangeable roles (lab time and conditioning jet).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from util.mass_shell import log_map, parallel_transport, pushforward_to_tangent
from util.minkowski_utils import dotsq4, normsq4


def _masked_softmax(logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    neg = torch.finfo(logits.dtype).min
    weights = torch.softmax(logits.masked_fill(~support.unsqueeze(1), neg), dim=-1)
    valid = support.any(dim=-1, keepdim=True).unsqueeze(1)
    return torch.where(valid, weights, torch.zeros_like(weights))


def mass_shell_barycenter(x64: torch.Tensor, mask: torch.Tensor, m: float) -> torch.Tensor:
    """Lorentz-covariant, permutation-invariant anchor on the future mass shell."""
    real = mask > 0
    if not bool(real.any(dim=1).all()):
        raise ValueError("jet token requires at least one real constituent per jet")
    summed = (x64 * real.to(torch.float64).unsqueeze(-1)).sum(dim=1)
    summed_norm = normsq4(summed)
    if not bool(torch.isfinite(summed).all() and torch.isfinite(summed_norm).all()):
        raise FloatingPointError("non-finite mass-shell jet-token anchor")
    if not bool((summed_norm > 0).all()):
        raise FloatingPointError("mass-shell jet-token anchor is not timelike")
    anchor = m * summed / torch.sqrt(summed_norm).unsqueeze(-1)
    if not bool((anchor[..., 0] > 0).all()):
        raise FloatingPointError("mass-shell jet-token anchor is not future-directed")
    return anchor


def _shell_edge_features(p64: torch.Tensor, q64: torch.Tensor, m: float,
                         dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    relative64 = log_map(p64, q64, m)
    distance = torch.sqrt((-normsq4(relative64)).clamp(min=0.0))
    pair_dot = dotsq4(p64, q64)
    edge = torch.stack([
        torch.log1p(distance),
        torch.log1p((pair_dot / (m * m) - 1).clamp(min=0.0)),
        distance / (distance + m),
    ], dim=-1).to(dtype)
    return edge, relative64


class JetTokenInteraction(nn.Module):
    """Typed global gather/broadcast path; the token is not a constituent node."""

    def __init__(self, width: int, heads: int, vector_channels: int, mode: str):
        super().__init__()
        if mode not in ("scalar", "scalar_vector"):
            raise ValueError(f"unknown jet token mode {mode!r}")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.vector_channels = vector_channels
        self.has_vectors = mode == "scalar_vector"

        self.particle_norm = nn.LayerNorm(width)
        self.token_norm = nn.LayerNorm(width)
        self.token_query = nn.Linear(width, width)
        self.particle_kv = nn.Linear(width, 2 * width)
        self.edge_bias = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, heads))
        self.gather_out = nn.Linear(width, width)
        self.token_ff_norm = nn.LayerNorm(width)
        self.token_ff = nn.Sequential(
            nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width)
        )
        self.broadcast_gate = nn.Sequential(
            nn.Linear(2 * width + 3, width), nn.SiLU(), nn.Linear(width, width), nn.Sigmoid()
        )
        self.broadcast_value = nn.Linear(width, width)

        if self.has_vectors:
            self.gather_vector_gate = nn.Sequential(
                nn.Linear(2 * width + 3, width), nn.SiLU(), nn.Linear(width, vector_channels)
            )
            self.token_vector_mix = nn.Parameter(torch.eye(vector_channels))
            self.broadcast_vector_gate = nn.Sequential(
                nn.Linear(2 * width + 3, width), nn.SiLU(),
                nn.Linear(width, 2 * vector_channels)
            )

    def forward(self, x64, h, vectors64, token, token_vectors64, anchor64, mask, m):
        batch, particles, _ = x64.shape
        real = mask > 0
        particle_state = self.particle_norm(h)
        token_state = self.token_norm(token)
        edge, from_anchor64 = _shell_edge_features(
            anchor64.unsqueeze(1), x64, m, h.dtype
        )

        query = self.token_query(token_state).view(batch, self.heads, self.head_dim)
        key, value = self.particle_kv(particle_state).chunk(2, dim=-1)
        key = key.view(batch, particles, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, particles, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.einsum("bhd,bhjd->bhj", query, key) / math.sqrt(self.head_dim)
        logits = logits + self.edge_bias(edge).permute(0, 2, 1)
        neg = torch.finfo(logits.dtype).min
        attention = torch.softmax(logits.masked_fill(~real.unsqueeze(1), neg), dim=-1)
        gathered = torch.einsum("bhj,bhjd->bhd", attention, value).reshape(batch, self.width)
        token = token + self.gather_out(gathered)
        token = token + self.token_ff(self.token_ff_norm(token))

        updated_token_state = self.token_norm(token)
        token_expanded = updated_token_state.unsqueeze(1).expand(-1, particles, -1)
        gate_input = torch.cat([particle_state, token_expanded, edge], dim=-1)
        scalar_gate = self.broadcast_gate(gate_input)
        h = h + scalar_gate * self.broadcast_value(updated_token_state).unsqueeze(1)
        h = h * mask.to(h.dtype).unsqueeze(-1)

        if self.has_vectors:
            geom_attention64 = attention.mean(dim=1).to(torch.float64)
            transported = parallel_transport(
                x64.unsqueeze(2), anchor64[:, None, None, :], vectors64, m
            )
            transported_agg = torch.einsum("bi,bicf->bcf", geom_attention64, transported)
            transported_agg = torch.einsum(
                "bcf,dc->bdf", transported_agg, self.token_vector_mix.to(torch.float64)
            )
            gather_gate = self.gather_vector_gate(gate_input).to(torch.float64)
            relative_agg = torch.einsum(
                "bi,bic,bif->bcf", geom_attention64, gather_gate, from_anchor64
            )
            token_vectors64 = token_vectors64 + transported_agg + relative_agg
            token_vectors64 = pushforward_to_tangent(
                anchor64.unsqueeze(1), token_vectors64, m
            )

            vector_gates = self.broadcast_vector_gate(gate_input).to(torch.float64)
            channel_gate, relative_gate = vector_gates.chunk(2, dim=-1)
            token_at_particles = parallel_transport(
                anchor64[:, None, None, :], x64.unsqueeze(2), token_vectors64.unsqueeze(1), m
            )
            to_anchor64 = log_map(x64, anchor64.unsqueeze(1), m)
            vectors64 = vectors64 + channel_gate.unsqueeze(-1) * token_at_particles
            vectors64 = vectors64 + relative_gate.unsqueeze(-1) * to_anchor64.unsqueeze(2)
            vectors64 = pushforward_to_tangent(x64.unsqueeze(2), vectors64, m)
            vectors64 = vectors64 * mask.to(torch.float64).unsqueeze(-1).unsqueeze(-1)

        diagnostics = {
            "attention_entropy": (-(attention.clamp_min(1e-12).log() * attention).sum(-1).mean()).detach(),
            "broadcast_gate_mean": scalar_gate.mean().detach(),
            "broadcast_gate_saturation": ((scalar_gate < 0.01) | (scalar_gate > 0.99)).float().mean().detach(),
            "token_scalar_norm": token.norm(dim=-1).mean().detach(),
        }
        if self.has_vectors:
            diagnostics["token_vector_norm"] = torch.sqrt(
                (-normsq4(token_vectors64)).clamp(min=0.0)
            ).mean().detach()
        return h, vectors64, token, token_vectors64, diagnostics


class TangentAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, vector_channels: int, cond_width: int):
        super().__init__()
        if width % heads:
            raise ValueError("tangent-attention width must be divisible by num_heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.vector_channels = vector_channels

        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.edge_bias = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, heads))
        self.scalar_out = nn.Linear(width, width)

        self.edge_vector_gate = nn.Sequential(
            nn.Linear(2 * width + 3, width), nn.SiLU(), nn.Linear(width, vector_channels)
        )
        self.vector_mix = nn.Parameter(torch.eye(vector_channels))
        self.ref_gate = nn.Sequential(
            nn.Linear(width + cond_width, width), nn.SiLU(), nn.Linear(width, vector_channels)
        )
        self.role_embedding = nn.Embedding(2, cond_width)

        self.cond_mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_width, 2 * width))
        nn.init.zeros_(self.cond_mod[-1].weight)
        nn.init.zeros_(self.cond_mod[-1].bias)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width)
        )

    def forward(self, x64, h, vectors64, cond, mask, refs64, m):
        batch, particles, _ = x64.shape
        real = mask > 0
        eye = torch.eye(particles, device=x64.device, dtype=torch.bool).unsqueeze(0)
        support = real.unsqueeze(2) & real.unsqueeze(1) & ~eye

        h0 = self.norm1(h)
        scale, shift = self.cond_mod(cond).chunk(2, dim=-1)
        h0 = h0 * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        q, k, value = self.qkv(h0).chunk(3, dim=-1)
        q = q.view(batch, particles, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, particles, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, particles, self.heads, self.head_dim).transpose(1, 2)

        xi = x64.unsqueeze(2)
        xj = x64.unsqueeze(1)
        edge, rel64 = _shell_edge_features(xi, xj, m, h.dtype)

        logits = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        logits = logits + self.edge_bias(edge).permute(0, 3, 1, 2)
        attn = _masked_softmax(logits, support)
        scalar_message = torch.einsum("bhij,bhjd->bhid", attn, value)
        scalar_message = scalar_message.transpose(1, 2).reshape(batch, particles, self.width)
        h = h + self.scalar_out(scalar_message)

        # Average attention over heads for geometric aggregation. Vector values are first
        # transported from the sender tangent space into the receiver tangent space.
        geom_attn = attn.mean(dim=1)
        geom_attn64 = geom_attn.to(torch.float64)
        vi = vectors64.unsqueeze(2).expand(-1, -1, particles, -1, -1)
        vj = vectors64.unsqueeze(1).expand(-1, particles, -1, -1, -1)
        transported = parallel_transport(
            xj.unsqueeze(3), xi.unsqueeze(3), vj, m
        )
        transported_agg = torch.einsum("bij,bijcf->bicf", geom_attn64, transported)
        mixed = torch.einsum("bicf,dc->bidf", transported_agg, self.vector_mix.to(torch.float64))

        hi = h0.unsqueeze(2).expand(-1, -1, particles, -1)
        hj = h0.unsqueeze(1).expand(-1, particles, -1, -1)
        edge_gate = self.edge_vector_gate(torch.cat([hi, hj, edge], dim=-1)).to(torch.float64)
        relative_agg = torch.einsum(
            "bij,bijc,bijf->bicf", geom_attn64, edge_gate, rel64
        )

        # References have typed scalar roles; swapping their four-vectors is intentionally
        # not a symmetry. They are projected into each particle's tangent space rather than
        # treated as points on the constituent shell.
        roles = self.role_embedding(torch.arange(2, device=h.device))
        role_cond = cond.unsqueeze(1) + roles.unsqueeze(0)
        # The gate network expects [node state, typed condition].
        typed = role_cond.unsqueeze(1).expand(-1, particles, -1, -1)
        node = h0.unsqueeze(2).expand(-1, -1, 2, -1)
        ref_gates = self.ref_gate(torch.cat([node, typed], dim=-1)).to(torch.float64)
        projected_refs = pushforward_to_tangent(
            x64.unsqueeze(2), refs64.unsqueeze(1).expand(-1, particles, -1, -1), m
        )
        ref_update = torch.einsum("birc,birf->bicf", ref_gates, projected_refs)

        vectors64 = vectors64 + mixed + relative_agg + ref_update
        vectors64 = pushforward_to_tangent(
            x64.unsqueeze(2), vectors64, m
        ) * mask.to(torch.float64).unsqueeze(-1).unsqueeze(-1)
        h = h + self.ff(self.norm2(h))
        h = h * mask.to(h.dtype).unsqueeze(-1)
        return h, vectors64


class TangentAttentionBackbone(nn.Module):
    def __init__(self, cond_dim: int, width: int, num_layers: int, num_heads: int,
                 vector_channels: int, regulator_mass: float,
                 readout_init: str = "small_normal", jet_token_mode: str = "none"):
        super().__init__()
        if jet_token_mode not in ("none", "scalar", "scalar_vector"):
            raise ValueError(f"unknown jet token mode {jet_token_mode!r}")
        self.regulator_mass = regulator_mass
        self.jet_token_mode = jet_token_mode
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.time_embed = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.node_seed = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([
            TangentAttentionBlock(width, num_heads, vector_channels, width)
            for _ in range(num_layers)
        ])
        if jet_token_mode != "none":
            self.token_type = nn.Parameter(torch.zeros(width))
            self.token_seed = nn.Sequential(
                nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)
            )
            self.token_interactions = nn.ModuleList([
                JetTokenInteraction(width, num_heads, vector_channels, jet_token_mode)
                for _ in range(num_layers)
            ])
            self.token_readout = nn.Sequential(
                nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.SiLU(),
                nn.Linear(width, vector_channels),
            )
            output_layer = self.token_readout[-1]
            if readout_init == "small_normal":
                nn.init.normal_(output_layer.weight, std=1e-3)
            elif readout_init == "zero":
                nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
        self.readout = nn.Linear(width, vector_channels)
        if readout_init == "small_normal":
            nn.init.normal_(self.readout.weight, std=1e-3)
        elif readout_init == "zero":
            nn.init.zeros_(self.readout.weight)
        else:
            raise ValueError(f"unknown tangent velocity readout initialization {readout_init!r}")
        nn.init.zeros_(self.readout.bias)

    def forward(self, x, t, conditions, mask, references):
        if references is None or references.shape[1] != 2:
            raise ValueError("tangent_attention requires exactly two typed references")
        x64 = x.to(torch.float64)
        refs64 = references.to(torch.float64)
        dtype = next(self.parameters()).dtype
        cond = self.cond_embed(conditions.to(dtype))
        time_features = torch.stack([t, torch.sin(torch.pi * t), torch.cos(torch.pi * t)], dim=-1)
        cond = cond + self.time_embed(time_features.to(dtype))

        et = refs64[:, 0:1]
        jet = refs64[:, 1:2]
        seed = torch.stack([
            normsq4(x64), dotsq4(x64, et), dotsq4(x64, jet)
        ], dim=-1)
        seed = torch.sign(seed) * torch.log1p(seed.abs())
        h = ((self.node_seed(seed.to(dtype)) + cond.unsqueeze(1))
             * mask.to(dtype).unsqueeze(-1))
        vectors = torch.zeros(
            *x64.shape[:2], self.blocks[0].vector_channels, 4,
            dtype=torch.float64, device=x64.device,
        )
        if self.jet_token_mode != "none":
            anchor64 = mass_shell_barycenter(x64, mask, self.regulator_mass)
            token_features = torch.stack([
                normsq4(anchor64), dotsq4(anchor64, et.squeeze(1)),
                dotsq4(anchor64, jet.squeeze(1)),
            ], dim=-1)
            token_features = torch.sign(token_features) * torch.log1p(token_features.abs())
            token = self.token_seed(token_features.to(dtype)) + cond + self.token_type.unsqueeze(0)
            token_vectors = torch.zeros(
                x64.shape[0], self.blocks[0].vector_channels, 4,
                dtype=torch.float64, device=x64.device,
            )
            self.last_token_diagnostics = []
        for index, block in enumerate(self.blocks):
            h, vectors = block(x64, h, vectors, cond, mask, refs64, self.regulator_mass)
            if self.jet_token_mode != "none":
                h, vectors, token, token_vectors, diagnostics = self.token_interactions[index](
                    x64, h, vectors, token, token_vectors, anchor64, mask,
                    self.regulator_mass,
                )
                self.last_token_diagnostics.append(diagnostics)
        if self.jet_token_mode == "none":
            coeff = self.readout(h).to(torch.float64)
        else:
            expanded_token = token.unsqueeze(1).expand(-1, x64.shape[1], -1)
            coeff = self.token_readout(torch.cat([h, expanded_token], dim=-1)).to(torch.float64)
        velocity = torch.einsum("bic,bicf->bif", coeff, vectors)
        velocity = pushforward_to_tangent(x64, velocity, self.regulator_mass)
        return velocity * mask.to(torch.float64).unsqueeze(-1)
