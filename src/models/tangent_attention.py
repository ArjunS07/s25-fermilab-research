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

    def forward(self, x64, h, vectors64, cond, mask, geometry, m):
        batch, particles, _ = x64.shape
        edge, rel64, projected_refs, support = geometry

        h0 = self.norm1(h)
        scale, shift = self.cond_mod(cond).chunk(2, dim=-1)
        h0 = h0 * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        q, k, value = self.qkv(h0).chunk(3, dim=-1)
        q = q.view(batch, particles, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, particles, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, particles, self.heads, self.head_dim).transpose(1, 2)

        xi = x64.unsqueeze(2)
        xj = x64.unsqueeze(1)
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
                 readout_init: str = "small_normal"):
        super().__init__()
        self.regulator_mass = regulator_mass
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.time_embed = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.node_seed = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([
            TangentAttentionBlock(width, num_heads, vector_channels, width)
            for _ in range(num_layers)
        ])
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
        particles = x64.shape[1]
        real = mask > 0
        eye = torch.eye(particles, device=x64.device, dtype=torch.bool).unsqueeze(0)
        support = real.unsqueeze(2) & real.unsqueeze(1) & ~eye
        edge, rel64 = _shell_edge_features(
            x64.unsqueeze(2), x64.unsqueeze(1), self.regulator_mass, dtype
        )
        projected_refs = pushforward_to_tangent(
            x64.unsqueeze(2), refs64.unsqueeze(1).expand(-1, particles, -1, -1),
            self.regulator_mass,
        )
        geometry = edge, rel64, projected_refs, support
        for block in self.blocks:
            h, vectors = block(
                x64, h, vectors, cond, mask, geometry, self.regulator_mass
            )
        coeff = self.readout(h).to(torch.float64)
        velocity = torch.einsum("bic,bicf->bif", coeff, vectors)
        velocity = pushforward_to_tangent(x64, velocity, self.regulator_mass)
        return velocity * mask.to(torch.float64).unsqueeze(-1)
