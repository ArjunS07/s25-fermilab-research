"""Softmax-free Lorentz-equivariant message passing on the regulated mass shell.

This is the single reference architecture ("variant a"): readout-only geometric state,
no global pooling. The historical tangent-channel and global-pooling variants, and the
separate tangent-attention backbone, have been removed. Parameter names
(``null_cond``, ``tangent_backbone.*``) are kept stable so existing checkpoints load.
"""
from __future__ import annotations
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

from util.geometry.mass_shell import log_map, pushforward_to_tangent
from util.geometry.minkowski_utils import dotsq4, normsq4


def _small_normal(layer):
    nn.init.normal_(layer.weight, std=1e-3); nn.init.zeros_(layer.bias)


class Geometry(NamedTuple):
    """Per-forward, layer-invariant context shared by every block (nothing here evolves)."""
    x: torch.Tensor                # (B,N,4) shell positions, float64
    cond: torch.Tensor             # (B,width) conditioning + time embedding
    mask: torch.Tensor             # (B,N) float
    edge: torch.Tensor             # (B,N,N,3) invariant edge features, model dtype
    direction: torch.Tensor        # (B,N,N,4) normalized geodesic directions, float64
    projected_refs: torch.Tensor   # (B,N,2,4) refs pushed to each tangent space, float64
    typed_refs: torch.Tensor       # (B,2,width) role-tagged reference tokens
    support: torch.Tensor          # (B,N,N) bool adjacency (no self, real-real)
    count: torch.Tensor            # (B,N) sqrt(degree), float64
    mass: float


def _geometry(x, mass, dtype):
    rel, d = log_map(x.unsqueeze(2), x.unsqueeze(1), mass, return_distance=True)
    distance = d.squeeze(-1)
    dot = dotsq4(x.unsqueeze(2), x.unsqueeze(1))
    edge = torch.stack((torch.log1p(distance),
                        torch.log1p((dot / mass**2 - 1).clamp_min(0)),
                        distance / (distance + mass)), -1).to(dtype)
    return edge, rel / (mass + distance).unsqueeze(-1)


class MassShellGNNBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.edge_mlp = nn.Sequential(nn.Linear(3 * width + 3, width), nn.SiLU(),
                                      nn.Linear(width, width))
        self.node_mlp = nn.Sequential(nn.Linear(3 * width, 2 * width), nn.SiLU(),
                                      nn.Linear(2 * width, width))

    def forward(self, h, g):
        n = h.shape[1]
        hn = self.norm(h)
        fields = [hn.unsqueeze(2).expand(-1, -1, n, -1),
                  hn.unsqueeze(1).expand(-1, n, -1, -1), g.edge,
                  g.cond[:, None, None, :].expand(-1, n, n, -1)]
        message = self.edge_mlp(torch.cat(fields, -1)) * g.support.unsqueeze(-1)
        aggregate = message.sum(2) / g.count.unsqueeze(-1).to(message.dtype)
        node_fields = [hn, aggregate, g.cond[:, None, :].expand(-1, n, -1)]
        h = (h + self.node_mlp(torch.cat(node_fields, -1))) * g.mask.unsqueeze(-1).to(h.dtype)
        return h


class MassShellGNNBackbone(nn.Module):
    def __init__(self, cond_dim, width, num_layers, regulator_mass):
        super().__init__()
        self.mass = regulator_mass
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.time_embed = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        # Node seed carries only informative invariants: <x,e_t> (energy) and <x,jet_p4>.
        # <x,x> is identically m^2 on the shell, so it is dropped (a dead constant channel).
        self.node_seed = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([MassShellGNNBlock(width) for _ in range(num_layers)])
        self.ref_roles = nn.Parameter(torch.randn(2, width))
        self.edge_readout = nn.Sequential(nn.Linear(2 * width + 3, width), nn.SiLU(), nn.Linear(width, 1))
        self.ref_readout = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 1))
        _small_normal(self.edge_readout[-1]); _small_normal(self.ref_readout[-1])

    def forward(self, x, t, conditions, mask, references):
        if references is None or references.shape[1] != 2:
            raise ValueError("mass_shell_gnn requires exactly two typed references")
        x64, refs = x.to(torch.float64), references.to(torch.float64)
        dtype = next(self.parameters()).dtype
        tf = torch.stack([t, torch.sin(torch.pi * t), torch.cos(torch.pi * t)], -1)
        cond = self.cond_embed(conditions.to(dtype)) + self.time_embed(tf.to(dtype))
        seed = torch.stack([dotsq4(x64, refs[:, :1]), dotsq4(x64, refs[:, 1:2])], -1)
        h = (self.node_seed((seed.sign() * torch.log1p(seed.abs())).to(dtype)) + cond[:, None])
        h *= mask.unsqueeze(-1).to(dtype)
        n = x.shape[1]
        support = (mask.bool().unsqueeze(2) & mask.bool().unsqueeze(1) &
                   ~torch.eye(n, device=x.device, dtype=torch.bool).unsqueeze(0))
        count = support.sum(-1).clamp_min(1).to(torch.float64).sqrt()
        edge, direction = _geometry(x64, self.mass, dtype)
        projected = pushforward_to_tangent(x64.unsqueeze(2), refs.unsqueeze(1).expand(-1, n, -1, -1), self.mass)
        ref_norm = torch.sqrt((-normsq4(projected)).clamp_min(0))
        projected = projected / (self.mass + ref_norm).unsqueeze(-1)
        projected = projected * mask[:, :, None, None].to(torch.float64)
        typed_refs = cond[:, None, :] + self.ref_roles[None]
        g = Geometry(x=x64, cond=cond, mask=mask, edge=edge, direction=direction,
                     projected_refs=projected, typed_refs=typed_refs, support=support,
                     count=count, mass=self.mass)
        for block in self.blocks:
            h = block(h, g)
        hi = h.unsqueeze(2).expand(-1, -1, n, -1); hj = h.unsqueeze(1).expand(-1, n, -1, -1)
        coeff = self.edge_readout(torch.cat([hi, hj, edge], -1)).squeeze(-1).to(torch.float64) * support
        velocity = torch.einsum("bij,bijf->bif", coeff, direction) / count.unsqueeze(-1)
        rcoeff = self.ref_readout(torch.cat([h.unsqueeze(2).expand(-1, -1, 2, -1),
            typed_refs.unsqueeze(1).expand(-1, n, -1, -1)], -1)).squeeze(-1)
        velocity += torch.einsum("bir,birf->bif", rcoeff.to(torch.float64), projected)
        return pushforward_to_tangent(x64, velocity, self.mass) * mask.unsqueeze(-1).to(torch.float64)


class MassShellGNNFlow(nn.Module):
    """Typed-reference mass-shell RFM velocity field on ``H_m`` (the reference model).

    Parameter names intentionally match existing checkpoints (``null_cond`` and
    ``tangent_backbone.*``).
    """

    def __init__(self, condition_dim, n_particle_types, width, num_layers, regulator_mass):
        super().__init__()
        self.cond_dim, self.n_particles_idx = condition_dim, n_particle_types
        self.regulator_mass, self.backbone = regulator_mass, "mass_shell_gnn"
        self.null_cond = nn.Parameter(torch.zeros(condition_dim))
        self.tangent_backbone = MassShellGNNBackbone(condition_dim, width, num_layers, regulator_mass)

    def make_null_cond(self, conditions: torch.Tensor) -> torch.Tensor:
        """Replace physical conditions by a learned null, retaining multiplicity."""
        null = self.null_cond.unsqueeze(0).expand(conditions.shape[0], -1)
        keep = torch.zeros(conditions.shape[-1], device=conditions.device)
        keep[self.n_particles_idx] = 1.0
        return null * (1 - keep) + conditions * keep

    def forward(self, x, t, jet_conditions, mask, ref_vectors=None):
        if ref_vectors is None:
            raise ValueError("mass_shell_gnn requires the two typed reference vectors")
        if jet_conditions.shape[-1] != self.cond_dim:
            raise ValueError(
                f"MassShellGNNFlow expected {self.cond_dim} conditions, "
                f"got {jet_conditions.shape[-1]}"
            )
        return self.tangent_backbone(x, t, jet_conditions, mask, ref_vectors)

    def _mass_shell_velocity(self, state, conditions, mask, t, use_cfg,
                             guidance_weight, references):
        from util.geometry.mass_shell import MassShellIntegrationError, pushforward_to_tangent

        batch_t = t.unsqueeze(0).expand(state.shape[0])
        model_dtype = next(self.parameters()).dtype
        model_refs = references.to(model_dtype) if references is not None else None
        velocity = self.forward(
            state, batch_t.to(model_dtype), conditions.to(model_dtype), mask, model_refs
        )
        if not torch.isfinite(velocity).all():
            raise MassShellIntegrationError(
                "nonfinite_velocity", "mass-shell model produced a non-finite velocity",
                current_t=float(t),
            )
        if use_cfg:
            unconditional = self.forward(
                state, batch_t.to(model_dtype),
                self.make_null_cond(conditions).to(model_dtype), mask, model_refs,
            )
            velocity = velocity + guidance_weight * (velocity - unconditional)
        if not torch.isfinite(velocity).all():
            raise MassShellIntegrationError(
                "nonfinite_velocity", "guided mass-shell velocity is non-finite",
                current_t=float(t), use_cfg=bool(use_cfg),
            )
        return pushforward_to_tangent(state, velocity, self.regulator_mass) * mask.unsqueeze(-1)

    def step_hyperbolic(self, y_t, jet_conditions, mask, t_start, t_end,
                        c=1.0, use_cfg=False, guidance_weight=2.0,
                        ref_vectors=None, hyperbolic_model="mass_shell",
                        regulator_mass=None, max_step_rapidity=None,
                        max_substeps=64, return_diagnostics=False):
        """Advance one nominal interval with optional rapidity-controlled substeps."""
        del c
        if hyperbolic_model != "mass_shell":
            raise ValueError("MassShellGNNFlow only supports mass-shell integration")
        if regulator_mass is not None and regulator_mass != self.regulator_mass:
            raise ValueError("sampler regulator mass differs from model regulator mass")

        from util.geometry.mass_shell import (MassShellIntegrationError, exp_map,
                                     project_to_shell, tangent_norm)

        current = y_t
        current_t = float(t_start.detach().cpu())
        end_t = float(t_end.detach().cpu())
        substeps = 0
        max_norm_seen = 0.0
        max_rapidity_seen = 0.0
        while current_t < end_t - 1e-15:
            t = torch.as_tensor(current_t, device=y_t.device, dtype=t_start.dtype)
            velocity = self._mass_shell_velocity(
                current, jet_conditions, mask, t, use_cfg, guidance_weight, ref_vectors
            )
            real_norms = tangent_norm(velocity).squeeze(-1)[mask > 0]
            if real_norms.numel() and not torch.isfinite(real_norms).all():
                raise MassShellIntegrationError(
                    "nonfinite_velocity", "tangent velocity contains non-finite values",
                    current_t=current_t, completed_substeps=substeps,
                )
            max_norm = float(real_norms.max().detach().cpu()) if real_norms.numel() else 0.0
            max_norm_seen = max(max_norm_seen, max_norm)
            remaining = end_t - current_t
            if max_step_rapidity is None or max_norm == 0.0:
                allowed_dt = remaining
            else:
                if max_step_rapidity <= 0 or max_substeps < 1:
                    raise ValueError("adaptive mass-shell limits must be positive")
                allowed_dt = min(
                    remaining, max_step_rapidity * self.regulator_mass / max_norm
                )
            if not allowed_dt > 0 or not np.isfinite(allowed_dt):
                raise MassShellIntegrationError(
                    "invalid_step_size", "adaptive sampler produced an invalid step size",
                    current_t=current_t, allowed_dt=float(allowed_dt),
                )
            substeps += 1
            if substeps > max_substeps:
                estimated_required = (substeps - 1) + int(np.ceil(remaining / allowed_dt))
                raise MassShellIntegrationError(
                    "substep_limit",
                    f"adaptive sampler exceeded {max_substeps} substeps",
                    current_t=current_t, completed_substeps=substeps - 1,
                    max_tangent_norm=max_norm_seen,
                    estimated_required_substeps=estimated_required,
                )
            max_rapidity_seen = max(
                max_rapidity_seen,
                max_norm * allowed_dt / self.regulator_mass,
            )
            current = project_to_shell(exp_map(current, velocity * allowed_dt,
                                               self.regulator_mass), self.regulator_mass)
            if not torch.isfinite(current).all():
                raise MassShellIntegrationError(
                    "nonfinite_state", "mass-shell step produced a non-finite state",
                    current_t=current_t, completed_substeps=substeps,
                )
            current_t = min(end_t, current_t + allowed_dt)

        if return_diagnostics:
            return current, {
                "substeps": substeps,
                "max_tangent_norm": max_norm_seen,
                "max_step_rapidity": max_rapidity_seen,
            }
        return current
