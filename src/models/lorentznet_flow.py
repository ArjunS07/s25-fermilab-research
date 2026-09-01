"""The H Lorentz-equivariant mass-shell flow field."""
from __future__ import annotations

import torch
import torch.nn as nn

from util.geometry.mass_shell import exp_map, log_map, project_to_shell, pushforward_to_tangent
from util.geometry.minkowski_utils import dotsq4, normsq4


def signed_log(value: torch.Tensor) -> torch.Tensor:
    return value.sign() * torch.log1p(value.abs())


def _small_normal(layer: nn.Linear, std: float = 1e-3) -> None:
    nn.init.normal_(layer.weight, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _pair_invariants(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    yi, yj = y.unsqueeze(2), y.unsqueeze(1)
    return signed_log(normsq4(yi - yj)), signed_log(dotsq4(yi, yj))


class LorentzNetLGEB(nn.Module):
    """H's scalar-message and equivariant-coordinate update block."""

    def __init__(self, width: int):
        super().__init__()
        self.coordinate_scale = 1e-2
        self.scalar_norm = nn.LayerNorm(width)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * width + 2, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
        )
        coordinate_final = nn.Linear(width, 1)
        _small_normal(coordinate_final)
        self.coordinate_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), coordinate_final)

    def forward(self, h: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                support: torch.Tensor, support_f: torch.Tensor,
                sqrt_degree: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_nodes = h.shape[1]
        hn = self.scalar_norm(h)
        hi = hn.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        hj = hn.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        diff_sq, dot = _pair_invariants(y)
        edge = torch.stack((diff_sq, dot), dim=-1).to(h.dtype)
        message = self.message_mlp(torch.cat((hi, hj, edge), dim=-1)) * support_f
        aggregate = message.sum(dim=2) / sqrt_degree
        h = (h + self.node_mlp(torch.cat((hn, aggregate), dim=-1))) * mask.unsqueeze(-1).to(h.dtype)

        displacement = y.unsqueeze(2) - y.unsqueeze(1)
        coefficient = self.coordinate_mlp(message).to(y.dtype) * support.unsqueeze(-1)
        delta = (coefficient * displacement).sum(dim=2) / sqrt_degree.to(y.dtype)
        y = (y + self.coordinate_scale * delta) * mask.unsqueeze(-1).to(y.dtype)
        return h, y


class LorentzNetBackbone(nn.Module):
    """H's invariant-scalar/equivariant-vector backbone and field head."""

    def __init__(self, condition_dim: int, width: int, num_layers: int, regulator_mass: float):
        super().__init__()
        self.regulator_mass = float(regulator_mass)
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.time_embed = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.node_seed = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([LorentzNetLGEB(width) for _ in range(num_layers)])
        self.field_mlp = nn.Sequential(
            nn.Linear(2 * width + 2, width), nn.SiLU(), nn.Linear(width, 1)
        )
        _small_normal(self.field_mlp[-1])
        self.reference_mlp = nn.Sequential(
            nn.Linear(width + 2, width), nn.SiLU(), nn.Linear(width, 2)
        )
        _small_normal(self.reference_mlp[-1])

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor,
                mask: torch.Tensor, references: torch.Tensor | None = None) -> torch.Tensor:
        if references is None or references.shape[1] != 2:
            raise ValueError("H requires ordered (e_t, jet_p4) references")
        dtype = next(self.parameters()).dtype
        x64 = x.to(torch.float64) * mask.unsqueeze(-1).to(torch.float64)
        y = x.to(dtype) * mask.unsqueeze(-1).to(dtype)
        n_nodes = y.shape[1]
        real = mask.bool()
        support = (real.unsqueeze(2) & real.unsqueeze(1)
                   & ~torch.eye(n_nodes, device=x.device, dtype=torch.bool).unsqueeze(0))
        support_f = support.unsqueeze(-1).to(dtype)
        sqrt_degree = support.sum(dim=2).clamp_min(1).to(dtype).sqrt().unsqueeze(-1)

        time_features = torch.stack((t, torch.sin(torch.pi * t), torch.cos(torch.pi * t)), dim=-1)
        seed = signed_log(dotsq4(y.unsqueeze(2), references.to(dtype).unsqueeze(1)))
        h = (self.node_seed(seed) - self.time_embed(time_features.to(dtype)).unsqueeze(1)
             + self.condition_embed(conditions.to(dtype)).unsqueeze(1))
        h = h * mask.unsqueeze(-1).to(dtype)
        for block in self.blocks:
            h, y = block(h, y, mask, support, support_f, sqrt_degree)

        hi = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        hj = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        diff_sq, dot = _pair_invariants(y)
        edge = torch.stack((diff_sq, dot), dim=-1).to(dtype)
        coefficients = self.field_mlp(torch.cat((hi, hj, edge), dim=-1)).squeeze(-1)
        relative, distance = log_map(
            x64.unsqueeze(2), x64.unsqueeze(1), self.regulator_mass, return_distance=True
        )
        direction = relative / (self.regulator_mass + distance)
        raw = torch.einsum("bij,bijf->bif", coefficients.to(torch.float64) * support, direction)
        raw = raw / support.sum(dim=2).clamp_min(1).to(torch.float64).sqrt().unsqueeze(-1)

        refs = references.to(dtype)
        reference_invariants = signed_log(dotsq4(y.unsqueeze(2), refs.unsqueeze(1)))
        reference_coefficients = self.reference_mlp(torch.cat((h, reference_invariants), dim=-1))
        expanded_refs = references.to(torch.float64).unsqueeze(1).expand(-1, n_nodes, -1, -1)
        projected = pushforward_to_tangent(x64.unsqueeze(2), expanded_refs, self.regulator_mass)
        projected_norm = torch.sqrt((-normsq4(projected)).clamp_min(0))
        normalized_refs = projected / (self.regulator_mass + projected_norm).unsqueeze(-1)
        reference_field = torch.einsum(
            "bir,birf->bif", reference_coefficients.to(torch.float64), normalized_refs
        )
        return (raw + reference_field) * mask.unsqueeze(-1).to(dtype)


class LorentzNetFlow(nn.Module):
    """H: LorentzNet with mass-shell Riemannian flow matching."""

    def __init__(self, condition_dim: int, n_particle_types: int, width: int = 96,
                 num_layers: int = 6, regulator_mass: float = 0.1):
        super().__init__()
        self.cond_dim = condition_dim
        self.n_particles_idx = n_particle_types
        self.regulator_mass = float(regulator_mass)
        self.null_cond = nn.Parameter(torch.zeros(condition_dim))
        self.lorentznet_backbone = LorentzNetBackbone(
            condition_dim, width, num_layers, regulator_mass
        )

    def make_null_cond(self, conditions: torch.Tensor) -> torch.Tensor:
        null = self.null_cond.unsqueeze(0).expand(conditions.shape[0], -1)
        keep = torch.zeros(conditions.shape[-1], device=conditions.device)
        keep[self.n_particles_idx] = 1.0
        return null * (1 - keep) + conditions * keep

    def forward(self, x, t, jet_conditions, mask, ref_vectors=None):
        if jet_conditions.shape[-1] != self.cond_dim:
            raise ValueError(
                f"LorentzNetFlow expected {self.cond_dim} conditions, got {jet_conditions.shape[-1]}"
            )
        raw = self.lorentznet_backbone(x, t, jet_conditions, mask, ref_vectors)
        return pushforward_to_tangent(
            x.to(torch.float64), raw.to(torch.float64), self.regulator_mass
        ) * mask.unsqueeze(-1).to(raw.dtype)

    def _mass_shell_velocity(self, state, conditions, mask, t, use_cfg,
                             guidance_weight, references):
        batch_t = t.unsqueeze(0).expand(state.shape[0])
        dtype = next(self.parameters()).dtype
        refs = references.to(dtype) if references is not None else None
        velocity = self.forward(state, batch_t.to(dtype), conditions.to(dtype), mask, refs)
        if use_cfg:
            unconditional = self.forward(
                state, batch_t.to(dtype), self.make_null_cond(conditions).to(dtype), mask, refs
            )
            velocity = velocity + guidance_weight * (velocity - unconditional)
        return velocity * mask.unsqueeze(-1)

    def step_hyperbolic(self, y_t, jet_conditions, mask, t_start, t_end,
                        use_cfg=False, guidance_weight=2.0, ref_vectors=None):
        t = torch.as_tensor(float(t_start.detach().cpu()), device=y_t.device, dtype=t_start.dtype)
        velocity = self._mass_shell_velocity(
            y_t, jet_conditions, mask, t, use_cfg, guidance_weight, ref_vectors
        )
        stepped = project_to_shell(
            exp_map(y_t, velocity * (float(t_end.detach().cpu()) - float(t_start.detach().cpu())),
                    self.regulator_mass),
            self.regulator_mass,
        )
        if not torch.isfinite(stepped).all():
            raise FloatingPointError("mass-shell Euler step produced a non-finite state")
        return stepped


def build_lorentznet(max_num_jet_types: int, *, num_layers: int = 6,
                     hidden_dim: int = 96, regulator_mass: float = 0.1) -> LorentzNetFlow:
    return LorentzNetFlow(
        condition_dim=max_num_jet_types + 3,
        n_particle_types=max_num_jet_types,
        width=hidden_dim,
        num_layers=num_layers,
        regulator_mass=regulator_mass,
    )


__all__ = ["LorentzNetFlow", "build_lorentznet"]
