"""LorentzNet-style equivariant vector field for matched FM geometry ablations.

This module is an independent implementation of the LorentzNet design described in
Gong et al. (arXiv:2201.08187).  In particular, the coordinate update follows the
released implementation and uses ``y_i - y_j``.  Equation 3.3 in the paper is written
with a different vector basis; the release's difference update is used here because it
is translation invariant, manifestly Lorentz equivariant, and is the implementation
whose empirical behaviour defines the architectural reference.

The latent vectors ``y`` are unconstrained equivariant features.  Only a mass-shell
model's final raw field is projected to the tangent space at the physical input ``x``.
No log maps, tangent bases, geodesic features, reference tokens, or component clamps
appear in the backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from util.geometry.mass_shell import log_map, pushforward_to_tangent
from util.geometry.minkowski_utils import dotsq4, normsq4


def signed_log(value: torch.Tensor) -> torch.Tensor:
    """LorentzNet's signed logarithmic compression, applied only to invariants."""
    return value.sign() * torch.log1p(value.abs())


def _small_normal(layer: nn.Linear, std: float = 1e-3) -> None:
    nn.init.normal_(layer.weight, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _pair_invariants(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed-log ((y_i-y_j)^2, <y_i,y_j>) for every ordered pair."""
    yi = y.unsqueeze(2)
    yj = y.unsqueeze(1)
    return signed_log(normsq4(yi - yj)), signed_log(dotsq4(yi, yj))


def _physical_geodesic_geometry(
    x: torch.Tensor, mass: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invariant edge features, distance, and bounded log-map directions on H_m."""
    relative, distance = log_map(
        x.unsqueeze(2), x.unsqueeze(1), mass, return_distance=True
    )
    d = distance.squeeze(-1)
    dot = dotsq4(x.unsqueeze(2), x.unsqueeze(1))
    edge = torch.stack(
        (
            torch.log1p(d),
            torch.log1p((dot / mass**2 - 1).clamp_min(0)),
            d / (d + mass),
        ),
        dim=-1,
    ).to(dtype)
    return edge, distance, relative / (mass + distance)


class LorentzNetLGEB(nn.Module):
    """One scalar-message, scalar-node, equivariant-vector update block."""

    def __init__(
        self,
        width: int,
        geometry_mode: str = "evolving_auxiliary",
        coordinate_scale: float = 1e-2,
    ):
        super().__init__()
        if geometry_mode not in {"evolving_auxiliary", "fixed_physical_geodesic"}:
            raise ValueError(f"unknown LorentzNet geometry mode: {geometry_mode!r}")
        self.geometry_mode = geometry_mode
        self.coordinate_scale = float(coordinate_scale)
        edge_dim = 3 if geometry_mode == "fixed_physical_geodesic" else 2
        self.scalar_norm = nn.LayerNorm(width)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * width + edge_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
        )
        if geometry_mode == "evolving_auxiliary":
            coordinate_final = nn.Linear(width, 1)
            _small_normal(coordinate_final)
            self.coordinate_mlp = nn.Sequential(
                nn.Linear(width, width), nn.SiLU(), coordinate_final
            )

    def forward(
        self,
        h: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        fixed_edge: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_nodes, _ = h.shape
        real = mask.bool()
        support = (
            real.unsqueeze(2)
            & real.unsqueeze(1)
            & ~torch.eye(n_nodes, device=h.device, dtype=torch.bool).unsqueeze(0)
        )
        support_f = support.unsqueeze(-1).to(h.dtype)
        sqrt_degree = support.sum(dim=2).clamp_min(1).to(h.dtype).sqrt().unsqueeze(-1)

        hn = self.scalar_norm(h)
        hi = hn.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        hj = hn.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        if self.geometry_mode == "fixed_physical_geodesic":
            if fixed_edge is None:
                raise ValueError("fixed physical geometry requires precomputed edge features")
            edge = fixed_edge
        else:
            diff_sq, dot = _pair_invariants(y)
            edge = torch.stack((diff_sq, dot), dim=-1).to(h.dtype)
        message = self.message_mlp(torch.cat((hi, hj, edge), dim=-1))
        # Signed sum: message components retain their signs.  There is deliberately
        # no softmax, absolute value, positivity constraint, or sigmoid gate.
        message = message * support_f

        aggregate = message.sum(dim=2) / sqrt_degree
        h = (h + self.node_mlp(torch.cat((hn, aggregate), dim=-1)))
        h = h * mask.unsqueeze(-1).to(h.dtype)

        if self.geometry_mode == "evolving_auxiliary":
            # The released LorentzNet uses y_i-y_j for this update.  A small learned
            # coefficient and fixed residual scale replace its non-equivariant emergency
            # component clamp.
            displacement = y.unsqueeze(2) - y.unsqueeze(1)
            coefficient = self.coordinate_mlp(message).to(y.dtype) * support.unsqueeze(-1)
            delta = (coefficient * displacement).sum(dim=2) / sqrt_degree.to(y.dtype)
            y = (y + self.coordinate_scale * delta) * mask.unsqueeze(-1).to(y.dtype)
        return h, y


class LorentzNetBackbone(nn.Module):
    """Shared invariant-scalar/equivariant-vector backbone and raw field head."""

    def __init__(
        self,
        condition_dim: int,
        width: int = 96,
        num_layers: int = 6,
        reference_mode: str = "none",
        scalar_init_mode: str = "normsq",
        particle_readout_mode: str = "ambient",
        geometry_mode: str = "evolving_auxiliary",
        field_degree_normalization: str = "none",
        regulator_mass: float = 0.1,
    ):
        super().__init__()
        if reference_mode not in {
            "none", "plain_readout", "normalized_tangent_readout"
        }:
            raise ValueError(f"unknown LorentzNet reference mode: {reference_mode!r}")
        if scalar_init_mode not in {"normsq", "reference_contractions"}:
            raise ValueError(f"unknown LorentzNet scalar init mode: {scalar_init_mode!r}")
        if particle_readout_mode not in {"ambient", "normalized_logmap"}:
            raise ValueError(
                f"unknown LorentzNet particle readout mode: {particle_readout_mode!r}"
            )
        if geometry_mode not in {"evolving_auxiliary", "fixed_physical_geodesic"}:
            raise ValueError(f"unknown LorentzNet geometry mode: {geometry_mode!r}")
        if field_degree_normalization not in {"none", "sqrt"}:
            raise ValueError(
                "unknown LorentzNet field degree normalization: "
                f"{field_degree_normalization!r}"
            )
        if field_degree_normalization == "sqrt" and particle_readout_mode != "normalized_logmap":
            raise ValueError("sqrt degree normalization requires normalized_logmap readout")
        self.reference_mode = reference_mode
        self.scalar_init_mode = scalar_init_mode
        self.particle_readout_mode = particle_readout_mode
        self.geometry_mode = geometry_mode
        self.field_degree_normalization = field_degree_normalization
        self.regulator_mass = float(regulator_mass)
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.time_embed = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.node_seed = nn.Sequential(
            nn.Linear(2 if scalar_init_mode == "reference_contractions" else 1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList(
            [LorentzNetLGEB(width, geometry_mode=geometry_mode) for _ in range(num_layers)]
        )

        edge_dim = 3 if geometry_mode == "fixed_physical_geodesic" else 2
        self.field_mlp = nn.Sequential(
            nn.Linear(2 * width + edge_dim, width), nn.SiLU(), nn.Linear(width, 1)
        )
        _small_normal(self.field_mlp[-1])
        if reference_mode != "none":
            # Ordered outputs are (e_t, jet_p4); role embeddings/tokens are unnecessary.
            self.reference_mlp = nn.Sequential(
                nn.Linear(width + 2, width), nn.SiLU(), nn.Linear(width, 2)
            )
            _small_normal(self.reference_mlp[-1])

    @property
    def needs_references(self) -> bool:
        return (
            self.reference_mode != "none"
            or self.scalar_init_mode == "reference_contractions"
        )

    def initial_scalar_state(
        self,
        y: torch.Tensor,
        t: torch.Tensor,
        conditions: torch.Tensor,
        mask: torch.Tensor,
        references: torch.Tensor | None,
    ) -> torch.Tensor:
        """Construct h^0 using the selected invariant particle seed."""
        dtype = next(self.parameters()).dtype
        cond_embedding = self.condition_embed(conditions.to(dtype))
        time_features = torch.stack(
            (t, torch.sin(torch.pi * t), torch.cos(torch.pi * t)), dim=-1
        )
        time_embedding = self.time_embed(time_features.to(dtype))
        if self.scalar_init_mode == "reference_contractions":
            if references is None or references.shape[1] != 2:
                raise ValueError(
                    "reference_contractions requires ordered (e_t, jet_p4) references"
                )
            seed = signed_log(
                dotsq4(y.unsqueeze(2), references.to(dtype).unsqueeze(1))
            )
            h = (
                self.node_seed(seed)
                - time_embedding.unsqueeze(1)
                + cond_embedding.unsqueeze(1)
            )
        else:
            seed = signed_log(normsq4(y)).unsqueeze(-1)
            h = (
                self.node_seed(seed)
                + time_embedding.unsqueeze(1)
                + cond_embedding.unsqueeze(1)
            )
        return h * mask.unsqueeze(-1).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: torch.Tensor,
        mask: torch.Tensor,
        references: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.needs_references:
            if references is None or references.shape[1] != 2:
                raise ValueError(
                    "configured LorentzNet requires ordered (e_t, jet_p4) references"
                )
        elif references is not None:
            raise ValueError("reference vectors were supplied but no model path uses them")

        dtype = next(self.parameters()).dtype
        x64 = x.to(torch.float64) * mask.unsqueeze(-1).to(torch.float64)
        y = x.to(dtype) * mask.unsqueeze(-1).to(dtype)
        h = self.initial_scalar_state(y, t, conditions, mask, references)

        fixed_edge = None
        physical_distance = None
        physical_direction = None
        if (self.geometry_mode == "fixed_physical_geodesic"
                or self.particle_readout_mode == "normalized_logmap"):
            physical_edge, physical_distance, physical_direction = (
                _physical_geodesic_geometry(x64, self.regulator_mass, dtype)
            )
            if self.geometry_mode == "fixed_physical_geodesic":
                fixed_edge = physical_edge

        for block in self.blocks:
            h, y = block(h, y, mask, fixed_edge=fixed_edge)

        n_nodes = y.shape[1]
        hi = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        hj = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        if self.geometry_mode == "fixed_physical_geodesic":
            edge = fixed_edge
        else:
            diff_sq, dot = _pair_invariants(y)
            edge = torch.stack((diff_sq, dot), dim=-1).to(dtype)
        coefficients = self.field_mlp(torch.cat((hi, hj, edge), dim=-1)).squeeze(-1)
        if self.particle_readout_mode == "normalized_logmap":
            support = (
                mask.bool().unsqueeze(2)
                & mask.bool().unsqueeze(1)
                & ~torch.eye(n_nodes, device=x.device, dtype=torch.bool).unsqueeze(0)
            )
            raw = torch.einsum(
                "bij,bijf->bif",
                coefficients.to(torch.float64) * support,
                physical_direction,
            )
            if self.field_degree_normalization == "sqrt":
                sqrt_degree = support.sum(dim=2).clamp_min(1).to(torch.float64).sqrt()
                raw = raw / sqrt_degree.unsqueeze(-1)
        else:
            support = (
                mask.bool().unsqueeze(2)
                & mask.bool().unsqueeze(1)
                & ~torch.eye(n_nodes, device=x.device, dtype=torch.bool).unsqueeze(0)
            )
            raw = torch.einsum("bij,bjf->bif", coefficients * support, y)

        if self.reference_mode != "none":
            refs = references.to(dtype)
            reference_invariants = signed_log(
                dotsq4(y.unsqueeze(2), refs.unsqueeze(1))
            )
            reference_coefficients = self.reference_mlp(
                torch.cat((h, reference_invariants), dim=-1)
            )
            if self.reference_mode == "normalized_tangent_readout":
                refs64 = references.to(torch.float64)
                expanded_refs = refs64.unsqueeze(1).expand(-1, n_nodes, -1, -1)
                projected = pushforward_to_tangent(
                    x64.unsqueeze(2), expanded_refs, self.regulator_mass
                )
                projected_norm = torch.sqrt((-normsq4(projected)).clamp_min(0))
                normalized_refs = projected / (
                    self.regulator_mass + projected_norm
                ).unsqueeze(-1)
                reference_field = torch.einsum(
                    "bir,birf->bif",
                    reference_coefficients.to(torch.float64),
                    normalized_refs,
                )
            else:
                reference_field = torch.einsum(
                    "bir,brf->bif", reference_coefficients, refs
                )
            raw = raw + reference_field.to(raw.dtype)

        return raw * mask.unsqueeze(-1).to(dtype)


class LorentzNetFlow(nn.Module):
    """Shared LorentzNet raw field with Euclidean or mass-shell FM adaptation.

    The Euclidean objective uses an ordinary component-wise MSE.  That objective is
    frame dependent even though this neural vector field is Lorentz equivariant.
    """

    def __init__(
        self,
        condition_dim: int,
        n_particle_types: int,
        width: int = 96,
        num_layers: int = 6,
        regulator_mass: float = 0.1,
        flow_geometry: str = "euclidean",
        reference_mode: str = "none",
        scalar_init_mode: str = "normsq",
        particle_readout_mode: str = "ambient",
        geometry_mode: str = "evolving_auxiliary",
        field_degree_normalization: str = "none",
    ):
        super().__init__()
        if flow_geometry not in {"euclidean", "mass_shell"}:
            raise ValueError(f"unknown flow geometry: {flow_geometry!r}")
        self.cond_dim = condition_dim
        self.n_particles_idx = n_particle_types
        self.regulator_mass = float(regulator_mass)
        self.flow_geometry = flow_geometry
        self.reference_mode = reference_mode
        self.scalar_init_mode = scalar_init_mode
        self.particle_readout_mode = particle_readout_mode
        self.geometry_mode = geometry_mode
        self.field_degree_normalization = field_degree_normalization
        self.backbone_name = "lorentznet"
        self.null_cond = nn.Parameter(torch.zeros(condition_dim))
        self.lorentznet_backbone = LorentzNetBackbone(
            condition_dim,
            width,
            num_layers,
            reference_mode,
            scalar_init_mode,
            particle_readout_mode,
            geometry_mode,
            field_degree_normalization,
            regulator_mass,
        )

    def make_null_cond(self, conditions: torch.Tensor) -> torch.Tensor:
        null = self.null_cond.unsqueeze(0).expand(conditions.shape[0], -1)
        keep = torch.zeros(conditions.shape[-1], device=conditions.device)
        keep[self.n_particles_idx] = 1.0
        return null * (1 - keep) + conditions * keep

    def raw_field(self, x, t, jet_conditions, mask, ref_vectors=None):
        if jet_conditions.shape[-1] != self.cond_dim:
            raise ValueError(
                f"LorentzNetFlow expected {self.cond_dim} conditions, "
                f"got {jet_conditions.shape[-1]}"
            )
        return self.lorentznet_backbone(x, t, jet_conditions, mask, ref_vectors)

    def forward(self, x, t, jet_conditions, mask, ref_vectors=None):
        raw = self.raw_field(x, t, jet_conditions, mask, ref_vectors)
        if self.flow_geometry == "mass_shell":
            raw = pushforward_to_tangent(
                x.to(torch.float64), raw.to(torch.float64), self.regulator_mass
            )
        return raw * mask.unsqueeze(-1).to(raw.dtype)

    # Reuse the existing tested geodesic Euler machinery without inheriting or
    # duplicating the historical mass-shell backbone.
    def _mass_shell_velocity(self, state, conditions, mask, t, use_cfg,
                             guidance_weight, references):
        if self.flow_geometry != "mass_shell":
            raise ValueError("mass-shell stepping requires flow_geometry='mass_shell'")
        batch_t = t.unsqueeze(0).expand(state.shape[0])
        model_dtype = next(self.parameters()).dtype
        refs = references.to(model_dtype) if references is not None else None
        velocity = self.forward(
            state, batch_t.to(model_dtype), conditions.to(model_dtype), mask, refs
        )
        if use_cfg:
            unconditional = self.forward(
                state, batch_t.to(model_dtype), self.make_null_cond(conditions).to(model_dtype),
                mask, refs,
            )
            velocity = velocity + guidance_weight * (velocity - unconditional)
        # Conditional and unconditional outputs have each received the one final
        # projection in forward(); their linear CFG combination remains tangent.
        return velocity * mask.unsqueeze(-1)

    # Keep the sampler contract identical to MassShellGNNFlow.
    from models.mass_shell_gnn import MassShellGNNFlow as _MassShellSampler
    step_hyperbolic = _MassShellSampler.step_hyperbolic


def build_lorentznet(
    max_num_jet_types: int,
    *,
    num_layers: int = 6,
    hidden_dim: int = 96,
    include_pt: bool = True,
    include_mass_condition: bool = True,
    regulator_mass: float = 0.1,
    flow_geometry: str = "euclidean",
    reference_mode: str = "none",
    scalar_init_mode: str = "normsq",
    particle_readout_mode: str = "ambient",
    geometry_mode: str = "evolving_auxiliary",
    field_degree_normalization: str = "none",
) -> LorentzNetFlow:
    condition_dim = (
        max_num_jet_types + 1 + int(include_pt) + int(include_mass_condition)
    )
    return LorentzNetFlow(
        condition_dim=condition_dim,
        n_particle_types=max_num_jet_types,
        width=hidden_dim,
        num_layers=num_layers,
        regulator_mass=regulator_mass,
        flow_geometry=flow_geometry,
        reference_mode=reference_mode,
        scalar_init_mode=scalar_init_mode,
        particle_readout_mode=particle_readout_mode,
        geometry_mode=geometry_mode,
        field_degree_normalization=field_degree_normalization,
    )


__all__ = [
    "LorentzNetBackbone", "LorentzNetFlow", "LorentzNetLGEB",
    "build_lorentznet", "signed_log",
]
