"""Current mass-shell RFM model, isolated from the historical LEFTJeN backbone."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.tangent_attention import TangentAttentionBackbone


class MassShellFlow(nn.Module):
    """Typed-reference tangent-attention velocity field on ``H_m``.

    The public parameter names intentionally match existing tangent-attention
    checkpoints (``null_cond`` and ``tangent_backbone.*``).
    """

    def __init__(self, condition_dim: int, n_particle_types: int, width: int,
                 num_layers: int, num_heads: int, vector_channels: int,
                 regulator_mass: float, readout_init: str):
        super().__init__()
        self.cond_dim = condition_dim
        self.n_particles_idx = n_particle_types
        self.regulator_mass = regulator_mass
        self.backbone = "tangent_attention"
        self.null_cond = nn.Parameter(torch.zeros(condition_dim))
        self.tangent_backbone = TangentAttentionBackbone(
            cond_dim=condition_dim,
            width=width,
            num_layers=num_layers,
            num_heads=num_heads,
            vector_channels=vector_channels,
            regulator_mass=regulator_mass,
            readout_init=readout_init,
        )

    def make_null_cond(self, conditions: torch.Tensor) -> torch.Tensor:
        """Replace physical conditions by a learned null, retaining multiplicity."""
        null = self.null_cond.unsqueeze(0).expand(conditions.shape[0], -1)
        keep = torch.zeros(conditions.shape[-1], device=conditions.device)
        keep[self.n_particles_idx] = 1.0
        return null * (1 - keep) + conditions * keep

    def forward(self, x, t, jet_conditions, mask, ref_vectors=None):
        if ref_vectors is None:
            raise ValueError("MassShellFlow requires the two typed reference vectors")
        if jet_conditions.shape[-1] != self.cond_dim:
            raise ValueError(
                f"MassShellFlow expected {self.cond_dim} conditions, "
                f"got {jet_conditions.shape[-1]}"
            )
        return self.tangent_backbone(x, t, jet_conditions, mask, ref_vectors)

    def _mass_shell_velocity(self, state, conditions, mask, t, use_cfg,
                             guidance_weight, references):
        from util.mass_shell import MassShellIntegrationError, pushforward_to_tangent

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

    def step_hyperbolic(self, state, jet_conditions, mask, t_start, t_end,
                        c=1.0, use_cfg=False, guidance_weight=2.0,
                        ref_vectors=None, hyperbolic_model="mass_shell",
                        regulator_mass=None, max_step_rapidity=None,
                        max_substeps=64, return_diagnostics=False):
        """Advance one nominal interval with optional rapidity-controlled substeps."""
        del c
        if hyperbolic_model != "mass_shell":
            raise ValueError("MassShellFlow only supports mass-shell integration")
        if regulator_mass is not None and regulator_mass != self.regulator_mass:
            raise ValueError("sampler regulator mass differs from model regulator mass")

        from util.mass_shell import (MassShellIntegrationError, exp_map,
                                     project_to_shell, tangent_norm)

        current = state
        current_t = float(t_start.detach().cpu())
        end_t = float(t_end.detach().cpu())
        substeps = 0
        max_norm_seen = 0.0
        max_rapidity_seen = 0.0
        while current_t < end_t - 1e-15:
            t = torch.as_tensor(current_t, device=state.device, dtype=t_start.dtype)
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
                raise MassShellIntegrationError(
                    "substep_limit",
                    f"adaptive sampler exceeded {max_substeps} substeps",
                    current_t=current_t, completed_substeps=substeps - 1,
                    max_tangent_norm=max_norm_seen,
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
