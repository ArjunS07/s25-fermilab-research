"""
util/mass_shell.py — Riemannian Flow Matching on the mass shell (hyperboloid model).

An alternative geometry to the Poincaré-ball embedding in ``util/hyperbolic.py``. Instead of
the mass-agnostic radial-tanh bijection, particles live *directly* on the mass shell

    H_m = { p in R^{1,3} : <p,p>_Mink = m^2,  p^0 > 0 }

the upper sheet of a two-sheeted hyperboloid, which is a model of hyperbolic 3-space with
"radius" m (curvature -1/m^2). JetNet particles are massless (E = |p|), i.e. on the light
cone; a small regulator mass m lifts them onto the shell (``project_to_shell``). Masked
particles are parked at the apex (m, 0, 0, 0).

Geometry (signature (+,-,-,-), via ``util/minkowski_utils.py``):
    tangent space   T_pH = { u : <p,u> = 0 }   (tangent vectors are spacelike, <u,u> < 0)
    tangent norm    ||u|| = sqrt(-<u,u>)         (induced Riemannian metric g = -<.,.>)
    distance        d(p,q) = m * arccosh(<p,q> / m^2)
    exp_p(u)        = cosh(||u||/m) p + m sinh(||u||/m) u/||u||
    log_p(q)        = d(p,q) * w/||w||,  w = q - (<p,q>/m^2) p   (the T_pH component of q)

Flow-matching path/field mirror Chen & Lipman, "Riemannian Flow Matching on General
Geometries" (ICLR 2024), Eqs. 14-15, exactly as the Poincaré version does:
    interpolant     x_t = exp_{x_1}( (1-t) * log_{x_1}(x_0) )
    conditional VF  u_t(x_t | x_1) = log_{x_t}(x_1) / (1 - t)

All geometric operations return float64.  The neural network may consume a float32 view of
the shell state, but projection, interpolation, exp/log maps, and the loss must never be
silently rounded back to the model dtype (especially for small regulator masses).
"""

import torch

from util.geometry.minkowski_utils import normsq4, dotsq4

# Numerical stability floor (in float64 space).
_EPS = 1e-12
# Cap on the geodesic step argument s = ||u||/m so cosh/sinh cannot overflow float64 (which
# happens for a near-lightlike shell, small m, under aggressive velocities such as CFG). ||u||
# is Lorentz-invariant, so clamping s preserves covariance; a genuine trained field never
# saturates this — it only prevents a hard NaN from a pathological step.
_S_MAX = 30.0


class MassShellIntegrationError(FloatingPointError):
    """Structured failure raised by adaptive mass-shell integration."""

    def __init__(self, reason: str, message: str, **diagnostics):
        super().__init__(message)
        self.reason = reason
        self.diagnostics = diagnostics

    def as_dict(self) -> dict:
        return {
            "reason": self.reason,
            "message": str(self),
            **self.diagnostics,
        }


def _to_f64(*tensors):
    return tuple(t.to(torch.float64) for t in tensors)


def project_to_shell(p: torch.Tensor, m: float) -> torch.Tensor:
    """Lift 4-vectors onto the mass shell by fixing the energy: p^0 <- sqrt(|p_vec|^2 + m^2).

    Massless (light-cone) data and zero-padding both map onto H_m; zero spatial momentum
    (padding) lands exactly at the apex (m, 0, 0, 0). Spatial momentum is preserved.

    p : (..., 4) 4-vectors (E, px, py, pz). Returns float64 points on the shell.
    """
    (p64,) = _to_f64(p)
    p3 = p64[..., 1:4]
    e = torch.sqrt((p3 * p3).sum(dim=-1, keepdim=True) + m * m)
    out = torch.cat([e, p3], dim=-1)
    return out


def massless_energy_view(p: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Diagnostic view with identical spatial momentum and E=|p|.

    This does not change the canonical shell sample; it isolates observables that depend on
    the regulator energy from JetNet observables built only from eta/phi/pT.
    """
    (p64,) = _to_f64(p)
    energy = torch.linalg.vector_norm(p64[..., 1:4], dim=-1, keepdim=True)
    out = torch.cat([energy, p64[..., 1:4]], dim=-1)
    if mask is not None:
        out = out * mask.to(torch.float64).unsqueeze(-1)
    return out


def geodesic_distance(p: torch.Tensor, q: torch.Tensor, m: float) -> torch.Tensor:
    """Geodesic distance d(p,q) = m * arccosh(<p,q>/m^2) on the mass shell.

    p, q : (..., 4) points on H_m. Returns (...) distances, input dtype.
    """
    p64, q64 = _to_f64(p, q)
    arg = dotsq4(p64, q64) / (m * m)
    arg = torch.clamp(arg, min=1.0 + _EPS)          # arccosh domain; equal points -> 0
    return m * torch.acosh(arg)


def _tangent_norm(u64: torch.Tensor) -> torch.Tensor:
    """||u|| = sqrt(-<u,u>) for a (float64) tangent vector, shape (..., 1).

    Returns the *true* norm (0 for a zero vector). Callers clamp the denominator where they
    divide by it — keeping the norm exact matters because it multiplies sinh(s), which can be
    huge when s is clamped; a nonzero floor here would leak macroscopically off the shell. The
    geometry ops are used to build data-side targets/inputs and are never back-propagated
    through, so sqrt(0) needs no gradient smoothing."""
    nsq = (-normsq4(u64)).clamp(min=0.0)
    return torch.sqrt(nsq).unsqueeze(-1)


def tangent_norm(u: torch.Tensor) -> torch.Tensor:
    """Public float64 induced norm for tangent vectors, with shape ``(..., 1)``."""
    return _tangent_norm(u.to(torch.float64))


def exp_map(p: torch.Tensor, u: torch.Tensor, m: float) -> torch.Tensor:
    """Exponential map exp_p(u): travel geodesic arc length ||u|| from p in direction u.

    p : (..., 4) on H_m.  u : (..., 4) tangent at p (<p,u> = 0). Returns (..., 4) on H_m.
    """
    p64, u64 = _to_f64(p, u)
    un = _tangent_norm(u64)                          # (..., 1), true norm (0 for zero vector)
    s = (un / m).clamp(max=_S_MAX)                    # invariant clamp: no cosh/sinh overflow
    direction = u64 / un.clamp(min=_EPS)             # unit tangent (0 when u is 0)
    out = torch.cosh(s) * p64 + m * torch.sinh(s) * direction
    return out


def log_map(p: torch.Tensor, q: torch.Tensor, m: float,
            return_distance: bool = False):
    """Logarithm map log_p(q): the tangent vector at p pointing toward q with norm d(p,q).

    p, q : (..., 4) on H_m. Returns (..., 4) tangent at p, input dtype. With
    ``return_distance=True`` also returns the geodesic distance ``d`` (..., 1) that
    the map already computes, so callers need not recompute ``sqrt(-<rel,rel>)``.
    """
    p64, q64 = _to_f64(p, q)
    pq = dotsq4(p64, q64).unsqueeze(-1)              # (..., 1)
    w = q64 - (pq / (m * m)) * p64                   # component of q in T_pH
    w_norm = _tangent_norm(w)                        # (..., 1), true norm
    arg = torch.clamp(pq / (m * m), min=1.0 + _EPS)
    d = m * torch.acosh(arg)                         # (..., 1)
    out = d * (w / w_norm.clamp(min=_EPS))           # 0 when p == q (d -> 0, w -> 0)
    if return_distance:
        return out, d
    return out


def pushforward_to_tangent(p: torch.Tensor, v_cartesian: torch.Tensor, m: float) -> torch.Tensor:
    """Project a Cartesian velocity onto T_pH: u = v - (<p,v>/m^2) p  (Minkowski-orthogonal).

    The model predicts an unconstrained Cartesian velocity; this makes it a valid tangent
    vector at p before stepping/comparison. p : (..., 4) on H_m; v_cartesian : (..., 4).
    """
    p64, v64 = _to_f64(p, v_cartesian)
    pv = dotsq4(p64, v64).unsqueeze(-1)
    out = v64 - (pv / (m * m)) * p64
    return out


def parallel_transport(p: torch.Tensor, q: torch.Tensor, v: torch.Tensor,
                       m: float) -> torch.Tensor:
    """Parallel-transport ``v`` from ``T_p H_m`` to ``T_q H_m``.

    The closed-form transport along the unique geodesic is

        PT_{p->q}(v) = v - <q,v> / (m^2 + <p,q>) * (p + q).

    Inputs broadcast over arbitrary leading dimensions. All arithmetic and the returned
    tangent vector are float64, matching the other mass-shell geometry primitives.
    """
    p64, q64, v64 = _to_f64(p, q, v)
    denom = (m * m + dotsq4(p64, q64)).unsqueeze(-1).clamp(min=_EPS)
    qv = dotsq4(q64, v64).unsqueeze(-1)
    return v64 - (qv / denom) * (p64 + q64)


def geodesic_interpolant(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor,
                         m: float) -> torch.Tensor:
    """Conditional path x_t = exp_{x_1}((1-t) log_{x_1}(x_0)) (Chen & Lipman 2024, Eq. 15).

    x_0, x_1 : (batch, ..., 4) on H_m.  t : (batch,) in [0, 1]. Returns (batch, ..., 4).
    """
    shape = (-1,) + (1,) * (x_0.dim() - 1)
    kappa = (1.0 - t).view(shape)
    v = log_map(x_1, x_0, m)                          # tangent at x_1 toward x_0
    return exp_map(x_1, kappa * v, m)


def conditional_vector_field(x_t: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor,
                             m: float, eps: float = 1e-5) -> torch.Tensor:
    """Target field u_t(x_t | x_1) = log_{x_t}(x_1) / (1 - t) (Chen & Lipman 2024, Eq. 14).

    x_t, x_1 : (batch, ..., 4) on H_m.  t : (batch,) in [0, 1). Returns tangent at x_t.
    """
    shape = (-1,) + (1,) * (x_t.dim() - 1)
    denom = (1.0 - t).view(shape).clamp(min=eps)
    return log_map(x_t, x_1, m) / denom


def geodesic_cost_matrix(x_0_real: torch.Tensor, x_1_real: torch.Tensor,
                         m: float) -> torch.Tensor:
    """Pairwise geodesic-distance cost for ICP assignment: cost[i, j] = d(shell(x0_i), shell(x1_j)).

    Both clouds are lifted onto H_m first. Only the real particles passed in participate, so a
    real particle can never be matched to apex-parked padding (the plan's masking guard).

    x_0_real, x_1_real : (n, 4) 4-vectors (normalised space). Returns (n, n) cost, input dtype.
    """
    p0 = project_to_shell(x_0_real, m)
    p1 = project_to_shell(x_1_real, m)
    return geodesic_distance(p0.unsqueeze(1), p1.unsqueeze(0), m)


def mass_shell_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                    m: float = 1.0, negative_rtol: float = 1e-10) -> torch.Tensor:
    """Masked Riemannian MSE on the shell: mean over real particles of ||pred - target||_g^2,
    with the induced tangent metric ||u||_g^2 = -<u,u>_Mink.

    pred, target : (batch, max_particles, 4) tangent vectors at x_t.
    mask         : (batch, max_particles)   1 = real, 0 = padding. Returns scalar.
    """
    pred64, target64, mask64 = _to_f64(pred, target, mask)
    diff = pred64 - target64
    raw_sq = -normsq4(diff)
    # A difference of two tangent vectors is tangent and has non-negative induced norm.
    # Permit only cancellation-level negatives, scaled by the Euclidean magnitude.
    scale = diff.square().sum(dim=-1).clamp(min=1.0)
    materially_invalid = (raw_sq < -(negative_rtol * scale)) & (mask64 > 0)
    if materially_invalid.any():
        worst = float((raw_sq / scale)[materially_invalid].min().detach().cpu())
        raise FloatingPointError(
            "mass-shell loss received a materially non-tangent error vector: "
            f"min induced/Euclidean squared norm={worst:.3e}"
        )
    sq = raw_sq.clamp(min=0.0)
    n_real = mask64.sum().clamp(min=1.0)
    return (sq * mask64).sum() / n_real


def tangent_error_diagnostics(p: torch.Tensor, pred: torch.Tensor, target: torch.Tensor,
                              mask: torch.Tensor, m: float, dt: float = 1.0) -> dict:
    """Summarize numerical health of tangent prediction and one geodesic Euler step."""
    p64, pred64, target64, mask64 = _to_f64(p, pred, target, mask)
    real = mask64 > 0
    diff = pred64 - target64
    raw_sq = -normsq4(diff)
    pred_norm = torch.sqrt((-normsq4(pred64)).clamp(min=0.0))
    target_norm = torch.sqrt((-normsq4(target64)).clamp(min=0.0))
    alignment = (-dotsq4(pred64, target64) /
                 (pred_norm * target_norm).clamp(min=_EPS))
    step_rapidity = pred_norm * abs(float(dt)) / float(m)
    p_dot_pred = dotsq4(p64, pred64).abs()
    euclid_scale = torch.linalg.vector_norm(p64, dim=-1) * torch.linalg.vector_norm(pred64, dim=-1)
    tangent_residual = p_dot_pred / euclid_scale.clamp(min=_EPS)

    def fraction(condition):
        return float(condition[real].to(torch.float64).mean().item()) if real.any() else 0.0

    def quantiles(values):
        selected = values[real]
        selected = selected[torch.isfinite(selected)]
        if selected.numel() == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(v) for v in torch.quantile(
            selected, torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float64,
                                   device=selected.device)
        ).cpu()]

    return {
        "n_real": int(real.sum().item()),
        "nonfinite_fraction": fraction(~torch.isfinite(p64).all(dim=-1)
                                       | ~torch.isfinite(pred64).all(dim=-1)
                                       | ~torch.isfinite(target64).all(dim=-1)),
        "raw_loss_negative_fraction": fraction(raw_sq < 0.0),
        "loss_zero_fraction": fraction(raw_sq <= 0.0),
        "tangency_residual_quantiles": quantiles(tangent_residual),
        "pred_tangent_norm_quantiles": quantiles(pred_norm),
        "target_tangent_norm_quantiles": quantiles(target_norm),
        "pred_target_alignment_quantiles": quantiles(alignment),
        "step_rapidity_quantiles": quantiles(step_rapidity),
        "step_rapidity_max": (float(step_rapidity[real & torch.isfinite(step_rapidity)].max())
                               if (real & torch.isfinite(step_rapidity)).any() else 0.0),
        "step_clamp_fraction": fraction(step_rapidity > _S_MAX),
    }
