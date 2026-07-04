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

All operations compute in float64 for geometric stability (per plan revision 9) and return
in the input dtype.
"""

import torch

from util.minkowski_utils import normsq4, dotsq4

# Numerical stability floor (in float64 space).
_EPS = 1e-12


def _to_f64(*tensors):
    return tuple(t.to(torch.float64) for t in tensors)


def project_to_shell(p: torch.Tensor, m: float) -> torch.Tensor:
    """Lift 4-vectors onto the mass shell by fixing the energy: p^0 <- sqrt(|p_vec|^2 + m^2).

    Massless (light-cone) data and zero-padding both map onto H_m; zero spatial momentum
    (padding) lands exactly at the apex (m, 0, 0, 0). Spatial momentum is preserved.

    p : (..., 4) 4-vectors (E, px, py, pz). Returns (..., 4) on the shell, input dtype.
    """
    dtype = p.dtype
    (p64,) = _to_f64(p)
    p3 = p64[..., 1:4]
    e = torch.sqrt((p3 * p3).sum(dim=-1, keepdim=True) + m * m)
    out = torch.cat([e, p3], dim=-1)
    return out.to(dtype)


def geodesic_distance(p: torch.Tensor, q: torch.Tensor, m: float) -> torch.Tensor:
    """Geodesic distance d(p,q) = m * arccosh(<p,q>/m^2) on the mass shell.

    p, q : (..., 4) points on H_m. Returns (...) distances, input dtype.
    """
    dtype = p.dtype
    p64, q64 = _to_f64(p, q)
    arg = dotsq4(p64, q64) / (m * m)
    arg = torch.clamp(arg, min=1.0 + _EPS)          # arccosh domain; equal points -> 0
    return (m * torch.acosh(arg)).to(dtype)


def _tangent_norm(u64: torch.Tensor) -> torch.Tensor:
    """||u|| = sqrt(-<u,u>) for a (float64) tangent vector, shape (..., 1)."""
    nsq = (-normsq4(u64)).clamp(min=0.0)
    return torch.sqrt(nsq + _EPS).unsqueeze(-1)


def exp_map(p: torch.Tensor, u: torch.Tensor, m: float) -> torch.Tensor:
    """Exponential map exp_p(u): travel geodesic arc length ||u|| from p in direction u.

    p : (..., 4) on H_m.  u : (..., 4) tangent at p (<p,u> = 0). Returns (..., 4) on H_m.
    """
    dtype = p.dtype
    p64, u64 = _to_f64(p, u)
    un = _tangent_norm(u64)                          # (..., 1)
    s = un / m
    out = torch.cosh(s) * p64 + m * torch.sinh(s) * (u64 / un)
    return out.to(dtype)


def log_map(p: torch.Tensor, q: torch.Tensor, m: float) -> torch.Tensor:
    """Logarithm map log_p(q): the tangent vector at p pointing toward q with norm d(p,q).

    p, q : (..., 4) on H_m. Returns (..., 4) tangent at p, input dtype.
    """
    dtype = p.dtype
    p64, q64 = _to_f64(p, q)
    pq = dotsq4(p64, q64).unsqueeze(-1)              # (..., 1)
    w = q64 - (pq / (m * m)) * p64                   # component of q in T_pH
    w_norm = _tangent_norm(w)                        # (..., 1)
    arg = torch.clamp(pq / (m * m), min=1.0 + _EPS)
    d = m * torch.acosh(arg)                         # (..., 1)
    out = d * (w / w_norm)
    return out.to(dtype)


def pushforward_to_tangent(p: torch.Tensor, v_cartesian: torch.Tensor, m: float) -> torch.Tensor:
    """Project a Cartesian velocity onto T_pH: u = v - (<p,v>/m^2) p  (Minkowski-orthogonal).

    The model predicts an unconstrained Cartesian velocity; this makes it a valid tangent
    vector at p before stepping/comparison. p : (..., 4) on H_m; v_cartesian : (..., 4).
    """
    dtype = v_cartesian.dtype
    p64, v64 = _to_f64(p, v_cartesian)
    pv = dotsq4(p64, v64).unsqueeze(-1)
    out = v64 - (pv / (m * m)) * p64
    return out.to(dtype)


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
                    m: float = 1.0) -> torch.Tensor:
    """Masked Riemannian MSE on the shell: mean over real particles of ||pred - target||_g^2,
    with the induced tangent metric ||u||_g^2 = -<u,u>_Mink.

    pred, target : (batch, max_particles, 4) tangent vectors at x_t.
    mask         : (batch, max_particles)   1 = real, 0 = padding. Returns scalar.
    """
    dtype = pred.dtype
    pred64, target64, mask64 = _to_f64(pred, target, mask)
    diff = pred64 - target64
    sq = (-normsq4(diff)).clamp(min=0.0)              # (batch, max_particles)
    n_real = mask64.sum().clamp(min=1.0)
    return ((sq * mask64).sum() / n_real).to(dtype)
