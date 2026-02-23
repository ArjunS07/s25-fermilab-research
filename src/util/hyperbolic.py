"""
util/hyperbolic.py — Poincaré Ball operations for Riemannian Flow Matching.

Implements the geometric operators from Table 5 of:
    Chen & Lipman, "Riemannian Flow Matching on General Geometries" (ICLR 2024)
    https://arxiv.org/pdf/2302.03660

All operations support arbitrary batch dimensions (..., d).

Mapping note
-----------
JetNet particles use massless kinematics (E = |p|), so the on-shell manifold is
the light cone rather than H^3. We instead use the radial tanh bijection

    phi(x)      = tanh(||x||) * x / ||x||       R^d -> B^d (open unit ball)
    phi_inv(y)  = atanh(||y||) * y / ||y||       B^d -> R^d

to embed normalized Cartesian 4-momenta into the Poincaré ball. This preserves
directions and compresses the exponential energy distribution via tanh, giving
the geodesic structure of hyperbolic space without requiring a fixed particle mass.
"""

import torch

# Numerical stability floor
_EPS = 1e-7


# ---------------------------------------------------------------------------
# Ball utilities
# ---------------------------------------------------------------------------

def clamp_to_ball(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Clamp x to strictly inside the Poincaré ball: ||x|| < 1/sqrt(c) - eps."""
    max_norm = (1.0 - eps) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    scale = (max_norm / norm).clamp(max=1.0)
    return x * scale


def _lambda_x(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Conformal factor lambda_x = 2 / (1 - c||x||^2). Shape: (..., 1)."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 - c * x2).clamp(min=_EPS)


# ---------------------------------------------------------------------------
# Core operators (Table 5, Chen & Lipman 2024)
# ---------------------------------------------------------------------------

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Möbius addition  x ⊕_c y  in the Poincaré ball with curvature c.

        (1 + 2c<x,y> + c||y||²) x  +  (1 - c||x||²) y
        -----------------------------------------------
            1 + 2c<x,y> + c²||x||²||y||²

    x, y : (..., d)
    """
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    denom = (1.0 + 2.0 * c * xy + c ** 2 * x2 * y2).clamp(min=_EPS)
    return num / denom


def log_map(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Logarithm map  log_x^c(y) : tangent vector at x pointing toward y.

        log_x(y) = (2 / (sqrt(c) * lambda_x)) * atanh(sqrt(c) * ||-x ⊕ y||)
                   * (-x ⊕ y) / ||-x ⊕ y||

    x, y : (..., d),  x strictly inside the ball.
    Returns (..., d) tangent vector at x.
    """
    minus_x_y = mobius_add(-x, y, c)                               # -x ⊕_c y
    norm = minus_x_y.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    lam = _lambda_x(x, c)                                           # (..., 1)
    # atanh argument must be in (-1, 1)
    atanh_arg = ((c ** 0.5) * norm).clamp(max=1.0 - _EPS)
    scale = (2.0 / ((c ** 0.5) * lam)) * torch.atanh(atanh_arg)
    return scale * (minus_x_y / norm)


def exp_map(x: torch.Tensor, u: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map  exp_x^c(u) : move from x in tangent direction u.

        exp_x(u) = x ⊕ (tanh(sqrt(c) * lambda_x * ||u|| / 2) * u / (sqrt(c) * ||u||))

    x : (..., d)  point on the ball
    u : (..., d)  tangent vector at x
    Returns (..., d) new point on the ball.
    """
    u_norm = u.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    lam = _lambda_x(x, c)
    # Clamp tanh arg to avoid overflow
    tanh_arg = ((c ** 0.5) * lam * u_norm / 2.0).clamp(max=15.0)
    direction = torch.tanh(tanh_arg) * u / ((c ** 0.5) * u_norm)
    return mobius_add(x, direction, c)


# ---------------------------------------------------------------------------
# Riemannian Flow Matching: path and conditional vector field
# ---------------------------------------------------------------------------

def geodesic_interpolant(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    c: float = 1.0,
) -> torch.Tensor:
    """
    Compute the conditional path from Equation 15 (Chen & Lipman 2024):

        x_t = exp_{x_1}( kappa(t) * log_{x_1}(x_0) )   with kappa(t) = 1 - t

    x_0, x_1 : (batch, ..., d)
    t         : (batch,) in [0, 1]
    Returns     (batch, ..., d)
    """
    # Reshape t for broadcasting against (..., d)
    shape = (-1,) + (1,) * (x_0.dim() - 1)
    kappa = (1.0 - t).view(shape)
    v = log_map(x_1, x_0, c)       # tangent at x_1 toward x_0
    return exp_map(x_1, kappa * v, c)


def conditional_vector_field(
    x_t: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    c: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Conditional vector field from Equation 14 (Chen & Lipman 2024):

        u_t(x_t | x_1) = log_{x_t}(x_1) / (1 - t)

    This is the target the model must learn to predict.

    x_t, x_1 : (batch, ..., d)
    t         : (batch,) in [0, 1)
    Returns     (batch, ..., d) tangent vector at x_t
    """
    shape = (-1,) + (1,) * (x_t.dim() - 1)
    denom = (1.0 - t).view(shape).clamp(min=eps)
    return log_map(x_t, x_1, c) / denom


def hyperbolic_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    x_t: torch.Tensor,
    mask: torch.Tensor,
    c: float = 1.0,
) -> torch.Tensor:
    """
    Riemannian MSE loss from Equation 14 (Chen & Lipman 2024):

        L = E[ ||v_theta - u_t||^2_g ]

    where  ||u||^2_g = lambda_x^2 * ||u||^2_E  and  lambda_x = 2 / (1 - c||x||^2).

    pred, target, x_t : (batch, max_particles, d)
    mask              : (batch, max_particles)  1 = real, 0 = padding
    Returns scalar loss.
    """
    lam = _lambda_x(x_t, c).squeeze(-1)                       # (batch, max_particles)
    diff = pred - target                                       # (batch, max_particles, d)
    # Riemannian squared norm per particle
    riem_sq = (lam ** 2) * (diff * diff).sum(dim=-1)          # (batch, max_particles)
    n_real = mask.sum().clamp(min=1)
    return (riem_sq * mask).sum() / n_real


# ---------------------------------------------------------------------------
# Pushforward / pullback of the radial tanh bijection
# ---------------------------------------------------------------------------

def pushforward(x: torch.Tensor, u: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Pushforward  (d phi_x)(u)  of the radial tanh bijection phi: R^d -> B^d_c.

    Decomposing u into radial and tangential parts:

        d phi_x(u) = (tanh(r)/r) * u_perp  +  sech²(r) * u_parallel

    where r = sqrt(c) * ||x||.

    Use this to map the model's Cartesian output velocity into the Poincaré ball
    tangent space so that the loss can be computed with the Riemannian metric.

    x : (..., d)  Cartesian point (pre-image of the ball point)
    u : (..., d)  Cartesian tangent vector at x
    Returns (..., d) tangent vector at phi(x) in the ball.
    """
    norm = x.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    r = (c ** 0.5) * norm                                     # (..., 1)
    x_hat = x / norm                                           # unit direction

    u_parallel = (u * x_hat).sum(dim=-1, keepdim=True) * x_hat
    u_perp = u - u_parallel

    g = torch.tanh(r.clamp(max=15.0)) / r                     # tanh(r)/r
    sech2 = 1.0 / torch.cosh(r.clamp(max=15.0)).clamp(min=_EPS) ** 2

    return g * u_perp + sech2 * u_parallel


# ---------------------------------------------------------------------------
# Bijection: R^d <-> Poincaré ball
# ---------------------------------------------------------------------------

def to_poincare_ball(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Radial tanh bijection  R^d -> B^d_c:

        phi(x) = tanh(sqrt(c) * ||x||) * x / (sqrt(c) * ||x||)

    Preserves direction; compresses exponential energy distributions via tanh.
    """
    norm = x.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    tanh_arg = ((c ** 0.5) * norm).clamp(max=15.0)
    return torch.tanh(tanh_arg) * x / ((c ** 0.5) * norm)


def from_poincare_ball(y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Inverse radial tanh bijection  B^d_c -> R^d:

        phi_inv(y) = atanh(sqrt(c) * ||y||) * y / (sqrt(c) * ||y||)
    """
    norm = y.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    atanh_arg = ((c ** 0.5) * norm).clamp(max=1.0 - _EPS)
    return torch.atanh(atanh_arg) * y / ((c ** 0.5) * norm)
