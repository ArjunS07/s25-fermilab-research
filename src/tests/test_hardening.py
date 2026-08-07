"""Geometry dtype guard: mass-shell ops keep float64 internally regardless of input dtype."""
import torch


def test_mass_shell_ops_promote_geometry_to_float64():
    from util.geometry.mass_shell import (project_to_shell, exp_map, log_map, pushforward_to_tangent,
                                 geodesic_interpolant, conditional_vector_field, mass_shell_loss)
    m = 0.5
    p = project_to_shell(torch.randn(2, 4, 4), m)
    q = project_to_shell(torch.randn(2, 4, 4), m)
    u = pushforward_to_tangent(p, torch.randn(2, 4, 4) * 0.2, m)
    t = torch.full((2,), 0.3)
    mask = torch.ones(2, 4)
    for out in (p, exp_map(p, u, m), log_map(p, q, m), pushforward_to_tangent(p, q, m),
                geodesic_interpolant(p, q, t, m), conditional_vector_field(p, q, t, m),
                mass_shell_loss(u, u, mask, m)):
        assert out.dtype == torch.float64
