import torch

from util.eval_report import _plot_jet_girth, _plot_jet_mass, _plot_jet_pt_total


def test_bulk_plots_survive_finite_explosive_jet(tmp_path):
    n, particles = 100, 3
    types = torch.zeros(n, dtype=torch.long)
    test_rel = torch.zeros(n, particles, 3, dtype=torch.float64)
    gen_rel = torch.zeros_like(test_rel)
    prior_rel = torch.zeros_like(test_rel)
    for values in (test_rel, gen_rel, prior_rel):
        values[:, 0, 0] = 0.05
        values[:, 0, 2] = 1.0
    gen_rel[-1, 0, 0] = 1e20

    test_abs = torch.zeros(n, particles, 4, dtype=torch.float64)
    gen_abs = torch.zeros_like(test_abs)
    prior_abs = torch.zeros_like(test_abs)
    for values in (test_abs, gen_abs, prior_abs):
        values[:, 0, 2] = 100.0
        values[:, 0, 3] = 100.0
    gen_abs[-1, 0, 2:] = 1e20

    test_cart = torch.zeros(n, particles, 4, dtype=torch.float64)
    gen_cart = torch.zeros_like(test_cart)
    prior_cart = torch.zeros_like(test_cart)
    for values in (test_cart, gen_cart, prior_cart):
        values[:, 0] = torch.tensor([100.0, 90.0, 0.0, 0.0])
    gen_cart[-1, 0] = torch.tensor([2e20, 1e20, 0.0, 0.0])

    _plot_jet_girth(test_rel, gen_rel, types, types, ["g"],
                    str(tmp_path / "girth.png"), prior_rel)
    _plot_jet_pt_total(test_abs, gen_abs, types, types, ["g"], None,
                       str(tmp_path / "pt.png"), prior_abs)
    _plot_jet_mass(test_cart, gen_cart, types, types, ["g"],
                   str(tmp_path / "mass.png"), prior_cart)
    for name in ("girth.png", "pt.png", "mass.png"):
        assert (tmp_path / name).stat().st_size > 0
