"""
Curated qualitative + distributional evaluation figures for LEFTJeN samples.

Single entry point `eval_report(...)` writes 14 figures:
  Pooled (6):        jet_image, jet_image_by_multiplicity, particle_kinematics_rel,
                     particle_kinematics_abs, physicality, isotropy
  Per-class overlay: jet_girth, jet_multiplicity, jet_pt_total, radial_profile,
                     energy_tails
  Per-class grid:    jet_mass, jet_z1_z2, jet_image_by_class

Grids only when side-by-side comparison is intrinsic to the message. Per-class
overlay when 5 clean 1D curves separate cleanly; per-class grid when
test-vs-gen overlay *inside* each class is essential.
"""
from __future__ import annotations

import math
from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import seaborn as sns
import torch
from scipy import stats

from util.minkowski_utils import normsq4
from util.ks import ks_statistic_vs_uniform, ks_two_sample


# ---- Style ----------------------------------------------------------------

plt.rc("mathtext", fontset="cm")
sns.set_style("whitegrid")

_CLASS_PALETTE = sns.color_palette("tab10")
_TEST_COLOR = sns.color_palette("deep")[0]
_GEN_COLOR = sns.color_palette("deep")[3]
_PRIOR_COLOR = sns.color_palette("deep")[2]

# Multiplicity buckets. Boundaries roughly track curriculum idea in MEMORY.md.
_MULT_BUCKETS = [
    ("sparse (n<15)",   0,  15),
    ("medium (15..50)", 15, 50),
    ("dense (n≥50)",    50, 10_000),
]


# ---- Public entry point ---------------------------------------------------

def eval_report(
    *,
    gen_cartesian: torch.Tensor,       # (Ng, P, 4) E, px, py, pz  (padding zeroed)
    test_polar_abs: torch.Tensor,      # (Nt, P, 4) eta, phi, pt, E
    gen_polar_abs: torch.Tensor,       # (Ng, P, 4) eta, phi, pt, E
    test_polar_rel: torch.Tensor,      # (Nt, P, ≥3) eta_rel, phi_rel, pt_rel
    gen_polar_rel: torch.Tensor,       # (Ng, P, ≥3)
    gen_jet_types: torch.Tensor,       # (Ng,) long, class index in [0, n_classes)
    test_jet_types: torch.Tensor,      # (Nt,) long
    jet_type_names: Sequence[str],     # e.g. ["g", "q", "t"]
    gen_pt_cond: Optional[torch.Tensor],  # (Ng,) normalized conditioning pT; None ok
    output_path: str,
    prior_cartesian: Optional[torch.Tensor] = None,
    prior_polar_abs: Optional[torch.Tensor] = None,
    prior_polar_rel: Optional[torch.Tensor] = None,
) -> None:
    """Emit the 14 evaluation figures into `output_path`."""

    gen_c   = gen_cartesian.detach().cpu()
    tp_abs  = test_polar_abs.detach().cpu()
    gp_abs  = gen_polar_abs.detach().cpu()
    tp_rel  = test_polar_rel.detach().cpu()
    gp_rel  = gen_polar_rel.detach().cpu()
    g_types = gen_jet_types.detach().cpu().long()
    t_types = test_jet_types.detach().cpu().long()
    pt_cond = gen_pt_cond.detach().cpu() if gen_pt_cond is not None else None
    prior_c = prior_cartesian.detach().cpu() if prior_cartesian is not None else None
    pp_abs = prior_polar_abs.detach().cpu() if prior_polar_abs is not None else None
    pp_rel = prior_polar_rel.detach().cpu() if prior_polar_rel is not None else None

    names = list(jet_type_names)

    # Pooled
    _plot_jet_image(tp_rel, gp_rel, f"{output_path}/jet_image.png", pp_rel)
    _plot_jet_image_by_multiplicity(tp_rel, gp_rel,
                                    f"{output_path}/jet_image_by_multiplicity.png")
    _plot_particle_kinematics(tp_rel, gp_rel, relative=True,
                              out_path=f"{output_path}/particle_kinematics_rel.png",
                              prior_polar=pp_rel)
    _plot_particle_kinematics(tp_abs, gp_abs, relative=False,
                              out_path=f"{output_path}/particle_kinematics_abs.png",
                              prior_polar=pp_abs)
    _plot_physicality(gen_c, f"{output_path}/physicality.png")
    _plot_isotropy(gen_c, tp_abs, f"{output_path}/isotropy.png")

    # Per-class overlay
    _plot_jet_girth(tp_rel, gp_rel, t_types, g_types, names,
                    f"{output_path}/jet_girth.png", pp_rel)
    _plot_jet_multiplicity(tp_rel, gp_rel, t_types, g_types, names,
                           f"{output_path}/jet_multiplicity.png")
    _plot_jet_pt_total(tp_abs, gp_abs, t_types, g_types, names, pt_cond,
                       f"{output_path}/jet_pt_total.png", pp_abs)
    _plot_radial_profile(tp_rel, gp_rel, t_types, g_types, names,
                         f"{output_path}/radial_profile.png", pp_rel)
    _plot_energy_tails(tp_abs, gp_abs, t_types, g_types, names,
                       f"{output_path}/energy_tails.png", pp_abs)

    # Per-class grid
    test_cart = _polar_abs_to_cartesian(tp_abs)
    _plot_jet_mass(test_cart, gen_c, t_types, g_types, names,
                   f"{output_path}/jet_mass.png", prior_c)
    _plot_jet_z1_z2(tp_rel, gp_rel, t_types, g_types, names,
                    f"{output_path}/jet_z1_z2.png")
    _plot_jet_image_by_class(tp_rel, gp_rel, t_types, g_types, names,
                             f"{output_path}/jet_image_by_class.png")

    print(f"eval_report: 14 figures written to {output_path}/")


# ---- Shared utilities -----------------------------------------------------

def _polar_abs_to_cartesian(polar_abs: torch.Tensor) -> torch.Tensor:
    """(eta, phi, pt, E) -> (E, px, py, pz); padding (pt==0) zeroed."""
    eta = polar_abs[..., 0]
    phi = polar_abs[..., 1]
    pt  = polar_abs[..., 2]
    E   = polar_abs[..., 3]
    real = (pt > 0).float()
    px = pt * torch.cos(phi) * real
    py = pt * torch.sin(phi) * real
    pz = pt * torch.sinh(eta) * real
    return torch.stack([E * real, px, py, pz], dim=-1)


def _jet_pt_scalar(polar_abs: torch.Tensor) -> torch.Tensor:
    return polar_abs[..., 2].clamp(min=0).sum(dim=1)


def _real_mask(pr_or_abs: torch.Tensor) -> torch.Tensor:
    return pr_or_abs[..., 2] > 0


def _pt_weighted_avg_image(
    eta_rel: torch.Tensor,   # (N, P)
    phi_rel: torch.Tensor,   # (N, P)
    pt_rel:  torch.Tensor,   # (N, P) — Σpt_rel per jet ≈ 1
    extent: float = 0.8,
    bins: int = 40,
) -> np.ndarray:
    real = pt_rel > 0
    if not bool(real.any()):
        return np.full((bins, bins), np.nan)
    eta = eta_rel[real].numpy()
    phi = phi_rel[real].numpy()
    wt  = pt_rel[real].numpy()
    # Wrap phi_rel to [-pi, pi] defensively.
    phi = ((phi + np.pi) % (2 * np.pi)) - np.pi
    hist, _, _ = np.histogram2d(
        eta, phi, bins=bins,
        range=[[-extent, extent], [-extent, extent]],
        weights=wt,
    )
    n_jets = eta_rel.shape[0]
    return hist / max(n_jets, 1)


def _draw_jet_image_triptych(
    ax_test, ax_gen, ax_diff,
    test_img: np.ndarray, gen_img: np.ndarray,
    title_prefix: str = "",
    extent: float = 0.8,
) -> None:
    # Jet images span 2-3 orders of magnitude between the pT-weighted core (the
    # leading constituent, always near (0,0)) and the soft halo. On a linear
    # scale the core saturates and the halo vanishes, so use a log color scale
    # with a shared vmax and a floor a few decades down. See docstring.
    vmax = float(np.nanmax(np.stack([test_img, gen_img])) or 1e-12)
    vmin = vmax * 1e-3
    diff = gen_img - test_img
    diff_max = float(np.nanmax(np.abs(diff)) or 1e-12)

    kw = dict(origin="lower", extent=(-extent, extent, -extent, extent),
              aspect="equal")
    img_norm = LogNorm(vmin=vmin, vmax=vmax)
    ax_test.imshow(test_img.T, norm=img_norm, cmap="viridis", **kw)
    ax_gen.imshow(gen_img.T,   norm=img_norm, cmap="viridis", **kw)
    # Signed difference: linear near zero, log-scaled in the tails so both the
    # over- and under-dense regions are visible.
    ax_diff.imshow(diff.T,
                   norm=SymLogNorm(linthresh=diff_max * 1e-2, vmin=-diff_max, vmax=diff_max),
                   cmap="RdBu_r", **kw)

    for ax, sub in zip([ax_test, ax_gen, ax_diff], ["Test", "Gen", "Gen − Test"]):
        ax.set_title(f"{title_prefix}{sub}" if title_prefix else sub, fontsize=12)
        ax.set_xlabel(r"$\Delta\eta$")
        ax.set_ylabel(r"$\Delta\phi$")


def _flatten_real(polar: torch.Tensor, idx: int) -> np.ndarray:
    real = _real_mask(polar)
    return polar[..., idx][real].numpy()


def _class_series(polar: torch.Tensor, types: torch.Tensor, n_classes: int):
    """Yield (class_idx, polar_slice) for jets of each class."""
    for c in range(n_classes):
        sel = (types == c)
        if bool(sel.any()):
            yield c, polar[sel]


# ---- Pooled figures ------------------------------------------------------

def _plot_jet_image(
    test_polar_rel: torch.Tensor,
    gen_polar_rel: torch.Tensor,
    out_path: str,
    prior_polar_rel: Optional[torch.Tensor] = None,
) -> None:
    """Averaged jet image, including the exact pre-transport state when available."""
    t_img = _pt_weighted_avg_image(test_polar_rel[..., 0], test_polar_rel[..., 1],
                                   test_polar_rel[..., 2])
    g_img = _pt_weighted_avg_image(gen_polar_rel[..., 0], gen_polar_rel[..., 1],
                                   gen_polar_rel[..., 2])
    if prior_polar_rel is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(r"Averaged jet image (pT-weighted, $\Delta\eta$–$\Delta\phi$)", fontsize=14)
        _draw_jet_image_triptych(*axes, t_img, g_img)
    else:
        p_img = _pt_weighted_avg_image(prior_polar_rel[..., 0], prior_polar_rel[..., 1],
                                       prior_polar_rel[..., 2])
        vmax = float(np.nanmax(np.stack([t_img, p_img, g_img])) or 1e-12)
        norm = LogNorm(vmin=vmax * 1e-3, vmax=vmax)
        fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
        kw = dict(origin="lower", extent=(-0.8, 0.8, -0.8, 0.8), aspect="equal")
        for ax, img, title in zip(axes[:3], [t_img, p_img, g_img], ["Test", "Prior", "Gen"]):
            ax.imshow(img.T, norm=norm, cmap="viridis", **kw)
            ax.set_title(title)
        for ax, diff, title in [(axes[3], g_img - p_img, "Gen − Prior"),
                                (axes[4], g_img - t_img, "Gen − Test")]:
            lim = float(np.nanmax(np.abs(diff)) or 1e-12)
            ax.imshow(diff.T, norm=SymLogNorm(linthresh=lim * 1e-2, vmin=-lim, vmax=lim),
                      cmap="RdBu_r", **kw)
            ax.set_title(title)
        for ax in axes:
            ax.set_xlabel(r"$\Delta\eta$"); ax.set_ylabel(r"$\Delta\phi$")
        fig.suptitle(r"Averaged jet image: transport from Prior to Gen", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_jet_image_by_multiplicity(
    test_polar_rel: torch.Tensor,
    gen_polar_rel: torch.Tensor,
    out_path: str,
) -> None:
    t_mult = _real_mask(test_polar_rel).sum(dim=1)
    g_mult = _real_mask(gen_polar_rel).sum(dim=1)

    fig, axes = plt.subplots(len(_MULT_BUCKETS), 3, figsize=(15, 4.5 * len(_MULT_BUCKETS)))
    fig.suptitle("Averaged jet image by multiplicity bucket", fontsize=14)
    for row, (label, lo, hi) in enumerate(_MULT_BUCKETS):
        t_sel = (t_mult >= lo) & (t_mult < hi)
        g_sel = (g_mult >= lo) & (g_mult < hi)
        if not (bool(t_sel.any()) and bool(g_sel.any())):
            for ax in axes[row]:
                ax.set_visible(False)
            continue
        t_img = _pt_weighted_avg_image(test_polar_rel[t_sel, :, 0],
                                       test_polar_rel[t_sel, :, 1],
                                       test_polar_rel[t_sel, :, 2])
        g_img = _pt_weighted_avg_image(gen_polar_rel[g_sel, :, 0],
                                       gen_polar_rel[g_sel, :, 1],
                                       gen_polar_rel[g_sel, :, 2])
        _draw_jet_image_triptych(*axes[row], t_img, g_img, title_prefix=f"{label}  ·  ")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_particle_kinematics(
    test_polar: torch.Tensor,
    gen_polar: torch.Tensor,
    relative: bool,
    out_path: str,
    prior_polar: Optional[torch.Tensor] = None,
) -> None:
    if relative:
        titles = [r"$\eta_{\rm rel}$", r"$\phi_{\rm rel}$", r"$p_T^{\rm rel}$"]
        suptitle = "Per-particle kinematics (relative to jet axis)"
    else:
        titles = [r"$\eta$", r"$\phi$", r"$p_T$"]
        suptitle = "Per-particle kinematics (absolute)"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(suptitle, fontsize=14)
    for i, ax in enumerate(axes):
        t = _flatten_real(test_polar, i)
        g = _flatten_real(gen_polar, i)
        p = _flatten_real(prior_polar, i) if prior_polar is not None else None
        bins = 100
        ax.hist(t, bins=bins, alpha=0.5, density=True, color=_TEST_COLOR, label="Test")
        if p is not None:
            ax.hist(p, bins=bins, density=True, histtype="step", linestyle="--",
                    linewidth=1.6, color=_PRIOR_COLOR, label="Prior")
        ax.hist(g, bins=bins, alpha=0.5, density=True, color=_GEN_COLOR, label="Gen")
        ax.set_xlabel(titles[i])
        ax.set_ylabel("Density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def physicality_isotropy_scalars(
    gen_cartesian: torch.Tensor,
    test_polar_abs: torch.Tensor,
) -> dict:
    """Scalar physicality/isotropy diagnostics for the metrics CSV.

    Returns frac_negative_energy, frac_spacelike, msq_mean, msq_median, and KS
    stats for jet-axis (cosθ, φ) vs. uniform *and* vs. test.
    """
    gen = gen_cartesian.detach().cpu()
    test = test_polar_abs.detach().cpu()
    real = gen.abs().sum(dim=-1) > 1e-12
    E = gen[..., 0]
    msq = normsq4(gen)
    if not bool(real.any()):
        return {"frac_negative_energy": float("nan"), "frac_spacelike": float("nan")}
    frac_neg_E     = (E < 0)[real].float().mean().item()
    frac_spacelike = (msq < 0)[real].float().mean().item()
    msq_real = msq[real]

    diag = {
        "frac_negative_energy": frac_neg_E,
        "frac_spacelike": frac_spacelike,
        "msq_mean":   msq_real.mean().item()   if msq_real.numel() else float("nan"),
        "msq_median": msq_real.median().item() if msq_real.numel() else float("nan"),
    }

    gen_total = gen[..., 1:4].sum(dim=1)
    gen_cos, gen_phi = _total_momentum_direction(gen_total)
    ks_cos_d, ks_cos_p = ks_statistic_vs_uniform(gen_cos.numpy(), -1.0, 1.0)
    ks_phi_d, ks_phi_p = ks_statistic_vs_uniform(gen_phi.numpy(), -math.pi, math.pi)

    eta, phi_t, pt = test[..., 0], test[..., 1], test[..., 2]
    real_t = (pt > 0).float()
    px = (pt * torch.cos(phi_t)) * real_t
    py = (pt * torch.sin(phi_t)) * real_t
    pz = (pt * torch.sinh(eta)) * real_t
    test_total = torch.stack([px.sum(1), py.sum(1), pz.sum(1)], dim=-1)
    test_cos, test_phi = _total_momentum_direction(test_total)
    gvd_cos_d, gvd_cos_p = ks_two_sample(gen_cos.numpy(), test_cos.numpy())
    gvd_phi_d, gvd_phi_p = ks_two_sample(gen_phi.numpy(), test_phi.numpy())

    diag.update({
        "isotropy_ks_costheta": ks_cos_d,
        "isotropy_ks_costheta_p": ks_cos_p,
        "isotropy_ks_phi": ks_phi_d,
        "isotropy_ks_phi_p": ks_phi_p,
        "isotropy_gen_vs_data_cos_ks": gvd_cos_d,
        "isotropy_gen_vs_data_cos_p": gvd_cos_p,
        "isotropy_gen_vs_data_phi_ks": gvd_phi_d,
        "isotropy_gen_vs_data_phi_p": gvd_phi_p,
    })
    return diag


def _plot_physicality(gen_cartesian: torch.Tensor, out_path: str) -> None:
    real = gen_cartesian.abs().sum(dim=-1) > 1e-12
    E = gen_cartesian[..., 0]
    msq = normsq4(gen_cartesian)
    if not bool(real.any()):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No real particles", ha="center", va="center")
        fig.savefig(out_path, dpi=200); plt.close(fig); return
    frac_neg_E    = (E < 0)[real].float().mean().item()
    frac_spacelike = (msq < 0)[real].float().mean().item()
    msq_real = msq[real].numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(msq_real, bins=120, density=True, color="tab:purple", alpha=0.8)
    ax.axvline(0.0, color="k", ls="--", lw=1, label=r"$m^2=0$ (lightlike)")
    ax.set_xlabel(r"$m^2 = \langle x, x\rangle$ (generated particles)")
    ax.set_ylabel("Density")
    ax.set_title(f"Physicality: E<0 {frac_neg_E:.2%}, spacelike (m²<0) {frac_spacelike:.2%}",
                 fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _total_momentum_direction(p3: torch.Tensor):
    norm = p3.norm(dim=-1).clamp(min=1e-12)
    cos_theta = p3[..., 2] / norm
    phi = torch.atan2(p3[..., 1], p3[..., 0])
    return cos_theta, phi


def _plot_isotropy(
    gen_cartesian: torch.Tensor,
    test_polar_abs: torch.Tensor,
    out_path: str,
) -> None:
    gen_total = gen_cartesian[..., 1:4].sum(dim=1)
    gen_cos, gen_phi = _total_momentum_direction(gen_total)

    eta, phi_t, pt = test_polar_abs[..., 0], test_polar_abs[..., 1], test_polar_abs[..., 2]
    real = (pt > 0).float()
    px = (pt * torch.cos(phi_t)) * real
    py = (pt * torch.sin(phi_t)) * real
    pz = (pt * torch.sinh(eta)) * real
    test_total = torch.stack([px.sum(1), py.sum(1), pz.sum(1)], dim=-1)
    test_cos, test_phi = _total_momentum_direction(test_total)

    ks_cos_d, _ = ks_two_sample(gen_cos.numpy(), test_cos.numpy())
    ks_phi_d, _ = ks_two_sample(gen_phi.numpy(), test_phi.numpy())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Isotropy of jet-axis direction — KS(cosθ)={ks_cos_d:.3f},  KS(φ)={ks_phi_d:.3f}",
                 fontsize=13)
    axes[0].hist(test_cos.numpy(), bins=60, density=True, alpha=0.5, color=_TEST_COLOR, label="Test")
    axes[0].hist(gen_cos.numpy(),  bins=60, density=True, alpha=0.5, color=_GEN_COLOR,  label="Gen")
    axes[0].axhline(0.5, color="gray", ls=":", lw=1, label="Isotropic (uniform)")
    axes[0].set_xlabel(r"$\cos\theta$ of total 3-momentum"); axes[0].set_ylabel("Density"); axes[0].legend()
    axes[1].hist(test_phi.numpy(), bins=60, density=True, alpha=0.5, color=_TEST_COLOR, label="Test")
    axes[1].hist(gen_phi.numpy(),  bins=60, density=True, alpha=0.5, color=_GEN_COLOR,  label="Gen")
    axes[1].set_xlabel(r"$\phi$ of total 3-momentum"); axes[1].set_ylabel("Density"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Per-class overlay ---------------------------------------------------

def _overlay_hist_by_class(
    ax,
    test_vals_by_class: list,
    gen_vals_by_class: list,
    names: Sequence[str],
    xlabel: str,
    bins,
    log_y: bool = False,
    density: bool = True,
    prior_vals_by_class: Optional[list] = None,
) -> None:
    """Test as filled+alpha, gen as solid step, one color per class."""
    for c, name in enumerate(names):
        color = _CLASS_PALETTE[c % len(_CLASS_PALETTE)]
        t = test_vals_by_class[c]
        g = gen_vals_by_class[c]
        if len(t) == 0 and len(g) == 0:
            continue
        if len(t) > 0:
            ax.hist(t, bins=bins, density=density, alpha=0.28, color=color,
                    label=f"{name} (test)")
        if prior_vals_by_class is not None and len(prior_vals_by_class[c]) > 0:
            ax.hist(prior_vals_by_class[c], bins=bins, density=density, histtype="step",
                    linestyle="--", color=_PRIOR_COLOR, linewidth=1.6,
                    label=f"{name} (prior)")
        if len(g) > 0:
            ax.hist(g, bins=bins, density=density, histtype="step", color=color,
                    linewidth=1.8, label=f"{name} (gen)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)


def _split_scalar_by_class(vals: torch.Tensor, types: torch.Tensor, n_classes: int):
    return [vals[types == c].numpy() for c in range(n_classes)]


def _jet_girth(polar_rel: torch.Tensor) -> torch.Tensor:
    """g = Σ_i (pT_i/jet_pT) · ΔR_i.  pT_rel already = pT_i/jet_pT."""
    eta, phi, pt_rel = polar_rel[..., 0], polar_rel[..., 1], polar_rel[..., 2]
    dR = torch.sqrt(eta * eta + phi * phi)
    return (pt_rel.clamp(min=0) * dR).sum(dim=1)


def _plot_jet_girth(tp_rel, gp_rel, t_types, g_types, names, out_path, pp_rel=None):
    t_g = _jet_girth(tp_rel)
    g_g = _jet_girth(gp_rel)
    n = len(names)
    fig, ax = plt.subplots(figsize=(8, 5))
    p_g = _jet_girth(pp_rel) if pp_rel is not None else None
    maxima = [t_g.max(), g_g.max()] + ([p_g.max()] if p_g is not None else [])
    bins = np.linspace(0.0, float(max(maxima).item()) * 1.02 + 1e-6, 60)
    _overlay_hist_by_class(
        ax,
        _split_scalar_by_class(t_g, t_types, n),
        _split_scalar_by_class(g_g, g_types, n),
        names,
        xlabel=r"Jet girth $g = \sum_i (p_T^i/p_T^{\rm jet})\, \Delta R_i$",
        bins=bins,
        prior_vals_by_class=(_split_scalar_by_class(p_g, g_types, n)
                             if p_g is not None else None),
    )
    ax.set_title("Jet girth by class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_jet_multiplicity(tp_rel, gp_rel, t_types, g_types, names, out_path):
    t_m = _real_mask(tp_rel).sum(dim=1)
    g_m = _real_mask(gp_rel).sum(dim=1)
    n = len(names)
    hi = int(max(t_m.max(), g_m.max()).item()) + 1
    bins = np.arange(0, hi + 1) - 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    _overlay_hist_by_class(
        ax,
        _split_scalar_by_class(t_m, t_types, n),
        _split_scalar_by_class(g_m, g_types, n),
        names,
        xlabel="Number of particles per jet",
        bins=bins,
    )
    ax.set_title("Particle multiplicity by class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_jet_pt_total(tp_abs, gp_abs, t_types, g_types, names, pt_cond, out_path, pp_abs=None):
    t_pt = _jet_pt_scalar(tp_abs)
    g_pt = _jet_pt_scalar(gp_abs)
    n = len(names)
    p_pt = _jet_pt_scalar(pp_abs) if pp_abs is not None else None
    maxima = [t_pt.max(), g_pt.max()] + ([p_pt.max()] if p_pt is not None else [])
    hi = float(max(maxima).item()) * 1.02
    bins = np.linspace(0.0, hi + 1e-6, 60)

    fig, ax = plt.subplots(figsize=(8, 5))
    _overlay_hist_by_class(
        ax,
        _split_scalar_by_class(t_pt, t_types, n),
        _split_scalar_by_class(g_pt, g_types, n),
        names,
        xlabel=r"Total jet $\sum_i p_T^i$",
        bins=bins,
        prior_vals_by_class=(_split_scalar_by_class(p_pt, g_types, n)
                             if p_pt is not None else None),
    )
    title = "Total jet pT by class"
    if pt_cond is not None:
        try:
            slope, intercept, r_value, _, _ = stats.linregress(pt_cond.numpy(), g_pt.numpy())
            title += rf"   ·   gen-vs-cond fit: slope={slope:.3f}, $R^2$={r_value ** 2:.3f}"
        except Exception:
            pass
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_radial_profile(tp_rel, gp_rel, t_types, g_types, names, out_path, pp_rel=None):
    """ρ(ΔR) weighted by pT_rel, per class overlay."""
    n = len(names)
    bins = np.linspace(0.0, 0.8, 60)

    fig, ax = plt.subplots(figsize=(8, 5))
    for c, name in enumerate(names):
        color = _CLASS_PALETTE[c % len(_CLASS_PALETTE)]
        series = [("test", tp_rel, t_types), ("gen", gp_rel, g_types)]
        if pp_rel is not None:
            series.insert(1, ("prior", pp_rel, g_types))
        for tag, polar, types in series:
            sel = (types == c)
            if not bool(sel.any()):
                continue
            p = polar[sel]
            real = _real_mask(p)
            eta = p[..., 0][real].numpy()
            phi = p[..., 1][real].numpy()
            wt  = p[..., 2][real].numpy()
            phi = ((phi + np.pi) % (2 * np.pi)) - np.pi
            dR = np.sqrt(eta * eta + phi * phi)
            if tag == "test":
                ax.hist(dR, bins=bins, weights=wt, density=True, alpha=0.28,
                        color=color, label=f"{name} (test)")
            elif tag == "prior":
                ax.hist(dR, bins=bins, weights=wt, density=True, histtype="step",
                        linestyle="--", color=_PRIOR_COLOR, linewidth=1.6,
                        label=f"{name} (prior)")
            else:
                ax.hist(dR, bins=bins, weights=wt, density=True, histtype="step",
                        color=color, linewidth=1.8, label=f"{name} (gen)")
    ax.set_xlabel(r"$\Delta R$ (from jet axis)")
    ax.set_ylabel(r"$p_T$-weighted density")
    ax.set_title("Radial energy profile by class")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_energy_tails(tp_abs, gp_abs, t_types, g_types, names, out_path, pp_abs=None):
    """(E, |p|) × (log-y hist, QQ).  Overlay 5 classes per hist panel; per-class
    QQ curves in the QQ panels."""
    tp_cart = _polar_abs_to_cartesian(tp_abs)
    gp_cart = _polar_abs_to_cartesian(gp_abs)
    real_t = _real_mask(tp_abs)
    real_g = _real_mask(gp_abs)
    pp_cart = _polar_abs_to_cartesian(pp_abs) if pp_abs is not None else None
    real_p = _real_mask(pp_abs) if pp_abs is not None else None

    def _per_particle_by_class(cart, real, jet_types, idx_col):
        """idx_col='E' or 'p' -> per-real-particle scalar, gathered by jet class."""
        n_classes = len(names)
        out = [[] for _ in range(n_classes)]
        for c in range(n_classes):
            sel = (jet_types == c)
            if not bool(sel.any()):
                continue
            m = real[sel].numpy()
            block = cart[sel].numpy()
            if idx_col == "E":
                vals = block[..., 0][m]
            else:
                vals = np.linalg.norm(block[..., 1:4], axis=-1)[m]
            out[c] = vals
        return out

    t_E = _per_particle_by_class(tp_cart, real_t, t_types, "E")
    g_E = _per_particle_by_class(gp_cart, real_g, g_types, "E")
    t_P = _per_particle_by_class(tp_cart, real_t, t_types, "p")
    g_P = _per_particle_by_class(gp_cart, real_g, g_types, "p")
    p_E = _per_particle_by_class(pp_cart, real_p, g_types, "E") if pp_cart is not None else None
    p_P = _per_particle_by_class(pp_cart, real_p, g_types, "p") if pp_cart is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Energy tails per class (log-y histograms + Q-Q)", fontsize=13)

    for ax, feats, xlabel in [
        (axes[0, 0], (t_E, g_E, p_E), "E (per real particle)"),
        (axes[0, 1], (t_P, g_P, p_P), r"$|\vec p|$ (per real particle)"),
    ]:
        t_by, g_by, p_by = feats
        all_v = np.concatenate([v for v in (t_by + g_by + (p_by or [])) if len(v)])
        if all_v.size == 0:
            continue
        bins = np.linspace(0, np.quantile(all_v, 0.999), 80)
        _overlay_hist_by_class(ax, t_by, g_by, names, xlabel=xlabel,
                               bins=bins, log_y=True, density=True,
                               prior_vals_by_class=p_by)

    # QQ plots — one solid line per class (gen quantiles vs test quantiles).
    for ax, feats, xlabel in [
        (axes[1, 0], (t_E, g_E), "E: test quantile"),
        (axes[1, 1], (t_P, g_P), r"$|\vec p|$: test quantile"),
    ]:
        t_by, g_by = feats
        qs = np.linspace(0.01, 0.99, 60)
        for c, name in enumerate(names):
            if len(t_by[c]) == 0 or len(g_by[c]) == 0:
                continue
            color = _CLASS_PALETTE[c % len(_CLASS_PALETTE)]
            tq = np.quantile(t_by[c], qs)
            gq = np.quantile(g_by[c], qs)
            ax.plot(tq, gq, color=color, label=name, lw=1.6)
        lims = ax.get_xlim() + ax.get_ylim()
        lo, hi = min(lims), max(lims)
        ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=0.8)
        ax.set_xlabel(xlabel); ax.set_ylabel("gen quantile")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Per-class grid ------------------------------------------------------

def _grid_shape(n_classes: int, include_pooled: bool = True) -> tuple[int, int, int]:
    """(rows, cols, n_panels) for n_classes (+1 if pooled)."""
    n_panels = n_classes + (1 if include_pooled else 0)
    if n_panels <= 3:
        return 1, n_panels, n_panels
    cols = 3
    rows = math.ceil(n_panels / cols)
    return rows, cols, n_panels


def _jet_invariant_mass(cartesian: torch.Tensor) -> torch.Tensor:
    """m² = (ΣE)² − |Σp|²  (padding already zeroed). Returns m (clamped)."""
    p_mu_sum = cartesian.sum(dim=1)   # (N, 4)
    E = p_mu_sum[..., 0]
    p2 = (p_mu_sum[..., 1:4] ** 2).sum(dim=-1)
    m2 = E * E - p2
    return torch.sqrt(m2.clamp(min=0.0))


def _plot_jet_mass(test_cart, gen_cart, t_types, g_types, names, out_path, prior_cart=None):
    t_m = _jet_invariant_mass(test_cart)
    g_m = _jet_invariant_mass(gen_cart)
    n = len(names)
    rows, cols, n_panels = _grid_shape(n, include_pooled=True)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(r"Invariant jet mass  $m_{\rm jet}=\sqrt{(\Sigma E)^2 - |\Sigma \vec p|^2}$",
                 fontsize=14)

    p_m = _jet_invariant_mass(prior_cart) if prior_cart is not None else None
    maxima = [t_m.max(), g_m.max()] + ([p_m.max()] if p_m is not None else [])
    hi = float(max(maxima).item()) * 1.02 + 1e-6
    bins = np.linspace(0.0, hi, 80)

    panels = [(f"class {name}", t_m[t_types == c].numpy(), g_m[g_types == c].numpy(),
               p_m[g_types == c].numpy() if p_m is not None else None)
              for c, name in enumerate(names)]
    panels.append(("pooled", t_m.numpy(), g_m.numpy(), p_m.numpy() if p_m is not None else None))
    for k, (title, t_vals, g_vals, p_vals) in enumerate(panels):
        ax = axes[k // cols, k % cols]
        if len(t_vals) == 0 and len(g_vals) == 0:
            ax.set_visible(False); continue
        if len(t_vals):
            ax.hist(t_vals, bins=bins, density=True, alpha=0.5, color=_TEST_COLOR, label="Test")
        if p_vals is not None and len(p_vals):
            ax.hist(p_vals, bins=bins, density=True, histtype="step", linestyle="--",
                    linewidth=1.6, color=_PRIOR_COLOR, label="Prior")
        if len(g_vals):
            ax.hist(g_vals, bins=bins, density=True, alpha=0.5, color=_GEN_COLOR, label="Gen")
        ax.set_xlabel(r"$m_{\rm jet}$")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=9)

    for k in range(len(panels), rows * cols):
        axes[k // cols, k % cols].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _leading_sub_pt_fractions(polar_rel: torch.Tensor):
    """z1 (leading), z2 (subleading) pT fractions per jet. pT_rel already normalized."""
    pt_rel = polar_rel[..., 2].clamp(min=0)
    sorted_, _ = torch.sort(pt_rel, dim=1, descending=True)
    z1 = sorted_[:, 0]
    z2 = sorted_[:, 1] if sorted_.shape[1] > 1 else torch.zeros_like(z1)
    return z1, z2


def _plot_jet_z1_z2(tp_rel, gp_rel, t_types, g_types, names, out_path):
    t_z1, t_z2 = _leading_sub_pt_fractions(tp_rel)
    g_z1, g_z2 = _leading_sub_pt_fractions(gp_rel)
    n = len(names)
    rows, cols, n_panels = _grid_shape(n, include_pooled=True)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(r"Leading & subleading $p_T$ fractions ($z_1$, $z_2$) by class", fontsize=13)

    bins = np.linspace(0.0, 1.0, 60)
    panels = []
    for c, name in enumerate(names):
        panels.append((name,
                       t_z1[t_types == c].numpy(), t_z2[t_types == c].numpy(),
                       g_z1[g_types == c].numpy(), g_z2[g_types == c].numpy()))
    panels.append(("pooled", t_z1.numpy(), t_z2.numpy(), g_z1.numpy(), g_z2.numpy()))

    for k, (title, tz1, tz2, gz1, gz2) in enumerate(panels):
        ax = axes[k // cols, k % cols]
        if all(len(v) == 0 for v in (tz1, tz2, gz1, gz2)):
            ax.set_visible(False); continue
        # z1 filled, z2 outlined — two observables in one panel, test vs gen distinguished by color.
        if len(tz1): ax.hist(tz1, bins=bins, density=True, alpha=0.35, color=_TEST_COLOR, label=r"$z_1$ test")
        if len(gz1): ax.hist(gz1, bins=bins, density=True, alpha=0.35, color=_GEN_COLOR,  label=r"$z_1$ gen")
        if len(tz2): ax.hist(tz2, bins=bins, density=True, histtype="step", color=_TEST_COLOR, lw=1.4, label=r"$z_2$ test")
        if len(gz2): ax.hist(gz2, bins=bins, density=True, histtype="step", color=_GEN_COLOR,  lw=1.4, label=r"$z_2$ gen")
        ax.set_xlabel(r"$p_T^i/p_T^{\rm jet}$")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)

    for k in range(len(panels), rows * cols):
        axes[k // cols, k % cols].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_jet_image_by_class(tp_rel, gp_rel, t_types, g_types, names, out_path):
    n = len(names)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.5 * n), squeeze=False)
    fig.suptitle("Averaged jet image per class", fontsize=14)
    for c, name in enumerate(names):
        t_sel = (t_types == c)
        g_sel = (g_types == c)
        if not (bool(t_sel.any()) and bool(g_sel.any())):
            for ax in axes[c]:
                ax.set_visible(False)
            continue
        t_img = _pt_weighted_avg_image(tp_rel[t_sel, :, 0],
                                       tp_rel[t_sel, :, 1],
                                       tp_rel[t_sel, :, 2])
        g_img = _pt_weighted_avg_image(gp_rel[g_sel, :, 0],
                                       gp_rel[g_sel, :, 1],
                                       gp_rel[g_sel, :, 2])
        _draw_jet_image_triptych(*axes[c], t_img, g_img, title_prefix=f"{name}  ·  ")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
