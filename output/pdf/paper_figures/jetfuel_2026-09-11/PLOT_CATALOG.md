# JetFUEL paper-figure catalog

All figures in this directory use completed saved runs only. The CPU job did not
generate new samples or recompute physics metrics. Exact PVC sources are recorded in
`figure_manifest.json`.

## Strongest main-text candidates

1. `selected_cfg_distributions.pdf` — direct overlays of JetNet test samples and
   JetFUEL-generated samples for relative constituent $\eta$, $\phi$, and $p_T$,
   plus relative jet mass, using the selected guidance value for each class. This
   is the closest counterpart to the MPGAN-style
   distribution panel.
2. `euler_step_scaling.pdf` — the cleanest inference ablation. It shows monotonic
   FPND improvement from 32 to 64 to 128 steps for every GQT class, with no retraining.
3. `icp_ablation.pdf` — the clearest training-time scientific result: removing ICP
   damages FPND, FPD, and W1M together while physicality remains exact.
4. `source_diagnostics/*/jet_image_by_class.png` — averaged test/generated jet images
   and residual maps. For the paper, use the row corresponding to the selected class
   and guidance value: g from `gqt_w000`, q from `gqt_w025`, and t from `gqt_w050`.

## Good appendix or diagnostic candidates

- `training_curves_icp.pdf` — useful for explaining why the no-ICP endpoint is worse;
  the training losses themselves separate by a large factor.
- `cfg_guidance_sweep.pdf` — supports the class-dependent CFG choice and shows the
  rapid degradation at excessive guidance.
- `architecture_g_to_j.pdf` — compact visual for evolving versus fixed auxiliary
  geometry and normalized versus raw references.
- `source_diagnostics/*/particle_kinematics_rel.png` — pooled relative constituent
  kinematics with the prior shown as a visual transport baseline.
- `source_diagnostics/*/radial_profile.png` and `jet_girth.png` — interpretable jet
  shape comparisons.
- `source_diagnostics/*/jet_pt_total.png` — exposes remaining conditioning-response
  bias rather than only showing favorable observables.
- `source_diagnostics/*/physicality.png`, `energy_tails.png`, and `isotropy.png` — best
  suited to an appendix or supplemental material.

## Additional figures supported by existing data

- A quality-versus-cost Pareto plot using the saved 32/64/128 generation times and
  per-class FPND values. The existing Euler figure can be extended with a time axis.
- A physical-support before/after panel using the A--D Euclidean/mass-shell spacelike,
  negative-energy, and invalid fractions. This would visually motivate the geometry
  more strongly than a metric table alone.
- A coverage-versus-MMD scatter for G--J, ICP/no-ICP, and the capacity ablations. This
  would show that a single marginal metric does not capture the design tradeoffs.
- A sampler-stability timeline for zero versus small-normal readout initialization,
  using the stored 0/100/500/1000/2000-step unstable-trajectory counts.
- Bootstrap error bars on W1M/W1P/FPD from the standard errors already stored in each
  `summary.json`. This is preferable before making claims from small secondary-metric
  differences.

The most defensible first paper draft would use the selected distribution panel, Euler
scaling, ICP ablation, and selected-class jet-image residuals in the main text; move CFG,
training curves, G--J, and physicality diagnostics to the appendix.

## Visual system

The curated root figures are generated locally with Seaborn, Computer Modern through
LaTeX, no in-figure titles, no gridlines, and quiet two-spine axes. Gluon, light-quark,
and top curves reserve blue, vermilion, and green, respectively, throughout the bundle.
Those hues are not reused for architectural states: canonical/ICP is black, no-ICP is
magenta, evolving or mass-shell geometry is purple, fixed geometry is light gray, and
small-normal initialization is gold. Both PDF and high-resolution PNG versions are
provided; captions and interpretation belong in the manuscript rather than the figure.

All curated figures except the averaged jet-image diagnostics can be regenerated from
the compact files in `data/` without PVC access. The source diagnostics are retained as
an archival record and preserve the evaluation harness's original styling.
