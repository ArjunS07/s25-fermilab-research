# Paper-standard evaluation results — JetFUEL, JetNet-30

This is the authoritative post-hoc evaluation record for the paper comparison
campaign.  It supersedes terminal metrics for one-class Q/T runs, whose old
sampler incorrectly used the global gluon conditioning code.  Every evaluation
below uses the repair in commit `c232473534c2614ddcc2d96ace74481947c28bdf`,
EMA weights, 64 Euler integration steps, the lognormal axis-aligned prior, and
the corrected canonical FPND input convention.

## Completed evaluations

| Endpoint | Intended generated sample count | FPND | FPD | W1M | Mean W1P | Coverage | MMD | Invalid | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GQT-994K, no CFG | 50K each G/Q/T | g **0.1343**; q 0.5329; t 2.9997 | ⚠ 1.0000 | 0.000974 | 0.001068 | 0.554 | 0.036616 | 4 / 150K | Per-class FPND valid; aggregate EFP/FPD tail-contaminated, not paper-ready |
| Q-331K | 50K q | **0.2535** | 0.001612 | 0.001226 | 0.001527 | 0.547 | 0.018396 | 0 / 50K | Valid corrected specialist result |
| T-994K | 50K t | 2.5867 | ⚠ 2039.25 | 0.002627 | 0.001492 | 0.566 | 0.055754 | 29 / 50K | Valid sampling audit; stability/tail warning |
| T-331K | 50K t | **0.7617** | 0.217991 | 0.002798 | 0.001204 | 0.577 | 0.054845 | 0 / 50K | Valid corrected specialist result |
| G-994K | 50K g | 0.4007 | 0.000196 | 0.002309 | 0.001515 | 0.555 | 0.026607 | 0 / 50K | Valid specialist result |
| G-331K | 50K g | 0.3968 | 0.000167 | 0.002234 | 0.001457 | 0.576 | 0.025888 | 0 / 50K | Valid specialist result |
| GQT-994K, CFG | 50K each G/Q/T | g 0.9160; q 1.6340; t **0.4261** | 0.192837 | 0.001062 | 0.001300 | 0.567 | 0.036554 | 11 / 150K | Valid per-class evaluation; CFG shifts the bottleneck toward g/q |

`Mean W1P` is the mean of the three reported particle-Wasserstein components.
All finite generated endpoints have 0% negative-energy and 0% spacelike
particles; the mass-shell residual is at floating-point precision.

## Interpretation and audit notes

- **The old Q/T specialist terminal metrics are invalid.**  Their training
  checkpoints are valid, but their former evaluation generated global class 0
  (gluon) for any one-class configuration.  The replay bundles above verify the
  repair directly: Q-331K contains `[0, 50,000, 0, 0, 0]` and T runs contain
  `[0, 0, 50,000, 0, 0]` counts in global `[g,q,t,w,z]` order.
- **T-331K, not T-994K, is currently the better top specialist.**  The 994K
  sample set has 29 integration failures (0.058%) and a tail-sensitive FPD;
  this is a model/sampler stability outcome, not the former type-label bug.
- **GQT-994K has valid per-class FPND but not a usable aggregate EFP/FPD row.**
  It has 4 failed trajectories (3 integration failures, 1 `nonfinite_velocity`) and
  2,926 finite jets with a component magnitude above 1 TeV, concentrated in Q.
  These finite tails make aggregate W1-EFP large and FPD=1.0.  Do not compare
  that aggregate FPD with specialist FPDs in the paper.  Its per-class FPND is
  still the appropriate ParticleNet comparison; after dropping failures the
  classes contain 49,999 G, 49,998 Q, and 49,999 T samples.

## Artifact provenance

| Endpoint | Evaluation root on NRP PVC | Local downloaded copy |
|---|---|---|
| GQT-994K | `/mnt/data/output/2026-08-23_04-20-40--8782785a-a3de-4145-a4b9-6ff6c2a54a74-paper30-gqt994k-eval/eval` | `src/downloaded_output/2026-08-23_04-20-40--8782785a-a3de-4145-a4b9-6ff6c2a54a74-paper30-gqt994k-eval/eval` |
| Q-331K | `/mnt/data/output/2026-08-23_05-05-04--571cdc15-e259-44ad-9189-f9ab39083017-paper30-q331k-eval/eval` | `src/downloaded_output/2026-08-23_05-05-04--571cdc15-e259-44ad-9189-f9ab39083017-paper30-q331k-eval/eval` |
| T-994K | `/mnt/data/output/2026-08-23_05-19-23--95469257-484f-4652-8069-c390a8bf5294-paper30-t994k-eval/eval` | `src/downloaded_output/2026-08-23_05-19-23--95469257-484f-4652-8069-c390a8bf5294-paper30-t994k-eval/eval` |
| T-331K | `/mnt/data/output/2026-08-23_05-37-01--2ad62d6e-6179-409c-96cf-5232b7feded6-paper30-t331k-eval/eval` | `src/downloaded_output/2026-08-23_05-37-01--2ad62d6e-6179-409c-96cf-5232b7feded6-paper30-t331k-eval/eval` |
| G-994K | `/mnt/data/output/2026-08-23_05-51-21--af9a1f03-4f89-4f3e-92da-2c2051ab26d4-paper30-g994k-eval/eval` | `src/downloaded_output/2026-08-23_05-51-21--af9a1f03-4f89-4f3e-92da-2c2051ab26d4-paper30-g994k-eval/eval` |
| G-331K | `/mnt/data/output/2026-08-23_06-05-40--f6369054-fd3b-491d-8b12-f2296443466d-paper30-g331k-eval/eval` | `src/downloaded_output/2026-08-23_06-05-40--f6369054-fd3b-491d-8b12-f2296443466d-paper30-g331k-eval/eval` |
| GQT-994K, CFG | `/mnt/data/output/2026-08-24_02-29-13--a85d5bf6-8808-4dd7-93bf-b831ca34fdf4-paper30-gqt994k-cfg-eval/eval` | `src/downloaded_output/2026-08-24_02-29-13--a85d5bf6-8808-4dd7-93bf-b831ca34fdf4-paper30-gqt994k-cfg-eval/eval` |

Each local copy includes `summary.json`, `metrics.csv`, samples, the exact replay
bundle, generation diagnostics, endpoint-tail diagnostics, and evaluation figures.

## Outstanding

- GQT-994K CFG training is complete; its corrected 150K evaluation is now queued
  as `as-jet-eval-paper30-h-cfg`.
- Q-994K is still training (approximately 661K / 994K); its corrected evaluation
- Q-994K training is complete at 994K; its corrected 50K-Q evaluation is now
  running as `as-jet-eval-paper30-h-q-qklbj`.
- A dedicated 331K GQT run was not launched; it remains necessary only if we
  retain the low-compute shared-representation comparison in the paper.
