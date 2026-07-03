# Implementation status — master context

*Written 2026-07-03. Snapshot of what has been built vs. what remains, across the
experiment plan (`02-experiment-plan.md`). All work is on `main`.*

## Design principle carried throughout

Every new capability is **flag-gated and default-off**, so the Phase-1 ablation grid
(runs A–F) is byte-for-byte the original model unless a flag is set. Flags select *what
varies*; a pinned git commit holds *everything else constant* (frozen-harness gate). CPU
tests (`src/tests/`, 62 passing) verify equivariance/wiring since no GPU is available
locally.

## Implemented (Phase 0, 1, 2)

**Phase 0 — diagnostics/harness**
- Blocking equivariance pytest (`tests/test_equivariance.py`): joint Lorentz equivariance
  (rotations+boosts), particles-only rotation changes output with refs on, residual SO(2)
  about jet axis, run-F stays isotropic.
- Physicality diagnostics + isotropy figure in `util/metrics.py::run_save_metrics`
  (`physicality.png`, `isotropy.png`, fractions in `metrics.csv`). Fixed `EVAL_SEED`.

**Phase 1 — symmetry breaking** (the headline fix; see `01-symmetry-mismatch-diagnosis.md`)
- `--use_reference_vectors`: e_t=(1,0,0,0) + jet 4-momentum as unmasked virtual particles
  (R=2), fixed across layers, displacements discarded. `models/LEFT_JeN.py`.
- `--use_node_scalars`: LorentzNet-style per-node hidden `h_i`, seeded from
  `[ψ(m²), ψ(E), ψ(⟨x,x_jet⟩)]` (E/axis zeroed when refs off → run F stays equivariant).
- `--prior_dist=axis_aligned`: collimated near-lightlike prior around the jet axis
  (`util/distributions.py`).
- References built in `train.py` (sum of scaled constituents) and reconstructed at
  inference (`generate_samples.build_reference_vectors`, from jet attrs via
  `EtaPhiPtE_to_cartesian`/`final_scale`).
- `--eta_min_factor` knob for the post-Phase-1 LR re-sweep.

**Phase 2 — training hygiene** (flag-gated)
- `--use_ema` / `--ema_decay` (`util/ema.py`): EMA weights saved as `ema_model.pth`, used
  for end-of-training sampling/metrics, resumable.
- `--use_adaln`: FiLM/adaLN conditioning replaces per-edge concat of g/t (memory fix,
  `phi_e` input 50→2 dims); verified equivariant.
- `--sampler {euler,heun}`: Heun RK2 integrator.

## Deployment (NRP / Kubernetes, `src/nrp/`)

- **Phase-1 grid**: `as-jet-train-job-g30-phase1-{a..f}.yaml`. A baseline, B refs,
  C refs+h_i, D refs+h_i+axis-prior, E axis-prior only, F h_i only. ICP off, gluon-30,
  1 GPU each (keep at 1 for comparability). Each **fails hard until `GIT_COMMIT` is pinned**
  and records `git_commit.txt`.
- All training jobs launch via `torchrun --nproc_per_node=$(auto-detect)` → use every GPU
  the pod is granted. Bump `nvidia.com/a40` only for 150p / scaled runs (effective batch
  scales with world_size).
- Artifacts slimmed: download only `train/` (drops dataset pkl); `summary.json`,
  `final_checkpoint.pth` (full resume state + self-describing `config`), `samples_subset.pt`.
  `integration_steps=64`, debug env flags removed.

## Remaining / not yet implemented

- **Phase 2 leftovers**: LR re-sweep run (knob exists, sweep not run — must be *after*
  Phase 1), curriculum re-ablation.
- **Phase 3 — backbone**: invariant-logit softmax attention (3.1); depth/width scaling
  (3.2). **L-GATr head-to-head (3.3) deferred** — predefine metric set first.
- **Phase 4 — mass-shell RFM (research headline, conditional on Phase 1 success)**:
  rewrite `util/hyperbolic.py` as hyperboloid/mass-shell (float64 by default), integrate via
  `--use_hyperbolic`, park padding at (m,0,0,0) **with masking enforced inside the geodesic
  ICP/OT cost**, regulator-mass ablation, geodesic ICP in `cache_icp.py`, Euclidean-vs-RFM
  head-to-head.
- **Phase 5**: full ablation table, steps-vs-quality curves, 150p then JetClass.

## Execution order

Pin `GIT_COMMIT` → run grid A–F (1 GPU each, 2 batches of 3) → read `isotropy.png`/`summary.json`
(run B must depart from uniform, else refs are broken) → if strong, Phase 4 becomes headline
and Phase 3 supports; if not, debug before adding geometry.

## Known caveats to watch on first real runs

- Inference reference reconstruction is scale-consistent with training by construction but
  differs slightly in jet mass/phi source — sanity-check run B's isotropy figure.
- `accumulation_steps` is not divided by world_size → multi-GPU = larger effective batch
  (wants LR scaling; keep the grid at 1 GPU).
