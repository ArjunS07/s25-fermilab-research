# AGENTS.md

Onboarding notes for a fresh coding-agent session working on this repo. Read this
first, then **`discussions/23-status-post-fpnd-fix.md`** — the canonical current-status
entry point. For phase-level history see `discussions/02-experiment-plan.md` and
`discussions/03-implementation-status.md`.

> **Note:** FPND-g ≈ 24 was a coordinate-convention artifact, not a model failure. Any
> doc premised on it being a real "joint-structure failure" is voided and moved to
> `discussions/archive/` (docs 12/19/20/21/22). See doc 23 + `FPND_PTREL_BUG.md`.

## What this project is

**LEFTJeN** — a Lorentz-equivariant flow-matching generative model for
particle-physics jet generation. Two-stage:

1. **Jet-attribute normalizing flow** (`src/jet_attr_model.py`) over
   `[type, η, pT, mass, n_particles]`.
2. **LEFTJeN flow-matching generator** (`src/models/LEFT_JeN.py`) over jet
   constituents (4-vectors), conditioned on Stage-1 output.

Primary target: JetNet **gluon-30** (30-particle gluon jets). Research thesis:
supersede L-GATr with a simpler backbone + **mass-shell Riemannian flow
matching** geometry + **reference-vector symmetry breaking**. See
`discussions/02-experiment-plan.md` for the full plan and
`discussions/06-physics-primer.md` for the physics context (spacelike /
lightlike / timelike, healthy distributions, failure-mode checklist).

## Nothing runs locally

Every real experiment is a Kubernetes Job on NRP/Nautilus under
`src/nrp/*.yaml`. Your laptop is for reading, editing, analyzing outputs
downloaded to `src/downloaded_output/`, and running the CPU pytest suite. There
is no GPU locally.

## Repo layout

**Root:**
- `PHASE1_STATUS.md` — running log of the Phase-1 grid launch (incl. the PVC
  provisioner outage timeline).
- `discussions/` — **gitignored, local-only**. Holds the plan (`02-*`), status
  (`03-*`), launch schedule (`04-*`), prior comparison notes (`05-*`), physics
  primer (`06-*`). A fresh agent won't see these in `git status` but they exist
  on disk and are the source of truth for phase-level context.
- `src/` — all code.
- `report/`, `week-*/`, `weekly-presentation/` — reporting artifacts.

**Single architecture:** the codebase was deslopped to one model — the **mass-shell
Lorentz-equivariant GNN, variant a** (`geometric_state=readout_only`, no global pooling).
The legacy/tangent-attention backbones, poincaré/euclidean geometries, and GNN variant
branches were removed. `ModelConfig` is just `{n_hidden, n_layers, regulator_mass,
use_reference_vectors, include_mass_condition}`.

**`src/` (code):** entrypoints stay at the root (so `torchrun train.py` is unchanged);
library code is grouped by concern.
- `train.py` — DDP training entry.
- `infer.py` — inference + metrics entry.
- `generate_samples.py` — shared sampling helper (`sns.set_style("whitegrid")`).
- `data.py` — JetNet download / preprocessing.
- `config.py` — pydantic run configs + YAML loader + `--set` dotlist merge.
- `prepare_stage1.py` — resolves/builds the Stage-1 jet-attribute artifacts.
- `models/` — `mass_shell_gnn.py` (the model), `LEFT_JeN.py` (thin checkpoint-compat
  factory), `stage1/` (jet-attribute NF: `jet_attr_model{,_v2,_v3}.py`).
- `util/geometry/` — `minkowski_utils`, `mass_shell`, `coordinates`, `online_coupling`,
  `conditioning`.
- `util/data/` — `distributions`, `fpnd_input`, `jet_attributes`.
- `util/metrics/` — `metrics`, `ks`, `eval_report`, `tail_diagnostics`, `qualification`.
- `util/infra/` — `ema`, `lr_schedule`, `rng`, `checkpoint_config`, `file_management`.
- `util/viz.py` — vector-field / training viz.
- `training/mass_shell.py` — the one flow-matching loss (`training/__init__.flow_matching_loss`).
- `configs/*.yaml` — one file per experiment (mass-shell-gnn, width-96 grid, infer configs).
- `tests/` — pytest suite (`conftest.py` puts `src/` on the path).
- `nrp/*.yaml` — Kubernetes Jobs.
- `analysis/aggregate_grid.py` — ablation-grid table builder.
- `downloaded_output/` — populated with training-run artifacts pulled off the
  PVC. Note: also a `downloaded_outputs/` (plural) directory exists and is
  empty — the gitignore only covers the singular. **Prefer `downloaded_output/`
  for real outputs.**

## Frozen conventions (do not rediscover)

- **4-vector order** `x = (E, px, py, pz)`. Metric signature `(+,−,−,−)`, so
  `m² = E² − |p|²`. See `util/minkowski_utils.py`.
- **Mask** is float, `1.0 = real particle, 0.0 = padding`. Padding is trailing;
  real particles come first, pT-ordered.
- **Reference vectors** `ref_vectors[:, 0] = e_t = (1,0,0,0)`, `ref_vectors[:, 1]
  = jet_p4`. `R = 2`. Unmasked virtual particles in message passing;
  displacements discarded; refs are re-supplied unchanged each layer. Only real
  rows carry the residual update. Built by `build_reference_vectors(...)` when
  `model.use_reference_vectors=true`.
- **`final_scale`** — computed in `train.py` from mean per-jet norm; written to
  `train/final_scale.txt`. All momenta are divided by it before training,
  re-multiplied at sample time. Same convention in `cache_icp.py` and
  `infer.py`. It's also embedded in the checkpoint's architecture dict.
- **`jet_conditions` layout** `[onehot_type(5), n_particles(1), pT_normalized(1)]`
  = 7 dims. `GlobalEmbedding(include_pt=True)` in `LEFT_JeN.py` consumes this.
- **Imports** are rooted at `src/` (see `src/tests/conftest.py`): use
  `from util.minkowski_utils import ...`, **not** `from src.util...`.

## Physics context (see `discussions/06-physics-primer.md`)

Jet constituents are idealized as **massless** — healthy generated particles
should be near-lightlike (`m² ≈ 0`), positive-energy, collimated around a jet
axis. Watch spacelike (`m² < 0`) fraction — anything above a few percent is the
model exploiting an unphysical channel. Full failure-mode checklist and metric
scale-sanity for g30 are in §6 of the primer.

## Config system

Every entry point takes `--config <path> [--set key.path=value ...]`. Backed by
pydantic models in `src/config.py`:

- `TrainRunConfig`, `InferRunConfig`, `CacheRunConfig`, composed of
  `DataConfig` / `ModelConfig` / `TrainingConfig` / `InferenceConfig` /
  `PathConfig` / `CacheConfig`.
- All models are `extra="forbid"` — a typo'd YAML key hard-fails validation.
  Do not "quietly ignore unknown fields."
- `--set` values are parsed with `yaml.safe_load` — `true`, `50`,
  `[g, q]` all just work. Quote lists: `--set data.jet_types="[g]"` (bare
  `--set data.jet_types=g` becomes the string `"g"` and will break downstream).

**Checkpoint embedding.** `train.py` writes `full_config = cfg.model_dump()`
into every checkpoint alongside weights. `infer.py` reads `ckpt["full_config"]`
to reconstruct model architecture flags automatically — the eval-grid job
(`src/nrp/as-jet-eval-grid-phase1.yaml`) relies on this via `jq` on
`train/summary.json`.

## NRP deployment pattern

Every job in `src/nrp/*.yaml` shares this skeleton:

- `kind: Job`, `restartPolicy: Never`, `backoffLimit: 0`,
  `ttlSecondsAfterFinished: 3600`.
- PVC `as-jet-train-pvc-2` mounted at `/mnt/data` (RWX rook-cephfs).
- Image `python:3.10-slim`; `POD_UID` from downward API.
- Resources: 1× `nvidia.com/a40`, 16Gi memory, ~2 CPU.

Entrypoint always:

1. `apt-get` build tools.
2. `git clone` the repo to `/tmp/repo`.
3. **`git checkout $GIT_COMMIT`** — the frozen-harness gate. Training YAMLs hard
   `exit 1` if `GIT_COMMIT` is still the `REPLACE_WITH_FROZEN_COMMIT_SHA`
   placeholder. Every Phase ≥1 run must come from one pinned SHA (plan §0.4).
4. Copy `src/` to `/mnt/data/src/`.
5. Reuse cached `/mnt/data/python-env` if present; otherwise create venv +
   `pip install torch --index-url https://download.pytorch.org/whl/cu121` +
   `pip install -r requirements.txt` + optional `torch_geometric` and PyG
   wheels (non-fatal — see gotchas).
6. `torchrun --standalone --nproc_per_node=$NGPU train.py --config
   configs/<x>.yaml --set paths.output_path=$OUTPUT_PATH ...`
7. Outputs land at `/mnt/data/output/<date>_<time>--$POD_UID-<label>/{train, ...}`.

Reference examples:

- Training: `src/nrp/as-jet-train-job-g30-phase1-a.yaml`.
- Eval iterating over all six phase-1 train dirs: `src/nrp/as-jet-eval-grid-phase1.yaml`.
- ICP cache build: `src/nrp/as-jet-cache-icp-30.yaml`.

## Frozen commits currently in use

- **Training**: `ba09a38` — Phase-1 A–F grid (200 epochs, gluon-30, ICP off,
  1 GPU each).
- **Eval**: `d7cec25` — fixes the prior-mismatch bug in `generate_samples`, adds
  a t-clip / value-clamp guard against pT blow-ups, and adds gen-vs-data 2-sample
  KS on jet-axis cos θ / φ.

Verdicts from the training grid are captured in the memory system
(`phase1-grid-results.md`) and summarized in
`discussions/03-implementation-status.md`. Downstream launch waves are enumerated
in `discussions/04-launch-schedule.md`.

## Tests (`src/tests/`)

Pytest suite, no `pytest.ini` — `conftest.py` inserts `src/` on `sys.path`. Run
locally with:

```
pytest src/tests
```

**`tests/test_equivariance.py` is a blocking gate** per plan §0.2: no Phase ≥1
training run is valid unless this test passes on the frozen commit. Enforced by
convention (not by CI). Also worth knowing: `test_reference_vectors.py`,
`test_mass_shell*.py`, `test_geodesic_icp.py`, `test_priors.py`,
`test_sampler.py`, `test_physicality.py`.

## Common commands

- **CPU tests**: `pytest src/tests`
- **Inspect a downloaded run**: files under
  `src/downloaded_output/<run>/train/`: `summary.json` (has `full_config`,
  `final_loss`, `metrics{}`), `metrics.csv`, `samples_subset.pt`,
  `training_loss.csv`, `physicality.png`, `isotropy.png`,
  `distribution_comparison.png`.
- **Aggregate a Phase-1 grid** (locally, on downloaded outputs):
  `python -m analysis.aggregate_grid --base <dir> --out <dir>`
  or `--runs A=<pathA> B=<pathB> ...`. Writes `ablation_table.md` + `.csv`.
- **Launch an NRP job**: edit `GIT_COMMIT` in the yaml, then
  `kubectl create -f src/nrp/<file>.yaml -n cms-ml`. Check status with
  `kubectl get pods -n cms-ml -l job-name=<name>`.
- **Read the plan**: `discussions/23-status-post-fpnd-fix.md` (current status),
  `discussions/02-experiment-plan.md`,
  `discussions/03-implementation-status.md`,
  `discussions/04-launch-schedule.md`,
  `discussions/06-physics-primer.md`.

## Gotchas

- **`discussions/` is gitignored.** It won't show in `git status` but exists
  locally and is load-bearing context. Do not assume it's absent because it's
  not in `git ls-files`.
- **`src/downloaded_output/` (populated) vs `src/downloaded_outputs/` (empty).**
  The gitignore rule only covers the singular form. Use the singular.
- **`accumulation_steps` is not divided by `world_size`** in `train.py` → under
  DDP the effective batch is `world_size × target_batch_size`, not
  `target_batch_size`. Keep the Phase-1 comparability grid at 1 GPU.
- **Cached `python-env` on the PVC** is reused across pods. If you add a
  dependency to `requirements.txt`, either delete the cached env or `pip
  install` on-pod before the training step, or the new dep won't be installed.
- **Pydantic `extra="forbid"`** — an unknown YAML key or a stale field name
  hard-fails validation. Deleted config fields still error until removed from
  the yaml.
- **`InferDataConfig` defaults to `num_particles=30`** while training YAMLs
  often use 150; easy to miss on 150-particle inference runs — set it
  explicitly.
- **FPND is optional.** `torch_geometric` + PyG wheels install is non-fatal in
  the NRP job (`|| echo WARN`). Missing FPND columns in `metrics.csv` are
  expected on some runs; don't treat as a bug.
- **`EVAL_SEED`** in `util/metrics.py` fixes the random jet-φ assignment used by
  the isotropy figure so cross-run comparisons are reproducible. Do not reseed
  it inline in eval code.
- **Frozen-harness gate is real.** Training YAMLs will `exit 1` if `GIT_COMMIT`
  is still the placeholder. This is intentional — do not comment it out.
