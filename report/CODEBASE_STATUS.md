# Codebase overview

Plain-language description of what this repository does, how the pieces connect, and what was added or heavily reworked from roughly **October 2025** onward. No dependency on slide decks or external docs.

---

## What problem this code solves

The project trains a **generative model** for **jets** as **unordered sets of particles** (four-momenta), using **conditional flow matching** on the **JetNet**-style pipeline: download or load jets, represent each jet as up to **N** particles with a **mask**, and learn a **time-dependent vector field** that transports a **simple prior** toward **real jet momenta**, conditioned on **jet-level information** (flavor, multiplicity, jet transverse momentum, etc.).

The neural network is **Lorentz-equivariant** (built from pairwise Minkowski invariants), so the field respects the symmetries of four-momentum space in the way the architecture encodes them.

---

## Main programs and how they chain together

**Typical cluster or full run:**

1. **`data.py`** — Fetches/prepares data and writes pickles (`x_train`, `x_test`, masks, jet metadata) under an `--output_path`.

2. **`jet_attr_model.py`** — Trains a **normalizing flow** on **jet-level** features (used when you need to **sample** coherent jet attributes at inference). Training reads the same `output_path` layout.

3. **`cache_icp.py`** (optional but standard for “ICP” experiments) — For each training jet, runs an **iterative closest point** style alignment between a **prior** cloud and the **target** cloud: finds a **permutation** of particles and a **rotation** of spatial momenta that reduces mismatch. Results are stored in **`icp_cache.pkl`** (permutation indices and rotation matrices), usually on a **shared cache directory** (e.g. PVC mount) keyed by jet types and `num_particles`.

4. **`train.py`** — The main trainer: loads particle tensors and jet conditioning, optionally loads **ICP cache**, builds **`LEFTJeN`**, optimizes flow-matching loss with **masked** particles, optional **classifier-free guidance** dropout, learning-rate schedules, and optional **distributed** training. Saves checkpoints, loss curves, optional vector-field visualizations, and can run **metrics** on generated samples.

5. **`generate_samples.py`** — Loads a trained **`LEFTJeN`** and the jet-attribute flow; integrates the ODE from the prior to produce **fake jets**; can write **p_T** sanity plots.

6. **`util/metrics.py`** — Compares generated vs real jets (histograms, jet-level summaries) for evaluation.

**Utilities:** **`infer.py`** drives vector-field visualization modes; **`viz_icp.py`**, **`viz_lr.py`**, **`viz_curriculum.py`** are standalone plotting/animation helpers. **`local.sh`** runs a **short local pipeline** (data → jet NF → ICP cache → train) for smoke tests.

---

## Training in more detail (`train.py`)

**Representation.** Particle data live in **Cartesian** four-vectors (normalized by a scale computed from **real** particles only so padding does not shrink the scale). You can choose whether **interpolation** between prior \(x_0\) and data \(x_1\) happens in **Cartesian** or **polar** (`--train_space`) before converting back for the network as needed.

**Prior.** The default prior is an **isotropic centre-of-mass** cloud: random back-to-back pairs so total three-momentum is zero, then rescaled to a stable magnitude. Each training step draws a **new** \(x_0\) unless you disable that path (the old “cache full \(x_0\) tensors” approach was abandoned—see ICP below).

**ICP.** If a cache is present, the code applies the stored **permutation** to \(x_0\), then the stored **rotation** to the **spatial** part of the four-vectors. That way training sees **aligned** particle ordering relative to \(x_1\) without fixing a single random draw of \(x_0\) forever (which would encourage memorizing one trajectory per jet).

**Time \(t\).** You can sample \(t\) **uniformly**, with a **power-law** bias, or **lognormal** (`--time_sampling`). A flag (`--use_time_sampling` / `--no-use_time_sampling`) forces uniform \(t\) for ablations.

**Loss.** Flow-matching loss is computed with **masks** so padded slots do not dominate; normalization accounts for **how many real particles** each jet has so dense and sparse jets are on a more comparable scale.

**Classifier-free guidance.** During training, jet conditioning is sometimes replaced with a **learned “null”** representation (and dropout is designed so **particle count** stays meaningful where intended). At **sample** time, **`generate_samples`** defaults to **not** using CFG unless you opt in; vector-field demos can still compare guided vs unguided fields.

**Optimization.** **AdamW** with fixed base learning rate (**6e-4** in code), **weight decay**, **cosine annealing with warm restarts**, and optional **linear warmup** over the first several epochs. **`--resume_weights`** continues model, optimizer, and scheduler.

**Curriculum.** Optional **reweighted sampling** by jet **multiplicity** (buckets + power-law exponent that decays to uniform). It is **on by default** in argparse but **many Kubernetes job YAMLs pass `--no-use_curriculum`** because curriculum runs have been **unstable** in practice—treat curriculum as a **research switch**, not necessarily what you ship to the cluster.

**Hyperbolic mode.** With **`--use_hyperbolic`**, the same pipeline can use a **Poincaré-ball** construction: map features to the ball, follow **geodesic** interpolation and a **Riemannian-style** loss branch (with **`pushforward`** from Cartesian network outputs into the tangent space). Default is **off**; Cartesian/polar FM is the usual path.

**Distributed.** **`torchrun`** / **`--distributed`** wraps the model in **DDP** and shards batches appropriately.

---

## The network (`models/LEFT_JeN.py`)

**LEFTJeN** is a stack of **Lorentz-equivariant layers**:

- **Time** is embedded (random Fourier–style features) and kept available at each layer.
- **Jet** information (one-hot type, normalized multiplicity, and optionally **jet p_T**) passes through an MLP into an initial **global** vector **`g⁰`**, with **LayerNorm** so its scale matches later normalized states.
- Each layer forms **pairwise messages** from **Minkowski** distance and inner product (via **ψ**-compressed features), **global** embeddings, and time.
- Messages are **gated** (scalar weights in **[0, 1]**) before aggregating to update the global state.
- **Particle positions** update via a **learned scalar** times a **normalized displacement direction** between pairs; a **learnable scale** modulates how large updates can be. **Residual** connections across layers are optional (`--use_residual`).

The forward pass outputs the **velocity / update field** demanded by the chosen FM formulation (standard branch vs hyperbolic branch in `train.py`).

---

## Supporting modules

| Module | Purpose |
|--------|---------|
| **`util/distributions.py`** | Prior sampling, **`time_dist`**, **`hyperbolic_interpolant`** |
| **`util/hyperbolic.py`** | Poincaré operations, geodesics, **`pushforward`**, **`hyperbolic_loss`** |
| **`util/coordinates.py`** | Rel ↔ Cartesian transforms and Jacobians for polar training |
| **`util/jet_attributes.py`** | Masks, conditioning vectors, sampling helpers for jet NF |
| **`util/minkowski_utils.py`** | Minkowski norm and dot products for invariant features |
| **`util/viz.py`** | Vector-field movies / figures around the trained field |
| **`util/mask_helpers.py`** | Masked statistics helpers |
| **`util/boost_equiv.py`** | Lorentz boost helpers; **not** used on the default training path (calls are commented out in `train.py` / `viz.py`) |
| **`util/file_management.py`** | Output directory hygiene |

**`archive/`** holds older experiments (**`old_train.py`**, **`FMLorentzNetNew.py`**, etc.) for reference, not the active path.

---

## Cluster jobs (`src/nrp/`)

YAML manifests define **Kubernetes Jobs** that clone the repo onto a volume, install **PyTorch** + **`requirements.txt`**, run **`data.py` → jet_attr_model → cache_icp (often) → train.py`** with explicit flags. Names containing **`icp`** are the **mainline** full pipeline with **shared `--cache_dir`** for **`icp_cache.pkl`**. There are variants for **30 vs 150** particles, **gluon-only vs multi-flavor**, **hyperbolic**, and **DDP**; some YAMLs still mention **curriculum** but operationally **ICP jobs tend to disable curriculum** for stability.

---

## Additions and major changes since ~October 2025

These are the big movements visible in **git history** and the **current tree** relative to an older “minimal FM + LEFT” baseline:

- **Polar training path** (`--train_space`) and **non-uniform** training-time sampling (`power_law`, `lognorm`) with ablation flags.
- **Cosine warm restarts**, **warmup**, **AdamW**, **6e-4** LR and revised **eta_min** / **weight decay** (iterating past earlier smaller gains and schedules).
- **Masked scale** and **masked, multiplicity-normalized loss**; removal of fragile **optional mask-off** behavior.
- **Curriculum learning** (bucketed multiplicity + decaying exponent), still in code but **often turned off** in deployed jobs.
- **ICP pipeline**: **`cache_icp.py`**, **`perm_cache` + `rot_cache`**, **`canonical_cache_path`**, shared PVC caches; removal of **cached full-coordinate** \(x_0\) in favor of **fresh prior + alignment**; **restore-to-best** behavior inside ICP iterations; **`viz_icp.py`**.
- **Jet p_T** in **`GlobalEmbedding`** and conditioning stack; **per-sample CFG dropout** and **learned null** / **`make_null_cond`**; deletion of standalone **`util/cfg.py`** in favor of logic on the model.
- **Architecture refinements**: wider embeddings (**128**), **Sigmoid** gating on message scalars, **linear** displacement head (no Tanh cap), **LayerNorm** on messages with **mask-aware** ordering, **Generator**-scoped RNG for time embedding, **persistent** Fourier buffer, Xavier **gain 1** and **zero** MLP biases.
- **Optional hyperbolic FM** (`util/hyperbolic.py`, `--use_hyperbolic`).
- **DDP** training path and **`--resume_weights`**.
- **Richer metrics** (relative/absolute kinematic histograms, jet-level plots, **p_T** comparisons) and **`infer.py`**; **`viz_lr.py`**, **`viz_curriculum.py`**.
- **`local.sh`** for the remaining local workflow.
- **NRP YAML** set expanded/renamed (ICP, hyperbolic, particle counts); **per-mode YAML duplication** reduced in favor of **CLI flags**.
- **Lorentz COM boost** experimented with then **removed from the default training path** (code left commented, **`boost_equiv`** still in repo).
- **Notebook / ancillary churn** (e.g. large `download.ipynb` updates) and removal of an old **jetnet tutorial** path—noise for line-count stats, not core logic.

For commit-level narrative and rationale citations (including Claude session notes), see **`NET_CHANGES_SINCE_OCT_2025.md`** and **`CLAUDE_ARCHITECTURE_JUSTIFICATIONS.md`** in this same `report/` folder.
