# Net changes since October 2025 (comprehensive report)

This document merges **git history** (net diff versus pre–October baseline) with **engineering rationale** where it exists. Most detailed justifications come from **Claude Code** session transcripts under `~/.claude/projects/-Users-arjunsharma-development-s25-fermilab-research-src/`; items without that provenance are labeled **fact (git only)** or **inferred**.

**Baseline for “net”:** last commit before October 2025 activity is `ca0d8a5` (2025-08-29, “Minor refactor”). First commit in the October window on `main` is `84a1909` (2025-10-18). Unless stated otherwise, “HEAD” means the repository state at the time this report was written.

**Companion docs:** [CLAUDE_ARCHITECTURE_JUSTIFICATIONS.md](./CLAUDE_ARCHITECTURE_JUSTIFICATIONS.md) (rationale extract), commit narrative in Cursor plan `commit_history_report_e6650721.plan.md`.

---

## How to read the tags

| Tag | Meaning |
|-----|---------|
| **Claude** | Reasoning appears in exported Claude transcripts (see justifications file for session IDs). |
| **Git** | Established from commits/diff only; no matching rationale in those transcripts. |
| **Claude + Git** | Claude argued for the change; git confirms it shipped (possibly with later tweaks). |

---

## 1. Executive summary

Since October 2025 the codebase evolved from a simpler Cartesian flow-matching trainer with COM-boost hooks and a smaller `LEFT_JeN` into a system with **optional hyperbolic (Poincaré) flow matching**, **curriculum sampling by particle multiplicity**, **ablation flags** for cosine LR and time sampling, **ICP alignment** via cached **permutation and rotation** applied to **fresh** prior draws, **stronger conditioning** (jet type, particle count, jet \(p_T\)), **classifier-free guidance** with a **learned null** and **careful dropout**, richer **metrics and visualization**, and **distributed training** support. Several experiments were **reverted or superseded** (fixed cached \(x_0\), Lorentz boost in the default path, separate YAMLs per polar/power-law mode, standalone `util/cfg.py`, gradient-plotting noise).

Raw `git diff ca0d8a5..HEAD --stat` line counts are **dominated by `src/download.ipynb`** (large notebook churn in the “Clean” commit); treat that as **data/output noise**, not core logic surface area.

---

## 2. Timeline (high level)

| Period | Themes |
|--------|--------|
| **Oct 2025** | Debugging, NRP polish, gradient tracking, tensor/training bugfixes, brief Lorentz boost experiment then removal from active path. |
| **Nov 2025** | `--train_space` (cartesian / polar), `time_dist` and training-time \(t\) sampling modes; job YAML churn. |
| **Dec 2025** | Large `download.ipynb` + presentation assets + small model touch-ups. |
| **Feb–Mar 2026** | Mega-feature batch: curriculum, ICP cache evolution, cosine warm restarts, masking/loss fixes, activations, hyperbolic FM, CFG/null refactor, metrics/viz, LR and time-sampling refinements. |

---

## 3. Training (`train.py` and related)

### 3.1 Learning rate and schedulers

**Net state:** fixed base **`lr = 6e-4`**, **`AdamW`** with **`weight_decay = 1e-6`**, **`CosineAnnealingWarmRestarts`** with **`eta_min = lr * 0.3`**, **`T_0`** defaulting to **`num_epochs // 4`** (or related heuristic when epochs are small), plus **`LinearLR` warmup** for **`lr_warmup_epochs`** (default **10**) composed with cosine via **`SequentialLR`**.

- **Claude + Git:** Warm restarts and a higher cosine floor were motivated by **misalignment** between **dense-first curriculum** (easy → hard by your hypothesis) and **monotonic cosine decay** (large steps early, small steps late): the hardest regime (sparse jets) saw the smallest learning rates. **Mitigations:** warm restarts, higher `eta_min`, later **`T_0 = num_epochs // 4`** per your preference for **gradual** difficulty rather than a single phase transition.
- **Claude + Git:** **`weight_decay = 1e-6`** discussed as reducing tension between **uniform weight decay** and **non-i.i.d. curriculum sampling**.
- **Git:** The move from earlier values (e.g. **3e-4** and **`eta_min = 0.1 × lr`**) to **6e-4** and **`0.3 × lr`** plus **10-epoch warmup** is visible in commits (`8deebdd`, `93b2ecf`, etc.) but the **final numeric choice** is not spelled out in the same detail in the exported Claude summary doc—treat as **tuned after** the main Claude LR discussion.

**Flags:** `--use_cosine_lr`, `--lr_t0`, `--lr_warmup_epochs` (BooleanOptionalAction defaults).

### 3.2 Curriculum learning

**Net state:** Jets bucketed by **particle count**; per-epoch sampling weights \(P(\text{bucket}) \propto (k+1)^\alpha\) with \(\alpha\) decaying linearly from **`curriculum_alpha_start`** (default **2.0**) to **0**; **`torch.multinomial` with replacement** over **`samples_per_epoch`** indices.

- **Claude + Git:** **Dense-first** schedule matches the hypothesis that **high-multiplicity jets are easier** early (cleaner loss signal); formal note that **GNN mechanics** (many pairs → harder gating) and **empirical ease** can point in opposite directions—curriculum should track **easy by loss**, not only by graph size.
- **Claude:** **Replacement sampling** implies an “epoch” of length \(N\) is **not** “each jet exactly once”; **unique jets per epoch** can be \(\ll N\) (birthday effect), and sparse buckets can be **skipped** for whole epochs at high \(\alpha\).

### 3.3 Time sampling

**Net state:** **`--time_sampling`**: `uniform` | `power_law` | `lognorm`; **`--use_time_sampling`** toggles whether non-uniform modes apply. When `power_law` is active, code passes **`a = -0.2`** into `time_dist`. Commits mention **“flip focus of time dist”** and **“fix time sampling completely”** (**Git**).

### 3.4 Coordinate space

**Net state:** **`--train_space`** `cartesian` | `polar` for interpolation / training geometry (**Git**, Nov 2025).

### 3.5 Scaling, masks, and loss

**Net state:** **Global scale** (`final_scale`) computed from **real particles only** (mask channel), not padding. **Loss** normalized by **real particle count** (divide by \(n_{\text{real}} \times 4\) style); **`--mask` flag removed** so masked loss is not optional.

- **Claude + Git:** Padding in the scale statistic **deflates** std. Plain `.mean()` over tensor slots **overweights dense jets**; combined with curriculum oversampling dense jets, loss curves can **look** unstable without true divergence.
- **Claude + Git:** **`--mask` with `type=bool`** was a **footgun**; masking is always applied when mask data exists.

### 3.6 Classifier-free guidance (training)

**Net state:** Per-sample dropout on conditioning; null built via **`make_null_cond`** (learned null for type/\(p_T\)-like slots, **preserve true particle count** where intended). **`cfg_null_dropout_rate`** CLI.

- **Claude + Git:** All-zero null **collides** with **legitimate** low-\(p_T\) / few-particle jets. **Learned `null_cond`** separates “no conditioning” from “weak conditioning.” **Keeping \(n_\text{particles}\)** in the null vector (while masking still encodes count) makes the **conditional vs unconditional gap** about **type and \(p_T\)**, not redundant mask information.

### 3.7 ICP cache usage

**Net state:** Load **`perm_cache`** and optional **`rot_cache`**; each step **draw fresh** \(x_0 \sim\) prior, then **permute** and apply **3-momentum rotation** as cached. Old **`x_0_cache`** format rejected with a clear error.

- **Claude + Git:** Caching **fixed** \(x_0\) per jet causes **trajectory memorization**, **train/inference shift** (inference uses random unaligned priors), and **wrong coupling** for flow matching. Caching **only alignment** (perm ± rot) restores **variance** in the prior while keeping **ordering** benefits.

### 3.8 Hyperbolic flow matching (optional)

**Net state:** **`--use_hyperbolic`**; **`hyperbolic_interpolant`**, **`pushforward`**, **`hyperbolic_loss`** branch.

- **Claude + Git:** Massless kinematics sit on the **light cone**; Poincaré ball used with a **pragmatic radial map**; **Euclidean** radius on \((E, p_x, p_y, p_z)\) because **Minkowski norm is \(\approx 0\)** for massless particles—cannot serve as radial coordinate. **Conformal factor** \(\lambda_x^2\) can **explode** near the ball boundary; transcripts discuss **lowering curvature** or **softer / Euclidean tangent loss** while keeping **geodesic paths**.
- **Claude:** Combining **CFG in Cartesian** with **hyperbolic training** is **geometrically sloppy** unless guidance is applied in the correct tangent space (noted as a design tension).

### 3.9 Distributed training and resume

**Net state:** **`torchrun` / `--distributed`**, DDP wrapping, strided index sharding for curriculum diversity across ranks (**Git**). **`--resume_weights`** loads model, optimizer, scheduler (**Git**).

### 3.10 Other training housekeeping

- **Git:** Commits for **explosion guards**, **weight restore**, **inner-loop print removal**, **ODE print cleanup**, **grad norm plotting removal** (after earlier gradient tracking add).
- **Git:** Script to **move old output directories** (`ff19237`).

---

## 4. Model (`src/models/LEFT_JeN.py`)

### 4.1 Width and conditioning

**Net state:** **`embed_dim`** increased **64 → 128** for time and global paths; **`GlobalEmbedding`** supports **`include_pt`**; **`train.py`** passes **`include_pt=True`** so jet **\(p_T\)** is part of the conditioning vector.

- **Claude + Git:** **`GlobalEmbedding` output `LayerNorm`** so **`g0`** is on a comparable scale to **layer-normalized `g_prev`**, avoiding **weak conditioning** in concatenations.
- **Git / transcript feature list:** **\(p_T\) in conditioning** for the forward path (not only null handling) appears in implementation summaries; the **justifications markdown** emphasized null/LayerNorm—**\(p_T\) as a first-class conditioning signal** is part of the **net** design.

### 4.2 Time embedding RNG and checkpointing

- **Claude + Git:** Replace global **`torch.manual_seed`** in **`TimeEmbedding.__init__`** with an isolated **`torch.Generator`** so constructing the model does not **clobber** training RNG.
- **Claude + Git:** **`persistent=True`** on the random Fourier projection buffer so **checkpoints reproduce** the same embedding without relying on a fixed global seed.

### 4.3 Initialization

- **Claude + Git:** **`PhiMLP` biases** → **zeros** (was **0.1**), avoiding **bias-dominated** early **`phi_x` / `phi_m`** outputs and spurious early velocity.
- **Claude + Git:** Xavier **gain 0.1 → 1.0** so the network is not stuck in a **near-zero** activation regime (especially after bias fix).

### 4.4 Message passing: LayerNorm and mask

- **Claude + Git:** Apply **pair mask before `LayerNorm` on messages**, then **re-mask** after, so padded pairs do not **skew** batch statistics; **`LayerNorm` bias** cannot leak into invalid pairs.

### 4.5 Activations: `phi_m` and `phi_x`

- **Claude + Git:** **`phi_m`**: **Tanh → Sigmoid** so gates are in **[0, 1]** (soft attention), avoiding **signed cancellation** in the global sum.
- **Claude + Git:** **`phi_x`**: remove **Tanh** so displacement magnitude is not **hard-capped** at \(\pm\gamma\); **sparse** jets can need **larger** updates; **`gamma`** and **gradient clipping** bound behavior.

### 4.6 `normalized_diff` denominator (review vs code)

- **Claude:** **`normsq4`** on pair differences can be **negative** (Minkowski); **`clamp(min=1e-8)`** breaks **real normalization** and can **blow up** displacements—review proposed **Euclidean** `diff.pow(2).sum` for the denominator.
- **Fact:** As of report writing, **`LEFT_JeN.py`** still uses **`normsq4`** in the **`normalized_diff`** construction (~lines 240–241). Treat the **Euclidean denominator** as a **documented recommendation**, not verified shipped fix.

### 4.7 Other model fixes

- **Claude + Git:** Remove **duplicate** **`g0` / `t_emb` expansions**; **`repeat` → `expand`** where appropriate in **`step`**.

### 4.8 Removed dependencies

**Net state:** No **`enforce_com_frame`** in `LEFT_JeN`; no import of **`util.cfg`** (module deleted).

---

## 5. Prior and distributions (`src/util/distributions.py`)

- **Claude + Git:** **`isotropic_com` prior**: explicit slices **`0:2*n_pairs:2`** / **`1:2*n_pairs:2`** so **odd `num_particles`** leaves a **zero pad** masked out.
- **Claude + Git:** Rescale assembled prior to **target std 1** so numeric scale matches **normalized data** and avoids **huge** initial flow steps.
- **Git:** **`hyperbolic_interpolant`** and imports for Poincaré operators when hyperbolic mode is used.
- **Git:** **`generate_x_0_com_frame`** now **`NotImplementedError`**; callers use **`gen_initial_distribution(..., prior_dist='isotropic_com')`**.

---

## 6. ICP pipeline (`cache_icp.py`, `train.py`, `viz_icp.py`)

- **Claude + Git:** Transition from **cached coordinates** to **`perm_cache`** (+ **`rot_cache`**) with **fresh** \(x_0\); **restore-to-best** over ICP iterations when the last iterate is worse than an intermediate one.
- **Git:** **`util/align_clouds.py` removed**; alignment logic lives in **cache / worker** paths. **`icp_max_iter`** and multiprocessing **chunked** workers.
- **Claude + Git:** **`viz_icp.py`**: 2D / 3D / panel modes, distance coloring, best-frame indicators.

---

## 7. CFG utilities and sampling

- **Git:** **`src/util/cfg.py` deleted** (“Crazy CFG fix”); null handling **in the model** (`make_null_cond`, learned parameter).
- **Git:** **`generate_samples.py`** defaults **`use_cfg=False`** (“Remove CFG for samples”); CFG remains available for **vector-field** paths and **`infer.py`** when requested.

---

## 8. Metrics and inference utilities

- **Git + partial Claude:** **`util/metrics.py`**: device-aware paths, **`.cpu()`** before numpy/jetnet; **histograms** (relative and absolute \(\eta\), \(\phi\), \(p_T\)), **jet-level** summaries, **global \(p_T\)** comparison plots.
- **Git:** **`infer.py`** for **vector-field** modes (cfg / nocfg / both / none).
- **Claude + Git:** **`viz_lr.py`**, **`viz_curriculum.py`** for schedule and curriculum visualization.

---

## 9. Shell scripts and local workflow

- **Git:** **`local.sh`**: multi-step smoke / local pipeline (**data → jet attribute NF → `cache_icp` → `train.py`** with documented flags).

---

## 10. NRP / configuration

- **Git:** Job YAMLs **renamed and expanded** (e.g. 30 vs 150 particles, ICP / curriculum / hyperbolic combinations). **Per-mode** polar / power-law YAMLs **removed** in favor of **`--train_space`** and **`--time_sampling`** CLI flags on shared templates.

*(YAML content and deploy details are omitted here; they are not the focus of the Claude architecture dump.)*

---

## 11. Lorentz boost (net outcome)

**Net state:** **`boost_to_com_frame`** and related calls **commented out** in **`train.py`** and **`viz.py`** with TODOs about scaling; **`util/boost_equiv.py`** remains in the repo but is **off** the default training path. **`LEFT_JeN`** does not use **`enforce_com_frame`**.

**Justification:** Intentionally **not** pulled from Claude exports in [CLAUDE_ARCHITECTURE_JUSTIFICATIONS.md](./CLAUDE_ARCHITECTURE_JUSTIFICATIONS.md); **Git** shows **add then revert** in late October 2025.

---

## 12. Other repository churn

- **Git:** **`week-2/jetnet/tutorial.ipynb`** deleted.
- **Git:** **`hash.txt`**, **weekly presentation** assets, **`.gitignore`** updates.
- **Git:** **`src/models/__init__.py`** empty package file.
- **Git:** **`src/util/mask_helpers.py`** (masked mean/std helpers for metrics or training).

---

## 13. Summary table: net structural additions

| Component | Role |
|-----------|------|
| `cache_icp.py` | Precompute **perm** (+ **rot**); canonical paths under **`cache_dir`** |
| `util/hyperbolic.py` | Poincaré ball FM operators |
| `viz_icp.py`, `viz_lr.py`, `viz_curriculum.py` | Visualization |
| `infer.py` | Inference / VF CLI |
| `curriculum.ipynb`, `icp.ipynb` | Notebooks (analysis) |
| `local.sh`, `archive_incomplete.sh` | Automation |
| `nrp/*.yaml` | Training job templates |

---

## 14. Caveats

1. **Line-count statistics** mix **notebook output** with logic; use **`git diff --stat` on `*.py` only** for a fairer view.
2. **This report is not a substitute for `git log -p`** on specific files when you need exact line-level authorship.
3. **Claude** rationale is **as stated in local transcripts**; it may be wrong or incomplete—**verify** against physics and ablations.

---

*Report generated to consolidate git net state (from `ca0d8a5` through recent `main`) with rationale from [CLAUDE_ARCHITECTURE_JUSTIFICATIONS.md](./CLAUDE_ARCHITECTURE_JUSTIFICATIONS.md) and the earlier commit narrative plan.*
