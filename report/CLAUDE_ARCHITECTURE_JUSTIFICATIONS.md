# Claude project logs: architecture and structural justifications

Synthesized from local Claude Code session transcripts under:

`/Users/arjunsharma/.claude/projects/-Users-arjunsharma-development-s25-fermilab-research-src`

**Excluded from this dump:** Lorentz/boost discussions and YAML / Kubernetes deploy threads.

**Primary sessions referenced:**

| Session file | Role |
|--------------|------|
| `37ebf2cc-602f-46ee-8e0c-22c64998bfe8.jsonl` | Main model, training, curriculum, ICP evolution, hyperbolic, activations |
| `de77672c-42ab-49b6-bf3a-189993cb6e7f.jsonl` | ICP overfitting → perm-only cache → perm + rotation cache |
| `e68232d3-8b8e-4bfb-ab5a-2f221a3a7273.jsonl` | CFG / null conditioning, `GlobalEmbedding` LayerNorm, `make_null_cond` |

---

## `LEFT_JeN` / model architecture

### RNG in `TimeEmbedding`

`torch.manual_seed` in `__init__` was resetting the **global** PyTorch RNG whenever the model was constructed, overriding `train.py` seeding.

**Fix:** use an isolated `torch.Generator` for the random Fourier projection.

### Projection buffer `persistent=False`

Projection weights were not in `state_dict`, so checkpoints could disagree with training unless the hardcoded seed never changed.

**Fix:** `persistent=True` so loads match exactly.

### `normalized_diff` denominator (Minkowski vs Euclidean)

`normsq4` can be **negative** for spacelike pairs; after `clamp(min=1e-8)` the denominator collapses and **does not normalize**, so displacements can blow up.

**Proposed fix in the same code review:** use **Euclidean** `diff.pow(2).sum` in the denominator (verify in-repo whether this landed).

### LayerNorm on messages vs mask

Applying LayerNorm **before** zeroing masked pairs lets padding pairs (still receiving signal from `g0` / `t_emb`) **distort** mean/variance for sparse jets.

**Fix:** mask → LayerNorm → mask again (re-zero after norm so bias does not leak into padded pairs).

### `gain=0.1` + `bias=0.1`

Tiny weights but **nonzero biases** meant initial `phi_x` / `phi_m` outputs were **bias-dominated**, giving a **non-trivial velocity at step 0**.

**Fix:** zero biases; later move Xavier **gain to 1.0** so the network is not stuck in a near-zero activation regime.

### `repeat` → `expand` in `step`

Use **broadcasting without copying** where appropriate (observation 17 in the implementation pass).

### Duplicate `g0_exp` / `t_emb_exp`

Removing redundant expansion (issue 12).

### `phi_m`: Tanh → Sigmoid

For a **gate** on pair contributions to the global sum, Tanh allows **negative** weights → **cancellation** and noisy gradients. Sigmoid is **[0, 1]** “how much does this pair matter?” — closer to soft attention.

### `phi_x`: remove Tanh

Tanh **caps** displacement at `±gamma`; **sparse** jets may need **larger** corrections. Use **linear `phi_x`** with learnable `gamma` and existing `normalized_diff` to bound direction; rely on gradient clipping for safety.

### Curriculum vs “difficulty”

Discussion contrasts **GNN mechanics** (dense jets → many pairs → hard selective gating) vs **empirical** ease (dense jets often easier to fit). Curriculum “easy” should mean **easy low loss early** → **dense-first** aligns with that hypothesis; open question whether dense-trained gating **transfers** or **hurts** sparse jets (empirical follow-up).

### `GlobalEmbedding` and CFG-related conditioning (`e68232d3`)

- **`g0` without normalization:** `g_prev` was LayerNorm’d but **`g0` was raw**, so **scale mismatch** in `cat([g0, g_prev, t_emb, …])` could weaken conditioning. **Fix:** `LayerNorm` on `GlobalEmbedding` output.

- **All-zero null vector:** Ambiguous with real low-pT / few-particle jets. **Fix:** learned `null_cond` parameter plus `make_null_cond` that nulls **type + pT** but keeps **real `n_particles`**, since count is already encoded in the **mask** and should not be the main axis CFG differentiates.

- **FiLM (outlined, not required for minimal fix):** using `g0` to predict scale/shift could strengthen conditioning vs pure concatenation; Lorentz considerations differ for global vs 4-vector modulation.

---

## Training / loss / schedules

### Scale from masked particles only

Padding zeros **deflate** std if included in the global scale computation.

### Masked loss + curriculum

Plain `.mean()` over tensor slots makes **dense jets contribute more total loss**; curriculum oversampling dense jets can **inflate** the loss curve without true instability.

**Fix:** normalize by **number of real particles**; unify masking (remove `--mask` footgun so loss is **always** masked).

### Cosine LR vs curriculum

**Anti-alignment:** easy = dense early, hard = sparse late, while cosine gives **large steps early** and **small steps late** — smallest updates when sparse jets matter most.

**Mitigations discussed:** warm restarts, delayed cosine start, higher `eta_min`. Implementation moved toward **CosineAnnealingWarmRestarts** with **`T_0 = num_epochs // 4`** (four restarts per user preference) because difficulty **ramps gradually**, not at a single phase boundary.

### Weight decay vs curriculum

Informal argument: decay is **roughly constant** while **per-step gradient signal** changes with jet density, so **effective regularization** shifts across curriculum; try **lower** `weight_decay` (e.g. `1e-6`).

### `eta_min` and base LR

Raising the cosine floor and using **3e-4** base LR (with gain `1.0`) justified as keeping **usable LR in late training** and matching **stronger initialization**.

---

## Prior / `gen_initial_distribution`

### Odd `num_particles`

Explicit slice bounds so the **last slot stays zero** and is **masked**, avoiding shape mismatches.

### Per-tensor std rescale to target 1

Aligns prior scale with normalized data to avoid **large initial flow displacements** (also reflected in the LaTeX summary of the prior in the same session).

---

## ICP pipeline (structural)

### Cached full `x_0` (original design)

Risks:

1. **Trajectory memorization** — same `(x_0, x_1)` every epoch; model never sees the full marginal over priors.
2. **Train / infer shift** — inference uses **fresh** unaligned draws.
3. **Fixed coupling** — CFM-style training should sample the coupling, not pin one draw.

**Fix:** cache **permutation indices** (and later **rotation**), sample **fresh `x_0`**, then **apply** stored perm (+ rotation on 3-momentum). Preserves **alignment** without memorizing a single cloud.

### Restore-to-best ICP

ICP can **oscillate**; the **last** iterate may be worse than an **intermediate** state. Track **best objective** and store **best permutation / rotation** (and surface in visualization).

### Full Π + R in cache

After permutation-only ICP, logs describe **alternating Hungarian + Kabsch** (Algorithm 3 style) for stronger alignment, with **`icp_max_iter`** to bound compute vs very large iteration counts.

---

## Hyperbolic / Riemannian flow matching

### Massless kinematics

On-shell geometry is **light cone**, not a clean hyperboloid embedding. Use **Poincaré ball** with a **pragmatic** map from 4-vectors.

### Euclidean radius in the tanh bijection

**Minkowski** squared norm is **~0** for massless particles — not usable as a radial coordinate. **Euclidean** norm on `(E, p_x, p_y, p_z)` is the deliberate choice for the radial compression into the ball.

### Training flow

Geodesic interpolant + conditional field in the ball; **`pushforward`** maps Cartesian network output into the **tangent space** for the loss target.

### Huge losses / spikes

**Conformal factor** \(\lambda_x^2\) blows up near the **boundary** of the ball; normalized momenta can sit in the **saturated** region of `tanh`, so loss is dominated by **metric weighting** and outliers.

**Options discussed:** lower curvature **`c`**, or **drop Riemannian weighting** and use **Euclidean** error on tangent vectors while **keeping** geodesic paths — argued as more **stable** when data already lives in the bad regime of the map.

---

## Metrics and visualization utilities

### GPU metrics

Tensors on **CUDA** must be moved **`.cpu()`** before numpy / jetnet code paths.

### Scripts

- **`viz_lr`**, **`viz_curriculum`** — inspect schedules and bucket weights against real `x_train` counts.
- **`viz_icp`** — 2D / 3D / panel animations, matching lines, distance colormaps, restore-to-best framing.

---

## Curriculum sampling caveat (with replacement)

With `torch.multinomial(..., replacement=True)` and `epoch_frac=1.0`, an “epoch” draws `N_train` samples **with replacement**, so **unique jets per epoch** can be **well below** `N_train` (birthday effect), especially under dense-heavy weights — unlike `randperm` without replacement. Sparse jets may be **skipped** for entire epochs at high \(\alpha\).

---

## Gaps in the logs

The transcripts do **not** always give a long explicit rationale for every numeric default (e.g. `embed_dim` 128 vs 64). Most reasoning is **stability, bugs, conditioning, and distribution shift**.

---

*Generated for the s25-fermilab-research project from local Claude session exports.*
