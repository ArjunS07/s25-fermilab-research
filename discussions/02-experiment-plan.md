# Experiment plan

*Written 2026-07-03. Revised 2026-07-03 (closes 9 review gaps; L-GATr head-to-head deferred).
Companion to `01-symmetry-mismatch-diagnosis.md`.*

Goal: Lorentz-equivariant (geometric) flow matching for jet generation that supersedes
L-GATr with a simpler backbone. Thesis: minimal universal LorentzNet-style backbone
+ mass-shell RFM geometry + reference-vector symmetry breaking.

Constraint kept throughout: architectural Lorentz equivariance (in the joint
inputs-and-references sense).

> **Deployment note.** Nothing runs locally. Every run is a Kubernetes Job under
> `src/nrp/*.yaml` that clones the repo and runs `train.py`/`infer.py` on the PVC. See the
> **frozen-harness gate** (0.4) — all Phase ≥1 runs must come from one pinned commit.

---

## Interfaces (frozen) — pin before any Phase-1 run

Weaker implementing agents burn days on ambiguity here, so these are fixed, not suggestions.

**Particle tensor.** `x (B, P, 4)` Minkowski `(E, px, py, pz)`, metric `(+,−,−,−)`
(`util/minkowski_utils.py`). `mask (B, P)` float, `1.0 = real, 0.0 = padding`; padding is
trailing (real particles first, pT-ordered). `jet_conditions (B, 7) = [onehot_type(5),
n_particles(1), pT(1)]`.

**Reference vectors.** `ref_vectors (B, R, 4)`, `R = 2`, order pinned:
`ref_vectors[:,0] = e_t = (1,0,0,0)`, `ref_vectors[:,1] = jet 4-momentum` — both in the
same scaled units as `x`. Train: `jet_p4 = (x_1 * mask).sum(1)` (true target jet momentum).
Infer: `jet_p4 = EtaPhiPtE_to_cartesian(jet_η, jet_φ, jet_pT, jet_E) / final_scale` from the
jet-attribute NF (φ assigned as in `util/coordinates.py`).

**Virtual-particle wiring.** Inside `forward`, augment
`x_aug = cat([x, ref_vectors], 1) (B, P+2, 4)`, `mask_aug = cat([mask, 1], 1)`. References
are **unmasked** ⇒ they both send and receive messages, and **ref–ref pairs DO enter** (2
rows, harmless, keeps the layer uniform). References are **fixed across layers**: only the
`P` real rows carry the residual update; refs are re-supplied unchanged each layer and their
displacement outputs are discarded. Velocity `= (x_particles − x0)[:, :P] * mask`.

**Per-node scalar `h_i` (LorentzNet-style hidden state, currently absent).**
`h (B, P+2, node_scalar_dim)`. Seed from a fixed 3-dim per-node invariant vector
`[ψ(⟨x,x⟩)=m_i², ψ(⟨x,e_t⟩)=E_i, ψ(⟨x,x_jet⟩)=axis]`; **with refs off (run F), columns 2–3
are zeroed** so the seed input dim stays 3 and F remains exactly rotation-equivariant (only
the true invariant m_i² is used). In each layer, `h_i`,`h_j` enter the message MLP; after
messages, aggregate `m_i = Σ_j scaled_messages / N_actual` and update
`h ← h + φ_h([h, m_i])`.

**Flags.** `--use_reference_vectors` and `--use_node_scalars` are independent
`BooleanOptionalAction`, default False ⇒ **run A is byte-for-byte the current model**.

---

## Phase 0 — Diagnostics & baselines (cheap, do first)

**0.1 Isotropy figure.** ✅ Informally confirmed (generated 3-momentum directions
~uniform). Make it a proper figure: histogram of generated total-3-momentum / principal
axis directions vs. test set (cos θ and φ marginals). This is the motivation figure for
the paper and the before/after yardstick for every fix below. (Lands in
`util/metrics.py::run_save_metrics` as `isotropy.png`.)

**0.2 Equivariance unit test — BLOCKING pytest, not a figure.** `tests/test_equivariance.py`,
pure-CPU, tiny model, fixed seed. Pass/fail assertions:
1. Joint Lorentz equivariance `f(Λx,…,Λrefs) = Λ·f(x,…,refs)` for random rotations *and*
   boosts, refs included, all flag settings.
2. **Symmetry breaking (the blocking one):** with references on, a **particles-only**
   rotation (refs held fixed) MUST change the output above threshold. This guards against
   the failure mode where an implementation error drops the reference invariants from the
   scalar path and silently reproduces run A.
3. Residual symmetry: SO(2) about the jet axis (rotate particles *and* jet-axis ref
   together) leaves invariants unchanged.
4. Run-F guard: refs off + node-scalars on ⇒ still exactly rotation-equivariant.
   No Phase ≥1 run is valid until this test passes on the frozen commit.

**0.3 Physicality diagnostics.** Fraction of generated particles with E < 0, spacelike
(m² < 0), and the m² distribution. Track for every run (`util/metrics.py`, `physicality.png`
+ columns in `metrics.csv`); mass-shell RFM should send these to exactly zero.

**0.4 Fixed evaluation harness — HARD GATE.** Freeze JetNet metrics (W1M, W1P, W1EFP, FPND,
coverage/MMD) + the diagnostics above into one script (`run_save_metrics`), fixed seeds,
fixed n_samples. **No Phase ≥1 run is valid unless produced by the frozen harness commit.**
The NRP yaml pins `GIT_COMMIT` and each run records `git_commit.txt`; runs from unpinned
HEAD are discarded, not compared.

---

## Phase 1 — Break the symmetry (biggest expected gain, smallest diff)

**1.1 Reference 4-vectors.** Add e_t = (1,0,0,0) and jet 4-momentum as unmasked virtual
particles in message passing (displacements discarded), plus their per-node invariants
ψ(E_i), ψ(⟨x_i, x_jet⟩) as scalar inputs. Wiring per **Interfaces (frozen)** above.
*Success:* isotropy figure collapses to data's axis distribution; η_rel/φ_rel/pT marginals
become fittable; loss floor drops.

**1.2 Per-node scalar channels h_i.** Seeded from ψ(m_i²), ψ(E_i), ψ(⟨x_i, x_jet⟩);
updated each layer via aggregated messages; entering the message MLP.
(No dataset attributes needed — these are derived from momenta + references.)

**1.3 Axis-aligned prior.** Collimated, positive-energy, near-lightlike prior around
the jet axis (new `prior_dist='axis_aligned'` in `util/distributions.py`, extending the
`jet_ref_frame` pattern). Does symmetry breaking via inputs suffice, or does the shorter
transport path add measurably? **Ablate the aligned prior with ICP OFF first.** An
axis-aligned prior changes both transport-path length *and* the coupling geometry; with the
Euclidean ICP coupling still on, the measured effect can flip sign. Only after the ICP-off
result is clean should a D+ICP follow-up run be scheduled.

**Ablation grid (30-particle gluon jets, fixed budget, ICP off, frozen commit):**
| run | ref vectors | h_i | aligned prior | purpose / abort threshold |
|-----|:-----------:|:---:|:-------------:|---------------------------|
| A (current) | – | – | – | baseline; isotropy figure ~uniform (expected) |
| B | ✓ | – | – | refs alone. **Abort if** axis cos θ histogram does not depart from uniform (KS vs. uniform, p<1e-3) → equivariance test / ref wiring is broken; stop and debug before spending more budget |
| C | ✓ | ✓ | – | refs + h_i. **Abort if** W1M/W1P not ≤ B within noise (h_i should not regress) |
| D | ✓ | ✓ | ✓ | full. **Abort if** worse than C → prior×coupling interaction; re-check ICP-off |
| E | – | – | ✓ | prior-only escape route — how far does it get alone? |
| F | – | ✓ | – | **h_i only (attribution run).** Isolates the node-scalar gain from references (C−B vs F−A). Must stay isotropic (still equivariant); value is loss-floor / invariant-observable improvement, not orientation |

C−B and F−A together attribute the h_i contribution; B−A and E−A attribute references vs.
prior. Without F the table cannot separate h_i from refs.

---

## Phase 2 — Training hygiene & efficiency (parallel to Phase 1, low risk)

**2.1 EMA of weights** (missing entirely; standard for FM/diffusion; near-free).
**2.2 FiLM/adaLN conditioning** of t_emb and g per node, replacing the 3×128-dim
concat into every pairwise message. Removes the (B, N, N, ~386) memory hog →
physical batch 128–256, several-times-faster epochs.
**2.3 Better sampler at eval**: Heun/midpoint vs. 16-step Euler; steps-vs-quality curve.
**2.4 Redo the LR sweep from scratch post-Phase-1.** Do **not** compare against pre-Phase-1
LR conclusions: the loss floor was symmetry-limited then, so it measured a different
hypothesis class and those conclusions must be discarded, not "re-examined." `eta_min` is
now a config knob (`--eta_min_factor`, currently 0.3) — sweep it fresh once run C/D is the
baseline.

---

## Phase 3 — Backbone modernization ("supersede L-GATr")

**3.1 Invariant-logit attention.** Replace sigmoid-gated message sums with softmax
attention whose logits are functions of pairwise invariants + node scalars
(invariant logits ⇒ equivariant output). Keeps the "minimal universal backbone" story.
**3.2 Scale.** After 2.2: 6–8 layers × 256 dim (~10–20M params). Depth/width sweep at
fixed FLOPs. Data (~850k jets) supports this without augmentation.
**3.3 L-GATr head-to-head — DEFERRED, out of scope now.** Required eventually for the
"supersedes" claim, but not run in this cycle. When it happens: predefine the exact metric
set *before* running (else "supersedes" looks post-hoc), keep strictly conditional on
Phase 1–3 numbers justifying it, budget ~1 week; also compare non-equivariant SOTA
(EPiC-FM, FPCD).

---

## Phase 4 — Mass-shell Riemannian flow matching (research headline)

**4.1 Rewrite `util/hyperbolic.py`** as hyperboloid/mass-shell version: exp/log/distance.
**float64 for all geometry ops is the DEFAULT, not a fallback** — the small-m NaN failure
mode is cheap to preempt and expensive for an agent to diagnose. Numerics hardened (stable
arccosh near 1; rapidity-domain formulas for far points). Unit tests: geodesic consistency
(exp∘log = id), tangency preservation, equivariance under random Λ.
**4.2 Integrate** via the existing `--use_hyperbolic` path: lift data/prior with
regulator mass, tangent-project model output, Riemannian loss, geodesic Euler sampling.
**Padding:** park masked particles at (m, 0, 0, 0) — the hyperboloid apex. This sits *on*
the shell, so log/exp are fine, **but geodesic-cost ICP/OT will happily match real particles
to parked apex points unless masking is enforced inside the cost.** The pairing cost MUST
mask padded rows explicitly (∞ cost to/from padded slots); this is a required guard, not an
optimization.
**4.3 Regulator mass ablation:** m ∈ {0.1, 0.3, 1.0} in scaled units. Watch numerical
stiffness (small m) vs. kinematic distortion (large m).
**4.4 Geodesic-distance ICP/OT.** Replace Euclidean ICP cost (`cache_icp.py`, currently
`cdist` euclidean) with geodesic distance → Lorentz-invariant coupling. Enforce padded-row
masking in the cost (see 4.2). Ablate vs. Euclidean ICP vs. no pairing.
**4.5 Head-to-head: Euclidean FM (Phase 1–3 best) vs. mass-shell RFM,** same backbone,
same budget. Key claims: physicality diagnostics exactly zero; equal or better distribution
metrics; sample-efficiency in integration steps.

---

## Phase 5 — Paper ablations & analysis

- Full ablation table: ref vectors / h_i / prior / attention / geometry / ICP variant /
  EMA / CFG weight & schedule / time sampling / curriculum (re-check whether curriculum
  still helps post-Phase-1; its benefit may have been compensating for the symmetry
  pathology).
- Steps-vs-quality (inference-time compute) curves per geometry.
- Scaling behavior (params × data), even if only 2–3 points.
- Isotropy figure: before/after as the motivating illustration.
- 150-particle JetNet once 30-particle results are solid; JetClass only if time allows.

---

## Risks & mitigations

- **1/(1−t) target blowup near t=1** (both geometries): existing time-sampling
  machinery; optionally clip t < 1−ε in training.
- **H³ numerics at small m:** log-domain implementation, **float64 for geometry ops by
  default** (they're O(N), cheap).
- **Curriculum/ICP interactions:** re-ablate after Phase 1 — components tuned against a
  symmetry-broken baseline may change sign.
- **Silent reproduction of run A:** the blocking pytest (0.2, assertion 2) is the tripwire.
- **Incomparable numbers across runs:** the frozen-commit hard gate (0.4).

## Priority order

Phase 0 (harness + blocking test) → Phase 1 grid A–F on the frozen commit → Phase 2
(concurrent) → decide: if Phase 1 numbers are strong, Phase 4 becomes the headline and
Phase 3 the supporting comparison; if not, debug before adding geometry.
