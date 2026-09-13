# JetFUEL layerwise capacity experiment

## Scientific question

Can the six-block, width-96 H model be reduced to three blocks, and can its
hidden width be reduced to 64, without materially degrading class-conditional
JetNet-30 quality?

The experiment deliberately separates depth from width. A post-hoc checkpoint
intervention measures dependence on existing blocks; it does not claim that a
shallower architecture retrained from scratch will have the same behavior.

## Phase A: frozen-checkpoint screen

Run `analysis/analyze_layerwise_importance.py` on the canonical GQT CFG-994K
EMA checkpoint and its 150k-jet replay bundle. Use 512 deterministic jets per
class, the 64-step reference trajectory, snapshots at steps
`0,8,16,24,32,40,48,56,63`, and the currently selected internal guidance
weights `(g,q,t)=(0,0.25,0.5)`.

The code records:

- masked RMS block updates to scalar state `h` and auxiliary coordinate state
  `y`;
- final guided-velocity change and cosine similarity under `alpha=0.5` and
  `alpha=0` for each block;
- cumulative prefix-depth interventions retaining only the first 1--5 blocks;
- alternating three-block interventions retaining blocks 1/3/5 or 2/4/6; and
- hidden-state covariance spectra, participation ratio, ranks explaining
  90/95/99% of variance, and variance explained by the top 64 directions.

Primary outputs are `layer_residuals.csv`, `field_sensitivity.csv`,
`summary.json`, and `layerwise_importance.{png,pdf}`. Rank interventions by
velocity sensitivity separately by class and time; do not rank by raw
`Delta y` alone.

## Phase B: causal common-replay FPND

Promote the plausible depth candidates to full frozen-checkpoint rollouts with
`analysis/evaluate_layer_gate.py`. The pre-specified set is:

1. baseline, all six blocks;
2. prefix depth 5 (disable block 6);
3. prefix depth 4 (disable blocks 5--6);
4. prefix depth 3 (disable blocks 4--6);
5. alternating blocks 1/3/5; and
6. alternating blocks 2/4/6.

Every arm uses the same 150k replay bundle, 64 Euler steps, EMA weights, and
unguided inference for an architecture-only comparison. Record per-class FPND,
W1M, W1P, FPD, coverage/MMD, invalid fraction, and physicality. Run the full
Phase-B manifest only after Phase A identifies at least one three-block pattern
with finite trajectories and tolerable field distortion.

## Interpretation and training decision

- A small residual does not establish dispensability: later blocks may amplify
  it. Prefer the gate-induced final-velocity and endpoint-quality effects.
- A damaging post-hoc gate does not prove a smaller model cannot learn: the
  existing head and later representations were trained for six blocks.
- If a three-block pattern remains competitive in Phase B, train a clean
  width-96/depth-3 model from scratch before combining depth and width changes.
- If the top-64 hidden directions explain nearly all activation variance across
  layers/classes/times, the independent width treatment is width 64/depth 6.
- Only after those controlled experiments should width 64/depth 3 be launched.

At 30 particles, the dense-layer estimates are:

| architecture | parameters | forward GFLOP | fraction of 96x6 |
|---|---:|---:|---:|
| width 96, depth 6 | 617,297 | 0.456024 | 1.000 |
| width 96, depth 3 | 337,646 | 0.245450 | 0.538 |
| width 64, depth 6 | 276,369 | 0.203492 | 0.446 |
| width 64, depth 3 | 151,374 | 0.109558 | 0.240 |

## Width-64 training and evaluation

`nrp/as-jet-train-gqt30-h-cfg-w64-994k.yaml` changes only hidden width from 96
to 64 relative to the canonical H recipe. It retains six blocks, GQT joint
conditioning, 10% CFG null dropout, online geodesic ICP, fresh axis-aligned
lognormal prior, EMA, optimizer schedule, seeds, and 994k optimizer updates.

Its terminal evaluation is the balanced 150k, 64-step, `w=0` reference and
writes a completion pointer only after checkpoint/config/artifact validation.
After training succeeds, launch
`nrp/as-jet-eval-gqt30-h-cfg-w64-sweep.yaml`. The common-replay sweep uses
internal weights `0.125, 0.25, 0.375, 0.5, 0.75, 1.0`; together with the
terminal `w=0` arm, this contains the current class-selected weights and a
focused neighborhood around them. Select the weight independently by class
only after all completed arms pass provenance and physicality checks.

