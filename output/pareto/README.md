# JetNet-30 GPU-model-FLOP–FPND Pareto study

This directory contains three 3×3 compute–quality figures, one for each
lifecycle deployment scale `M = 10^6, 10^7, 10^8`:

- columns: gluon, light-quark, and top jets;
- rows: training compute, marginal inference compute, and lifecycle compute;
- quality coordinate: FPND, where lower is better;
- lifecycle scenarios: one, ten, and one hundred million total generated jets
  across the applicable displayed classes.

The primary complete figures plot both compute and FPND on linear axes. The
OmniLearn/EPiC-GAN-omitted presentation copies use a logarithmic compute axis
and a linear FPND axis.

The analysis uses **GPU model FLOPs only**. It excludes CPU data loading,
preprocessing, evaluation metrics, disk/network I/O, GPU-hours, and measured
latency. It also does not convert FP64 work into an FP32-throughput equivalent:
one floating-point operation is one FLOP regardless of precision.

The analysis is for **conditional constituent generation**. Jet type and
jet-level kinematic attributes are supplied by the scientist. Consequently,
JetFUEL Stage 1 and the analogous FPCD/PET/OmniLearn jet-attribute generators
are excluded from training and inference costs.

## FLOP convention

One multiplication plus one addition is two FLOPs. For a dense layer applied to
`n` objects, the counted cost is

```text
F_linear(n, a, b) = 2 n a b.
```

For dense self-attention with `N` tokens and width `d`, the counted cost is

```text
F_attention(N, d) = 8 N d^2 + 4 N^2 d.
```

Training forward plus backward is estimated as three forward-pass FLOPs. For
the released MPGAN and EPiC-GAN loops, the discriminator and generator update
structure instead gives `6 F_G + 9 F_D` per exposed jet.

The portable GPU-model-FLOP convention counts dense linear, attention, and
message-passing contractions. Biases, nonlinearities, normalization,
random-number generation, optimizer updates, masking, top-k selection, and
transcendental/scalar geometry kernels are not assigned synthetic FLOP weights.
The same scope is used for every model. This is a model-arithmetic comparison,
not a runtime or energy comparison.

## Portfolio accounting

The figure is intended to compare methods capable of delivering the full set of
applicable `g/q/t` results. Training is therefore a **portfolio acquisition
cost**:

- JetFUEL is one joint `g/q/t` model and is counted once.
- FPCD, PET, and OmniLearn are joint conditional models and are counted once.
- MPGAN and EPiC-GAN use separate models for `g`, `q`, and `t`, so the three
  class-model costs are summed.
- PC-JeDi reports only `g` and `t`; its two applicable class models are summed
  and it remains absent from the `q` panels.

This avoids charging JetFUEL's joint training three times or comparing its full
joint run against only one of a baseline's three required fits. The same
portfolio training coordinate is repeated across the three quality columns.

The training row contains **one point per independently trained set of
weights**, not one point per inference configuration. JetFUEL's 64/128-step and
guided/unguided samples all use the same checkpoint, so this row displays the
lowest measured FPND from that checkpoint for each class: 128/unguided for `g`
and `q`, and 128/`w=0.5` for `t`. PC-JeDi's DDIM and EM samplers are likewise
collapsed to its better EM result. The inference and lifecycle rows retain all
sampler configurations because those choices change marginal deployment cost.

## Model-level calculation ledger

`PF` is PFLOP and `GF/jet` is GFLOP per generated jet. A `≤` training value uses
the published or released maximum epoch budget because the selected stopping
epoch is unavailable.

| Method | Portfolio training calculation | Training cost | Conditional inference calculation | GF/jet |
|---|---:|---:|---:|---:|
| MPGAN | `g + q + t` separate fits: 695.121 + 868.901 + 695.121 PF | 2,259.142 PF | `F_G` | 0.192822 |
| EPiC-GAN | three class fits, each including three selection seeds | 311.121 PF | `F_G` | 0.012732 |
| FPCD | `535.5k × 250 × 3F_particle` | ≤24.097 PF | `512F_particle` | 30.719410 |
| FPCD-1 | baseline plus nine `5F_particle` distillation stages | ≤385.553 PF | `F_particle` | 0.059999 |
| PET | `535.5k × 300 × 3F_particle` | ≤81.543 PF | `600F_body + 1200F_head` | 111.726490 |
| OmniLearn | upstream pretraining plus JetNet fine-tuning | ≤109,536.216 PF | `600F_body + 1200F_head` | 111.726490 |
| PC-JeDi | separate `g + t` fits: `2 × 59.592` PF | ≤119.184 PF | `200F` | 8.655360 |
| JetFUEL | `994k optimizer steps × 250 jets × 3F` | 339.966 PF | see variants below | — |

The downstream-only OmniLearn fine-tuning estimate is 65.235 PF. The figure
uses fully burdened pretraining; the alternative value is retained in
`compute_fpnd_ledger.csv` rather than silently treating the pretrained model as
free.

## Diffusion integration-step audit

The relevant quantity for GPU FLOPs is the number of neural-network
evaluations (NFE), not the number printed as “integration steps.” The released
samplers give the following source-level accounting:

| Method | Nominal steps | Calls in one step | Counted particle-model work per jet |
|---|---:|---|---:|
| FPCD | 512 | one DDIM particle-model call | 512 full particle evaluations |
| FPCD-1 | 1 | one distilled DDIM particle-model call | 1 full particle evaluation |
| PET / OmniLearn | 300 | predictor plus second-order correction | 600 body + 1,200 head evaluations |
| PC-JeDi DDIM / EM | 200 | one score-network call | 200 full particle evaluations |
| JetFUEL, unguided (`w=0`) | 64 or 128 | one vector-field call | 64 or 128 full evaluations |
| JetFUEL, positive guidance | 64 or 128 | conditional plus unconditional call | 128 or 256 full evaluations |

For FPCD, the official implementation loops over `self.num_steps` and calls the
particle model once inside each DDIM update; the paper and released checkpoint
documentation set the baseline to 512 and FPCD-1 to one. Thus the original
`512F_particle` and `F_particle` entries were correct.

PET and OmniLearn need more care. Their released JetNet sampler fixes
`num_steps = 300`, but every step calls `evaluate_models` once for the predictor
and once inside `second_order_correction`. Each `evaluate_models` call executes
the particle body once and, as written, calls the head twice: conditional and
zero-weight unconditional. The primary source-level estimate therefore counts
`600F_body + 1200F_head = 111.726490 GF/jet`. A TensorFlow deployment that
proves and prunes the `w=0` unconditional branch would instead execute
`600F_body + 600F_head = 101.516544 GF/jet`; this implementation-dependent
lower bound is not used in the primary table.

## JetFUEL forward calculation

An independent operator profiler confirms the hand calculation. For
`N=30`, width 96, and six LorentzNet blocks:

| Component | MAC per forward |
|---|---:|
| Embeddings | 301,728 |
| Six pair/node/coordinate blocks | 210,574,080 |
| Pairwise field head | 16,848,000 |
| Reference head | 288,000 |
| **Total** | **228,011,808 MAC** |

This is 456,023,616 GPU dense-layer FLOPs per model evaluation under the
two-FLOP MAC convention. The operator profiler reports 456,636,878 recognized
FLOPs, of which exactly 456,023,616 are the linear layers. A profiled forward
plus backward reports 1,369,987,598 recognized FLOPs, validating the
approximately three-forward training multiplier. The profiler was used only to
validate operation counts; no CPU execution time enters the plotted values.

The run log contains 994,000 optimizer updates. The physical minibatch is 50
and five minibatches are normally accumulated, giving a target effective batch
of 250. End-of-epoch partial accumulation groups make the exact exposure count
approximately 248.40--248.43 million jets rather than the simple upper estimate
of 248.50 million. This changes the stated 339.966 PF training estimate by less
than 0.05%.

| JetFUEL variant | Evaluations/jet | GFLOP/jet | FPND g | FPND q | FPND t |
|---|---:|---:|---:|---:|---:|
| 64 steps, unguided (`w=0`) | 64 | 29.185511 | 0.141994 | 0.267901 | 0.524383 |
| 128 steps, unguided (`w=0`) | 128 | 58.371023 | 0.079720 | 0.154623 | 0.369262 |
| 64 steps, class-optimal guidance | 64 (`g`); 128 (`q,t`) | 29.185511 (`g`); 58.371023 (`q,t`) | 0.141994 (`w=0`) | 0.213679 (`w=0.25`) | 0.201928 (`w=0.5`) |
| 128 steps, best completed guidance | 128 (`g,q`); 256 (`t`) | 58.371023 (`g,q`); 116.742046 (`t`) | 0.079720 (`w=0`) | 0.154623 (`w=0`) | 0.174388 (`w=0.5`) |

Guidance is selected independently by class, including `w=0` when every
positive guidance weight is worse. The completed 64-step sweep gives
`(w_g,w_q,w_t)=(0,0.25,0.5)`. At 128 steps, only `w=0` and `w=0.5` have
completed, giving the provisional best-measured tuple `(0,0,0.5)`. The
128-step `w=0.25` arm is still Pending on the cluster, so no unmeasured q-jet
FPND is inferred or plotted.

Here `w=0` means **unguided inference from the CFG-trained checkpoint**, not a
checkpoint trained without CFG dropout. The checkpoint used for this sweep was
trained with `cfg_null_dropout_rate=0.1`, set by the Kubernetes manifest's CLI
override and verified in the saved checkpoint.

## Lifecycle calculation

For a deployment of `M` jets,

```text
C_lifecycle(M) = C_train + M C_infer_per_jet.
```

For `M = 10^6, 10^7, 10^8`, conversion from GFLOP/jet to PFLOP simplifies to

```text
C_inference(PF) = (M / 10^6) C_infer(GF/jet)
C_lifecycle(PF) = C_train(PF) + C_inference(PF).
```

The lifecycle training term is the portfolio cost defined above. Each inference
term is for the stated total number of generated jets; an equal class mix is not
required because the reconstructed per-jet architecture cost is the same for
every class within each method.

The complete machine-readable calculation is in `gpu_lifecycle_costs.csv`.
Below, each deployment cell is `inference PF / lifecycle PF`.

| Method / sampler | Training PF | Marginal GF/jet | M=10^6 | M=10^7 | M=10^8 |
|---|---:|---:|---:|---:|---:|
| MPGAN | 2,259.142 | 0.192822 | 0.193 / 2,259.335 | 1.928 / 2,261.070 | 19.282 / 2,278.424 |
| FPCD, 512 step | 24.097 | 30.719410 | 30.719 / 54.816 | 307.194 / 331.291 | 3,071.941 / 3,096.038 |
| FPCD-1, 1 step | 385.553 | 0.059999 | 0.060 / 385.613 | 0.600 / 386.153 | 6.000 / 391.552 |
| EPiC-GAN | 311.121 | 0.012732 | 0.013 / 311.134 | 0.127 / 311.249 | 1.273 / 312.395 |
| PET | 81.543 | 111.726490 | 111.726 / 193.270 | 1,117.265 / 1,198.808 | 11,172.649 / 11,254.192 |
| OmniLearn, fully burdened | 109,536.216 | 111.726490 | 111.726 / 109,647.942 | 1,117.265 / 110,653.481 | 11,172.649 / 120,708.865 |
| OmniLearn, downstream only | 65.235 | 111.726490 | 111.726 / 176.961 | 1,117.265 / 1,182.499 | 11,172.649 / 11,237.883 |
| PC-JeDi, DDIM or EM | 119.184 | 8.655360 | 8.655 / 127.840 | 86.554 / 205.738 | 865.536 / 984.720 |
| JetFUEL 64, unguided (`w=0`) | 339.966 | 29.185511 | 29.186 / 369.151 | 291.855 / 631.821 | 2,918.551 / 3,258.517 |
| JetFUEL 128, unguided (`w=0`) | 339.966 | 58.371023 | 58.371 / 398.337 | 583.710 / 923.676 | 5,837.102 / 6,177.068 |
| JetFUEL 64, optimal guidance (`g`) | 339.966 | 29.185511 | 29.186 / 369.151 | 291.855 / 631.821 | 2,918.551 / 3,258.517 |
| JetFUEL 64, optimal guidance (`q,t`) | 339.966 | 58.371023 | 58.371 / 398.337 | 583.710 / 923.676 | 5,837.102 / 6,177.068 |
| JetFUEL 128, best completed guidance (`g,q`) | 339.966 | 58.371023 | 58.371 / 398.337 | 583.710 / 923.676 | 5,837.102 / 6,177.068 |
| JetFUEL 128, best completed guidance (`t`) | 339.966 | 116.742046 | 116.742 / 456.708 | 1,167.420 / 1,507.386 | 11,674.205 / 12,014.170 |

All JetFUEL rows share the same one-time 339.966 PF training run; their
different entries are inference configurations, not separately trained models.
The guidance rows use class-specific weights and therefore have class-specific
marginal inference costs. Likewise, PC-JeDi DDIM and EM share the same training
acquisition cost.

## Provenance

- FPND comparison values: `jetnet30_method_table.tex` in the paper repository.
- JetFUEL 64-step CFG sweep: `report/cfg_guidance_sweep_results.csv`; its
  checkpoint was trained with the manifest override
  `training.cfg_null_dropout_rate=0.1` in
  `src/nrp/as-jet-train-gqt30-lorentznet-h-cfg-994k.yaml`.
- JetFUEL 128-step no-CFG and `w=0.5`: completed outputs from
  `as-jet-eval-gqt994k-euler-cfg-ablations` on the shared NRP PVC.
- JetFUEL architecture and training exposure: `src/models/lorentznet_flow.py`,
  the CFG training manifest, and the CFG evaluation `summary.json`, which
  embeds the source checkpoint's full training configuration.
- Competitor architectures and schedules: official papers and released source
  repositories for MPGAN, EPiC-GAN, GSGM/FPCD, OmniLearn/PET, and PC-JeDi.
- FPCD step audit: [paper](https://arxiv.org/abs/2304.01266) and
  [official GSGM sampler](https://github.com/ViniciusMikuni/GSGM/blob/main/scripts/GSGM.py).
- PET/OmniLearn step audit:
  [official JetNet sampler](https://github.com/ViniciusMikuni/OmniLearn/blob/main/scripts/PET_jetnet.py).
- PC-JeDi step audit: [paper](https://arxiv.org/abs/2303.05376) and
  [official repository](https://github.com/rodem-hep/PC-JeDi).

The plotted coordinates and computed Pareto membership are recorded in
`compute_fpnd_ledger.csv`. The Python source is
`plot_compute_fpnd_pareto.py`.

The scale-specific figures are
`jetnet30_compute_fpnd_pareto_3x3_m1e6.{png,pdf}`,
`jetnet30_compute_fpnd_pareto_3x3_m1e7.{png,pdf}`, and
`jetnet30_compute_fpnd_pareto_3x3_m1e8.{png,pdf}`. The unsuffixed figure is a
backward-compatible copy of the `M=10^8` version.

Presentation copies that omit OmniLearn and EPiC-GAN from the points, legend,
Pareto calculation, and axis autoscaling use the suffix
`_no_omnilearn_epicgan`. The complete CSV ledgers and primary figures still
retain both methods.
