# JetFUEL campaign results

Canonical ledger for the completed JetFUEL campaigns. Values come from each run's
terminal `train/summary.json`; 20-step smoke jobs and interrupted GQT jobs are excluded.
Lower is better for FPND, FPD, W1M, mean W1P, and MMD. Higher is better for coverage.

The Markdown table shows point estimates. The companion
[`jetfuel_campaign_results.csv`](jetfuel_campaign_results.csv) retains metric
uncertainties, individual W1P components, configuration fields, physicality fractions, and
source-run identifiers.

| Run | Steps | Design | FPND | FPD | W1M | Mean W1P | Cov. | MMD | Physicality | Qualified |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|:---:|
| **A** | 200K | Euclidean; no refs | — | — | 0.249857 | — | — | — | 4.492% invalid; 25.45% spacelike; 2.29% neg-E | No |
| **B** | 200K | RFM; no refs | 91.7956 | 0.103140 | 0.025672 | 0.011939 | 0.436 | 0.030081 | 0 invalid | Yes |
| **C** | 200K | Euclidean; plain refs | 2.166e27 | 762.927 | 0.105699 | 0.050116 | 0.334 | 0.045125 | 0.006% invalid; 33.87% spacelike; 0.298% neg-E | No |
| **D** | 200K | RFM; plain refs | 16.3088 | 221.145 | 0.008915 | 0.012730 | 0.426 | 0.030113 | 0.004% invalid | Yes |
| **E** | 200K | RFM; ref scalars; ICP; no ref readout | 28.4260 | 0.052538 | 0.025347 | 0.016473 | 0.389 | 0.032081 | 0 invalid | Yes |
| **F** | 200K | RFM; ref scalars; ICP; plain refs | 23.0223 | 0.025269 | 0.023141 | 0.015200 | 0.405 | 0.029369 | 0 invalid | Yes |
| **G-200K** | 200K | Log-map; raw refs; evolving geometry | 0.3935 | 0.0001593 | 0.001884 | 0.001371 | 0.563 | **0.026474** | 0 invalid | Yes |
| **H-200K** | 200K | Log-map; normalized tangent refs; evolving geometry | 0.3745 | 0.0001469 | **0.001808** | **0.001264** | **0.565** | 0.026835 | 0 invalid | Yes |
| **I-200K** | 200K | Log-map; raw refs; fixed physical geometry | 0.4264 | 0.0001685 | 0.001981 | 0.001405 | 0.545 | 0.027930 | 0 invalid | Yes |
| **J-200K** | 200K | Log-map; normalized tangent refs; fixed physical geometry | 0.4563 | 0.0001747 | 0.002191 | 0.001465 | 0.539 | 0.028152 | 0 invalid | Yes |
| **H-BlockCond-200K** | 200K | H + condition/time injection in every block | 0.3868 | 0.0001524 | 0.002164 | 0.001299 | 0.544 | 0.027402 | 0 invalid | Yes |
| **H-500K** | 500K | H extended to 500K | **0.3678** | 0.0001397 | 0.002030 | 0.001334 | 0.551 | 0.028440 | 0 invalid | Yes |
| **G-500K** | 500K | G extended to 500K | 0.3754 | **0.0001389** | 0.001868 | 0.001384 | 0.555 | 0.026761 | 0 invalid | Yes |
| **G-331K** | 331K | H recipe; gluon specialist | 0.3931 (g) | 0.0001508 | 0.002070 | 0.001365 | 0.536 | 0.027612 | 0 invalid | Yes |
| **T-331K** | 331K | H recipe; top specialist | 2.8427 (t) | 0.428956 | 0.005829 | 0.003659 | 0.574 | 0.056644 | 0 invalid | Yes |
| **GQT-994K** | 994K | H recipe; joint g/q/t | **0.1311 (g); 0.2616 (q); 3.1530 (t)** | ⚠ raw 646,050 | **0.001149** | **0.000919** | 0.562 | 0.036148 | 2/50K invalid; 0 neg-E/spacelike among finite | Yes* |

\* GQT-994K passed the configured qualification gate and reached exactly 994,000 optimizer
steps. Its raw 50K FPD is quarantined: the saved 10K subset gives FPD
0.001714 ± 0.000244 (class-count-matched: 0.002260 ± 0.000368), proving that the terminal
646,050 value is driven by a rare event outside the saved subset rather than the bulk.
Do not use the raw FPD in paper comparisons until a full-sample tail diagnostic is rerun.

## Design key

- **A–D:** initial Euclidean/RFM × reference-readout factorial; equal-share axis prior; no ICP.
- **E–F:** invariant reference-contraction scalar initialization and online geodesic ICP;
  equal-share axis prior.
- **G–J:** normalized log-map particle readout, signed square-root-degree aggregation,
  lognormal shares, online geodesic ICP, and the raw/normalized-reference ×
  evolving/fixed-geometry factorial.
- **H-BlockCond:** H with time/condition context injected into every message-passing block.
- **500K rows:** exact G/H recipes extended from 200K to 500K optimizer steps.
- **G/T-331K:** compute-split specialist runs using the H architecture and frozen 331K
  campaign harness.
- **GQT-994K:** joint three-class H run at the paper-comparison compute budget; the CSV uses
  one provenance row per class-specific FPND while retaining the shared aggregate metrics.

## Main conclusions

- H-200K is the strongest balanced 200K arm and retains the best W1M, mean W1P, and
  coverage in this JetFUEL ledger.
- Extending H to 500K improves FPND and FPD but regresses W1M, mean W1P, coverage, and MMD.
- G-500K has the best FPD; H-500K has the best FPND.
- Fixed physical geometry (I/J) and per-block condition/time injection do not improve H.
- Every G–J/500K sample set is fully physical under the recorded invalid, negative-energy,
  and spacelike checks.
- GQT-994K is exceptionally strong for gluons and quarks, while top remains the bottleneck.
  Its W1M and mean W1P are the best in this ledger, but its raw FPD is not scientifically
  usable until the isolated rare-event pathology is identified or reproduced.
