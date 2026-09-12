# Campaign ablation trace

Generated from `report/jetfuel_campaign_results.csv` by `python src/analysis/generate_campaign_tables.py`. Percentages are relative changes in the second run versus the first; negative is better for all listed error metrics. This document deliberately separates controlled contrasts from confounded campaign changes.

## Strongest controlled findings

- **Mass-shell geometry is the dominant early change.** With no reference readout, A → B reduces W1M from 0.2499 to 0.0257 (-89.7%) and removes the recorded 4.49% invalid / 25.5% spacelike rate. With plain references, C → D reduces W1M -91.6% and removes the 33.9% spacelike rate. Historical FPND values are excluded from both claims.
- **Evolving auxiliary geometry beats fixed physical geometry.** G → I (raw references) changes FPND +8.4%, W1M +5.2%, mean W1P +2.4%, coverage -3.2%, and MMD +5.5%. H → J repeats the direction with tangent references: FPND +21.8%, W1M +21.2%, and mean W1P +15.9%. This is the cleanest architectural verdict after the geometry choice.
- **Normalized tangent reference readout is a small, not decisive, improvement.** At 200K with evolving geometry, G → H improves FPND -4.8%, FPD -7.8%, W1M -4.0%, and mean W1P -7.8%; G retains the better MMD (0.0265 vs 0.0268). The W1M difference is smaller than the stored single-run uncertainties, so use H as the balanced default rather than claiming a robust win.
- **Injecting condition/time into every block is rejected.** H → H-BlockCond changes W1M +19.7%, mean W1P +2.8%, coverage -3.7%, and FPND +3.3%; no listed metric improves.

## Optimization-length evidence

- **H at 500K is a Pareto trade-off, not an unqualified upgrade.** H-200K → H-500K improves FPND -1.8% and FPD -4.9%, but worsens W1M +12.3%, mean W1P +5.6%, coverage -2.5%, and MMD +6.0%.
- **G scales a little more smoothly.** G-200K → G-500K changes FPND -4.6%, FPD -12.8%, and W1M -0.8%, with a modest MMD regression (+1.1%). Both are single training trajectories, so these are checkpoint-selection observations, not independent seed estimates.

## Later campaign results: important, but not causal ablations

- **GQT-994K is the strongest observed multi-class result for g/q.** It records corrected FPND g/q/t = 0.1311/0.2616/3.1530, shared W1M 0.0011, and mean W1P $9.188\!\times\!10^{-4}$. Its raw full-sample FPD is quarantined and must remain omitted. The gluon gain versus G-331K is substantial, but it confounds joint-class training, extra optimization, and campaign-harness changes; it is evidence of a strong result, not evidence that any one of those factors caused it.
- **Top remains the bottleneck.** The matched single-top specialist T-331K has FPND-t 2.8427 and W1M 0.0058, far above gluon-specialist G-331K's W1M 0.0021. This is a class-difficulty observation, not a fair g-vs-t model comparison.

## What the ledger does not identify

- E/F change multiple things relative to D (scalar initialization, online coupling, and readout), so their weaker W1M does not isolate any one of those ingredients.
- A–F FPND values are historical and uncorrected; never rank those arms with post-fix FPND or literature FPND.
- There is one seed per arm. Any claim that a small difference is reliable needs reruns or a paired evaluation protocol.
- PC-JeDi and MPGAN use test-set jet attributes as conditioning, while JetFUEL samples attributes in Stage 1. The paper table is useful context, not a fully protocol-matched leaderboard.
