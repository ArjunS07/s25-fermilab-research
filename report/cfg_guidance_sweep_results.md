# CFG guidance-weight sweep (GQT-30, 994K checkpoint)

This sweep evaluates classifier-free guidance (CFG) at seven inference weights. Each fresh-generation arm produced exactly 150,000 jets: 50,000 gluon, 50,000 quark, and 50,000 top jets. Metrics use the corrected canonical FPND input with the conditioned jet axis and physical jet pT. All arms used EMA weights, 64 integration steps, the `axis_aligned_lognormal` prior, and the same test set.

The `w=0` arm reuses the originally saved endpoint samples and replay bundle; only metrics were rerun after fixing the replay pT-unit and class-label alignment issues. Its generated samples are therefore directly comparable, but its generation wall time is not.

| CFG weight | FPND G | FPND Q | FPND T | Evaluation output |
|---:|---:|---:|---:|---|
| 0.00 | **0.141994** | 0.267901 | 0.524383 | `/mnt/data/output/2026-08-26_05-10-21--ef9da51c-6ce0-4bb2-9d68-d88e8f0af41a-paper30-gqt994k-cfg-w000-repaired/eval` |
| 0.25 | 0.318242 | **0.213679** | 0.280137 | `/mnt/data/output/2026-08-25_22-18-31--a31c196a-5009-42a1-82e9-a9696e6146ee-paper30-gqt994k-cfg-w025-eval/eval` |
| 0.50 | 0.525544 | 0.313784 | **0.201928** | `/mnt/data/output/2026-08-25_20-59-00--3a48a5f7-5493-4335-be28-c301daee932c-paper30-gqt994k-cfg-w050-eval/eval` |
| 0.75 | 0.706259 | 0.579637 | 0.258170 | `/mnt/data/output/2026-08-25_23-37-54--0ecc6c71-2a39-4c73-a06f-8209b8439363-paper30-gqt994k-cfg-w075-eval/eval` |
| 1.00 | 0.916041 | 1.633959 | 0.426059 | `/mnt/data/output/2026-08-26_00-57-28--0fba2724-6417-4776-b21c-90730b5fca28-paper30-gqt994k-cfg-w100-eval/eval` |
| 2.00 | 4.538778 | 5.721714 | 2.100775 | `/mnt/data/output/2026-08-26_03-49-38--85890744-d6b5-4783-968b-d0cd369be7d0-paper30-gqt994k-cfg-w200-eval/eval` |
| 3.00 | 25.027028 | 15.177389 | 8.481164 | `/mnt/data/output/2026-08-26_02-17-00--f3049b1b-fd67-4cf7-98b2-91ba172f9b88-paper30-gqt994k-cfg-w300-eval/eval` |

## Selected class-conditional deployment weights

Selecting the lowest FPND independently by class gives:

\[
(w_g,w_q,w_t)=(0,\;0.25,\;0.5).
\]

That is, use no CFG for gluon jets, mild CFG for quark jets, and moderate CFG for top jets.

## Provenance

- Checkpoint: `/mnt/data/output/2026-08-21_10-04-25--cf87d826-9bab-45fb-b52e-02c9a60af5b8-gqt30-lnet-h-cfg-994k/train/models/final_checkpoint.pth`
- Training checkpoint: optimizer step 994,000; EMA weights used for generation.
- Repository commit used by the sweep: `b4eb1daf934f5a4ee270fb0b0ee59ac6e1c41295`.
- Kubernetes jobs: `as-jet-eval-paper30-h-cfg-sweep-r3` and corrected replay repair `as-jet-eval-paper30-h-cfg-w000-repair-r2`.
- Per-class stratified W1M and FPD are also stored in each arm's `metrics.csv`, alongside aggregate W1EFP/W1M/W1P, FPD, Cov/MMD, physicality, and diagnostics.
- Physicality was clean for the completed arms: 0% negative-energy and 0% spacelike generated jets. A small number of non-finite generated jets were dropped before metric calculation; exact counts are recorded in the run logs.
