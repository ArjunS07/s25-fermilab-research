# Compact plotting data

This directory is sufficient to regenerate every curated root figure in the paper
bundle. `results_snapshot.json` contains the validated ablation endpoints;
`training_loss_*.csv` contains the three matched 994k training curves; and
`distribution_histograms.json` contains density histograms reduced from the selected
GQT-994k replay bundles. The large sample tensors remain on the PVC.

From the repository root:

```bash
python src/analysis/generate_local_paper_plots.py \
  --data-dir output/pdf/paper_figures/leftjen_2026-09-11/data \
  --output-dir output/pdf/paper_figures/leftjen_2026-09-11
```

The plotting script requires Seaborn, Matplotlib, NumPy, and a working LaTeX
installation. It writes vector PDF and high-resolution PNG versions.
