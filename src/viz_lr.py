"""
viz_lr.py — Plot the cosine annealing with warm restarts LR schedule.

Mirrors the scheduler logic in train.py so what you see is exactly what
the model experiences during training.

Usage:
    python viz_lr.py
    python viz_lr.py --num_epochs 100 --lr 3e-4 --lr_t0 25
    python viz_lr.py --num_epochs 50 --output /tmp/lr.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

plt.rc("mathtext", fontset="cm")
sns.set_style("whitegrid")
_PALETTE = sns.color_palette("deep")


def plot_lr_schedule(num_epochs=50, lr=3e-4, lr_t0=0, output="lr_schedule.png"):
    t0 = lr_t0 if lr_t0 > 0 else (num_epochs // 4 if num_epochs >= 20 else max(1, num_epochs // 2))
    eta_min = lr * 0.1

    # Dummy model/optimizer — identical setup to train.py
    dummy = nn.Linear(1, 1)
    optimizer = optim.AdamW(dummy.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t0, T_mult=1, eta_min=eta_min
    )

    lrs = []
    for epoch in range(num_epochs):
        scheduler.step(epoch)
        lrs.append(optimizer.param_groups[0]["lr"])

    epochs = list(range(num_epochs))
    n_restarts = num_epochs // t0

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(epochs, lrs, color=_PALETTE[0], linewidth=2, zorder=3)
    ax.axhline(eta_min, color="gray", linestyle="--", linewidth=1.2,
               label=rf"$\eta_{{\min}}$ = {eta_min:.1e}", zorder=2)

    # Restart markers
    restart_epochs = [i * t0 for i in range(n_restarts + 1) if i * t0 < num_epochs]
    for i, re in enumerate(restart_epochs):
        ax.axvline(re, color=_PALETTE[3], linestyle=":", linewidth=1.0,
                   alpha=0.7, label=f"Restarts ($T_0={t0}$)" if i == 0 else None,
                   zorder=1)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Learning Rate", fontsize=13)
    ax.set_title(
        rf"CosineAnnealingWarmRestarts  "
        rf"($\eta_{{max}}={lr:.0e}$,  $\eta_{{min}}={eta_min:.0e}$,  "
        rf"$T_0={t0}$,  {num_epochs} epochs)",
        fontsize=12,
    )
    ax.set_xlim(0, num_epochs - 1)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot LR schedule")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_t0", type=int, default=0,
                        help="Warm restart period. 0 = auto (num_epochs // 4)")
    parser.add_argument("--output", type=str, default="lr_schedule.png")
    args = parser.parse_args()

    plot_lr_schedule(
        num_epochs=args.num_epochs,
        lr=args.lr,
        lr_t0=args.lr_t0,
        output=args.output,
    )
