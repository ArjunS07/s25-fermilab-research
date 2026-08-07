"""Density-bucketed curriculum sampler for dense-first jet-multiplicity training."""
import torch


class CurriculumSampler:
    """Buckets training jets by particle count and draws epoch indices with a dense-first
    curriculum.

    Bucket ``k`` in ``{0, …, N-1}`` (k=0 sparsest, k=N-1 densest). Sampling weight per bucket is
    ``P(k) ∝ (k+1)^α``; ``α`` decays linearly from ``alpha_start`` (epoch 0) to 0 (final epoch),
    so training oversamples dense jets early and flattens to uniform-per-jet later. Empty buckets
    get zero weight automatically (no jets belong to them).
    """

    def __init__(self, n_particles_per_jet, n_buckets, alpha_start):
        self.n_buckets = n_buckets
        self.alpha_start = alpha_start
        p = n_particles_per_jet.cpu().float()
        p_min, p_max = p.min().item(), p.max().item()
        width = (p_max - p_min + 1e-6) / n_buckets
        self.assignments = ((p - p_min) / width).long().clamp(0, n_buckets - 1)
        self.counts = torch.bincount(self.assignments, minlength=n_buckets).float()
        self.n_nonempty = int((self.counts > 0).sum().item())

    def sample(self, epoch, total_epochs, n_samples, generator):
        """Draw ``n_samples`` training indices for ``epoch`` (with replacement, so the
        curriculum distribution is exact). ``total_epochs`` is used for the α schedule so it
        stays continuous across resume boundaries."""
        alpha = self.alpha_start * (1.0 - epoch / max(total_epochs - 1, 1))
        probs = torch.pow(torch.arange(1, self.n_buckets + 1, dtype=torch.float), alpha)
        probs = probs / probs.sum()
        weights = probs[self.assignments] / self.counts.clamp(min=1)[self.assignments]
        return torch.multinomial(weights, n_samples, replacement=True, generator=generator)
