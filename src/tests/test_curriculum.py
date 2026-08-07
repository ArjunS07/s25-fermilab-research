import torch

from training.curriculum import CurriculumSampler


def test_bucketing_and_counts():
    # 10 jets, particle counts 1..10, 5 buckets → 2 jets per bucket.
    counts = torch.arange(1, 11, dtype=torch.float)
    cs = CurriculumSampler(counts, n_buckets=5, alpha_start=2.0)
    assert cs.n_buckets == 5
    assert cs.n_nonempty == 5
    assert cs.counts.sum().item() == 10
    assert cs.assignments.min() == 0 and cs.assignments.max() == 4


def test_sample_shape_and_index_range():
    counts = torch.arange(1, 11, dtype=torch.float)
    cs = CurriculumSampler(counts, n_buckets=5, alpha_start=2.0)
    g = torch.Generator().manual_seed(0)
    idx = cs.sample(epoch=0, total_epochs=10, n_samples=256, generator=g)
    assert idx.shape == (256,)
    assert int(idx.min()) >= 0 and int(idx.max()) < 10


def test_dense_first_then_flattens():
    """High alpha (early) oversamples dense jets; alpha=0 (final epoch) is per-jet uniform."""
    counts = torch.arange(1, 11, dtype=torch.float)  # jet 9 is densest
    cs = CurriculumSampler(counts, n_buckets=5, alpha_start=4.0)
    g = torch.Generator().manual_seed(0)
    early = cs.sample(epoch=0, total_epochs=10, n_samples=20000, generator=g)
    late = cs.sample(epoch=9, total_epochs=10, n_samples=20000, generator=g)  # alpha=0
    # Early curriculum draws the densest jet far more often than the sparsest.
    assert (early == 9).sum() > (early == 0).sum() * 3
    # At alpha=0 the per-jet weights are equal → sparsest and densest are within ~15%.
    assert abs(int((late == 9).sum()) - int((late == 0).sum())) < 0.15 * 20000 / 10
