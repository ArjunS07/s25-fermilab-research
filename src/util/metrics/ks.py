"""
util/ks.py — dependency-free one-sample Kolmogorov-Smirnov test against a uniform reference.

Used for the experiment plan's isotropy abort signal: under an isotropic jet-axis distribution
the cos(theta) of the total 3-momentum is Uniform[-1, 1] and phi is Uniform[-pi, pi]. A run
that breaks the spurious rotational symmetry (references / axis-aligned prior on) must *depart*
from uniform; the KS p-value quantifies that.

Pure numpy so it runs both at eval time (in util/metrics.py, on the cluster) and locally in the
grid aggregator (no scipy needed).
"""
import numpy as np


def ks_pvalue(d: float, n: int) -> float:
    """Asymptotic KS p-value (Kolmogorov distribution, Numerical-Recipes form with the
    Stephens small-sample correction)."""
    if n <= 0:
        return float("nan")
    en = np.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * d
    if lam <= 0:
        return 1.0
    j = np.arange(1, 101)
    terms = 2.0 * ((-1.0) ** (j - 1)) * np.exp(-2.0 * (j ** 2) * (lam ** 2))
    return float(min(max(terms.sum(), 0.0), 1.0))


def ks_two_sample(samples_a, samples_b):
    """Two-sample KS test. Returns (D, p). Either input empty → (nan, nan)."""
    a = np.sort(np.asarray(samples_a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(samples_b, dtype=np.float64).ravel())
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    all_vals = np.concatenate([a, b])
    all_vals.sort()
    cdf1 = np.searchsorted(a, all_vals, side="right") / n1
    cdf2 = np.searchsorted(b, all_vals, side="right") / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))
    n_eff = int(round(n1 * n2 / (n1 + n2)))
    return d, ks_pvalue(d, n_eff)


def ks_statistic_vs_uniform(samples, low: float, high: float):
    """Two-sided KS statistic D and p-value of ``samples`` against Uniform[low, high].

    Returns (D, p). Empty input returns (nan, nan).
    """
    x = np.sort(np.asarray(samples, dtype=np.float64))
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0 or high <= low:
        return float("nan"), float("nan")
    # Reference uniform CDF evaluated at each sample point.
    cdf = np.clip((x - low) / (high - low), 0.0, 1.0)
    ecdf_upper = np.arange(1, n + 1) / n
    ecdf_lower = np.arange(0, n) / n
    d = float(np.max(np.maximum(np.abs(ecdf_upper - cdf), np.abs(cdf - ecdf_lower))))
    return d, ks_pvalue(d, n)
