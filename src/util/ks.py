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
