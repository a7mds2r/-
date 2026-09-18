"""
metrics.py
==========
Multi-objective quality indicators + inferential statistics
(Wilcoxon, Friedman + Nemenyi, Vargha-Delaney A12, Cliff's Delta).
"""

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
import scikit_posthocs as sp


def hypervolume(F: np.ndarray, ref_point: np.ndarray) -> float:
    """2D/3D Monte-Carlo hypervolume approximation (minimization, dominated volume)."""
    F = F[np.all(F <= ref_point, axis=1)]
    if len(F) == 0:
        return 0.0
    n_samples = 200_000
    rng = np.random.default_rng(0)
    lo = F.min(axis=0)
    samples = rng.uniform(lo, ref_point, size=(n_samples, F.shape[1]))
    dominated = np.zeros(n_samples, dtype=bool)
    for f in F:
        dominated |= np.all(samples >= f, axis=1)
    box_vol = np.prod(ref_point - lo)
    return dominated.mean() * box_vol


def igd(F: np.ndarray, ref_front: np.ndarray) -> float:
    """Inverted Generational Distance: avg distance from each ref point to nearest F point."""
    d = np.linalg.norm(ref_front[:, None, :] - F[None, :, :], axis=2)
    return d.min(axis=1).mean()


def spacing(F: np.ndarray) -> float:
    n = len(F)
    if n < 2:
        return 0.0
    d = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    di = d.min(axis=1)
    return np.std(di)


def maximum_spread(F: np.ndarray) -> float:
    return np.linalg.norm(F.max(axis=0) - F.min(axis=0))


def build_reference_front(all_fronts: list) -> np.ndarray:
    """Combine all algorithms' final fronts and extract the non-dominated set as
    an approximate 'true' reference Pareto front."""
    from algorithms import fast_non_dominated_sort  # local import to avoid cycle at module load
    F_all = np.vstack(all_fronts)
    fronts = fast_non_dominated_sort(F_all)
    return F_all[fronts[0]]


# ---------------- Inferential statistics ----------------

def wilcoxon_vs_proposed(proposed_scores: np.ndarray, baseline_scores: np.ndarray):
    """Wilcoxon signed-rank test, proposed vs one baseline, across independent runs."""
    try:
        stat, p = wilcoxon(proposed_scores, baseline_scores)
    except ValueError:
        stat, p = np.nan, 1.0
    return stat, p


def friedman_nemenyi(score_matrix: dict):
    """score_matrix: {algo_name: np.array of per-run scalar scores (same length)}.
    Returns (friedman_stat, friedman_p, nemenyi_pvalue_dataframe)."""
    names = list(score_matrix.keys())
    data = np.array([score_matrix[n] for n in names]).T  # (n_runs, n_algos)
    stat, p = friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])
    nemenyi = sp.posthoc_nemenyi_friedman(data)
    nemenyi.columns = names
    nemenyi.index = names
    return stat, p, nemenyi


def vargha_delaney_a12(a: np.ndarray, b: np.ndarray) -> float:
    """A12 effect size: P(a > b) + 0.5*P(a==b)."""
    m, n = len(a), len(b)
    ranks = np.argsort(np.argsort(np.concatenate([a, b]))) + 1
    r1 = ranks[:m].sum()
    a12 = (r1 / m - (m + 1) / 2) / n
    return a12


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))
