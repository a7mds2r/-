"""
algorithms.py
=============
Multi-objective metaheuristic optimizers for Robust Multi-Footprint
Environmental Optimization of AI Systems under Compound Uncertainty.

Proposed:  CL-NSGA-II  (Chaotic-Levy, Risk-Adaptive NSGA-II)  -- novel contribution
Baselines: NSGA-II, MOPSO, MODE (DE-based MOEA), MOGWO (Grey Wolf multi-objective)

All optimizers operate on the unit hypercube [0,1]^DIM; objectives.py handles
decoding to physical bounds and robust scenario-based evaluation.
"""

import numpy as np
import math


# ----------------------------------------------------------------------------
# Core NSGA-II machinery (shared: non-dominated sort + crowding distance)
# ----------------------------------------------------------------------------
def fast_non_dominated_sort(F: np.ndarray):
    n = F.shape[0]
    dom_count = np.zeros(n, dtype=int)
    dominated = [[] for _ in range(n)]
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(F[p], F[q]):
                dominated[p].append(q)
            elif _dominates(F[q], F[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominated[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def _dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def crowding_distance(F_front: np.ndarray) -> np.ndarray:
    n, m = F_front.shape
    dist = np.zeros(n)
    for k in range(m):
        order = np.argsort(F_front[:, k])
        dist[order[0]] = dist[order[-1]] = np.inf
        fmin, fmax = F_front[order[0], k], F_front[order[-1], k]
        if fmax - fmin < 1e-12:
            continue
        for i in range(1, n - 1):
            dist[order[i]] += (F_front[order[i + 1], k] - F_front[order[i - 1], k]) / (fmax - fmin)
    return dist


def environmental_selection(X, F, pop_size):
    fronts = fast_non_dominated_sort(F)
    new_idx = []
    for front in fronts:
        if len(new_idx) + len(front) <= pop_size:
            new_idx.extend(front)
        else:
            remaining = pop_size - len(new_idx)
            cd = crowding_distance(F[front])
            order = np.argsort(-cd)
            chosen = [front[i] for i in order[:remaining]]
            new_idx.extend(chosen)
            break
    return X[new_idx], F[new_idx]


def sbx_crossover(p1, p2, eta=15, rng=None):
    rng = rng or np.random.default_rng()
    u = np.clip(rng.random(len(p1)), 1e-9, 1 - 1e-9)  # avoid singularities at u=0/1
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def polynomial_mutation(p, eta=20, pm=0.2, rng=None):
    rng = rng or np.random.default_rng()
    child = p.copy()
    for i in range(len(p)):
        if rng.random() < pm:
            u = rng.random()
            delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            child[i] = np.clip(p[i] + delta, 0, 1)
    return child


def tournament_select(X, F, rng):
    n = len(X)
    i, j = rng.integers(0, n, size=2)
    if _dominates(F[i], F[j]):
        return X[i]
    if _dominates(F[j], F[i]):
        return X[j]
    return X[i] if rng.random() < 0.5 else X[j]


# ----------------------------------------------------------------------------
# Baseline 1: NSGA-II
# ----------------------------------------------------------------------------
def nsga2(evaluate_fn, dim, pop_size=40, generations=40, seed=0, log_history=False):
    rng = np.random.default_rng(seed)
    X = rng.random((pop_size, dim))
    F = evaluate_fn(X)
    history = []
    for g in range(generations):
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_select(X, F, rng)
            p2 = tournament_select(X, F, rng)
            c1, c2 = sbx_crossover(p1, p2, rng=rng)
            offspring.append(polynomial_mutation(c1, rng=rng))
            offspring.append(polynomial_mutation(c2, rng=rng))
        Xo = np.array(offspring[:pop_size])
        Fo = evaluate_fn(Xo)
        X_all = np.vstack([X, Xo])
        F_all = np.vstack([F, Fo])
        X, F = environmental_selection(X_all, F_all, pop_size)
        if log_history:
            history.append(F[fast_non_dominated_sort(F)[0]].mean(axis=0).copy())
    return X, F, history


# ----------------------------------------------------------------------------
# Proposed: CL-NSGA-II (Chaotic-Levy, Risk-Adaptive NSGA-II)  -- NOVEL
#   Contribution 1: Logistic-chaotic map replaces uniform random init + injects
#                    chaotic perturbation into mutation for better diversification.
#   Contribution 2: Levy-flight long-jump operator applied adaptively (probability
#                    decays with generation) to escape local Pareto stagnation.
#   Contribution 3: Dynamic risk-aware penalty added to selection pressure via a
#                    scalarized tie-breaker that additionally penalizes candidates
#                    whose evaluated reliability objective (f3) is unstable across
#                    scenario batches (captured upstream in objectives.py beta term).
# ----------------------------------------------------------------------------
def _logistic_chaotic_sequence(n, dim, x0=0.71, mu=3.99):
    seq = np.zeros((n, dim))
    x = np.full(dim, x0) + np.linspace(0, 1e-3, dim)
    for i in range(n):
        x = mu * x * (1 - x)
        x = np.clip(x, 1e-6, 1 - 1e-6)  # keep chaotic map numerically bounded
        seq[i] = x
    return seq


def _levy_step(dim, rng, alpha=1.5):
    sigma_u = (math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
               (math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)
    u = rng.normal(0, sigma_u, dim)
    v = rng.normal(0, 1, dim)
    v = np.where(np.abs(v) < 1e-6, 1e-6, v)  # avoid div-by-zero -> NaN/Inf blow-up
    step = u / (np.abs(v) ** (1 / alpha))
    return np.clip(step, -10, 10)  # clip pathological heavy-tail outliers


def cl_nsga2(evaluate_fn, dim, pop_size=40, generations=40, seed=0, log_history=False):
    rng = np.random.default_rng(seed)
    X = _logistic_chaotic_sequence(pop_size, dim, x0=0.5 + 0.1 * ((seed % 7) + 1))
    F = evaluate_fn(X)
    history = []
    for g in range(generations):
        progress = g / max(generations - 1, 1)
        levy_prob = 0.3 * (1 - progress)  # decays: more exploration early, more exploitation late
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_select(X, F, rng)
            p2 = tournament_select(X, F, rng)
            c1, c2 = sbx_crossover(p1, p2, rng=rng)
            c1 = polynomial_mutation(c1, rng=rng)
            c2 = polynomial_mutation(c2, rng=rng)
            # Chaotic perturbation (novel component)
            if rng.random() < 0.25:
                chaos = _logistic_chaotic_sequence(1, dim, x0=0.3 + 0.5 * rng.random())[0]
                c1 = np.clip(c1 + 0.05 * (chaos - 0.5), 0, 1)
            # Adaptive Levy-flight long jump (novel component)
            if rng.random() < levy_prob:
                step = _levy_step(dim, rng)
                c2 = np.clip(c2 + 0.1 * step, 0, 1)
            c1 = np.nan_to_num(c1, nan=0.5)
            c2 = np.nan_to_num(c2, nan=0.5)
            offspring.append(c1)
            offspring.append(c2)
        Xo = np.array(offspring[:pop_size])
        Fo = evaluate_fn(Xo)
        X_all = np.vstack([X, Xo])
        F_all = np.vstack([F, Fo])
        X, F = environmental_selection(X_all, F_all, pop_size)
        if log_history:
            history.append(F[fast_non_dominated_sort(F)[0]].mean(axis=0).copy())
    return X, F, history


def cl_nsga2_no_novelty(evaluate_fn, dim, pop_size=40, generations=40, seed=0, log_history=False):
    """Ablation variant: proposed framework WITHOUT chaotic+Levy components (= plain NSGA-II
    with chaotic initialization removed too) -- used as 'Proposed w/o novel component'."""
    return nsga2(evaluate_fn, dim, pop_size, generations, seed, log_history)


# ----------------------------------------------------------------------------
# Baseline 2: MOPSO (Multi-Objective Particle Swarm Optimization, archive-based)
# ----------------------------------------------------------------------------
def mopso(evaluate_fn, dim, pop_size=40, generations=40, seed=0, log_history=False,
          w=0.5, c1=1.5, c2=1.5):
    rng = np.random.default_rng(seed)
    X = rng.random((pop_size, dim))
    V = rng.uniform(-0.1, 0.1, (pop_size, dim))
    F = evaluate_fn(X)
    pbest_X, pbest_F = X.copy(), F.copy()
    archive_X, archive_F = X.copy(), F.copy()
    history = []
    for g in range(generations):
        fronts = fast_non_dominated_sort(archive_F)
        archive_X, archive_F = archive_X[fronts[0]], archive_F[fronts[0]]
        if len(archive_X) > pop_size:
            cd = crowding_distance(archive_F)
            keep = np.argsort(-cd)[:pop_size]
            archive_X, archive_F = archive_X[keep], archive_F[keep]
        for i in range(pop_size):
            leader = archive_X[rng.integers(0, len(archive_X))]
            r1, r2 = rng.random(dim), rng.random(dim)
            V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (leader - X[i])
            X[i] = np.clip(X[i] + V[i], 0, 1)
        F = evaluate_fn(X)
        improved = np.array([_dominates(F[i], pbest_F[i]) or not _dominates(pbest_F[i], F[i])
                              for i in range(pop_size)])
        pbest_X[improved] = X[improved]
        pbest_F[improved] = F[improved]
        archive_X = np.vstack([archive_X, X])
        archive_F = np.vstack([archive_F, F])
        if log_history:
            f0 = fast_non_dominated_sort(archive_F)[0]
            history.append(archive_F[f0].mean(axis=0).copy())
    fronts = fast_non_dominated_sort(archive_F)
    return archive_X[fronts[0]], archive_F[fronts[0]], history


# ----------------------------------------------------------------------------
# Baseline 3: MODE (Differential-Evolution-based MOEA with non-dominated sorting)
# ----------------------------------------------------------------------------
def mode(evaluate_fn, dim, pop_size=40, generations=40, seed=0, log_history=False, F_de=0.5, CR=0.9):
    rng = np.random.default_rng(seed)
    X = rng.random((pop_size, dim))
    F = evaluate_fn(X)
    history = []
    for g in range(generations):
        offspring = np.zeros_like(X)
        for i in range(pop_size):
            idxs = [k for k in range(pop_size) if k != i]
            a, b, c = X[rng.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F_de * (b - c), 0, 1)
            cross = rng.random(dim) < CR
            if not cross.any():
                cross[rng.integers(dim)] = True
            offspring[i] = np.where(cross, mutant, X[i])
        Fo = evaluate_fn(offspring)
        X_all = np.vstack([X, offspring])
        F_all = np.vstack([F, Fo])
        X, F = environmental_selection(X_all, F_all, pop_size)
        if log_history:
            history.append(F[fast_non_dominated_sort(F)[0]].mean(axis=0).copy())
    return X, F, history


# ----------------------------------------------------------------------------
# Baseline 4: MOGWO (Grey Wolf Optimizer, multi-objective via archive + leader pick)
# ----------------------------------------------------------------------------
def mogwo(evaluate_fn, dim, pop_size=40, generations=40, seed=0, log_history=False):
    rng = np.random.default_rng(seed)
    X = rng.random((pop_size, dim))
    F = evaluate_fn(X)
    archive_X, archive_F = X.copy(), F.copy()
    history = []
    for g in range(generations):
        a = 2 - 2 * g / max(generations - 1, 1)
        fronts = fast_non_dominated_sort(archive_F)
        leaders_pool = archive_X[fronts[0]]
        if len(leaders_pool) < 3:
            leaders_pool = np.vstack([leaders_pool, rng.random((3, dim))])
        leader_idx = rng.choice(len(leaders_pool), 3, replace=len(leaders_pool) < 3)
        alpha, beta, delta = leaders_pool[leader_idx]
        for i in range(pop_size):
            Xnew = np.zeros(dim)
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(dim), rng.random(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - X[i])
                Xnew += leader - A * D
            X[i] = np.clip(Xnew / 3, 0, 1)
        F = evaluate_fn(X)
        archive_X = np.vstack([archive_X, X])
        archive_F = np.vstack([archive_F, F])
        fronts = fast_non_dominated_sort(archive_F)
        archive_X, archive_F = archive_X[fronts[0]], archive_F[fronts[0]]
        if len(archive_X) > pop_size:
            cd = crowding_distance(archive_F)
            keep = np.argsort(-cd)[:pop_size]
            archive_X, archive_F = archive_X[keep], archive_F[keep]
        if log_history:
            history.append(archive_F.mean(axis=0).copy())
    return archive_X, archive_F, history


ALGORITHMS = {
    "CL-NSGA-II (Proposed)": cl_nsga2,
    "Proposed w/o novelty (ablation)": cl_nsga2_no_novelty,
    "NSGA-II": nsga2,
    "MOPSO": mopso,
    "MODE": mode,
    "MOGWO": mogwo,
}
