"""CTMC, precipitation law and annual 100-state distribution (Brief steps 4-6).

Probability convention (explicit and tested): probabilities are row vectors,
transition matrix ``P = expm(Q)`` is row-stochastic, and the one-year joint
earthquake distribution is the first row of ``expm(Q)`` (start state (0,0)).
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
from scipy.linalg import expm

# ---- state algebra ---------------------------------------------------------

A_STATES = 5              # earthquake quarter steps 0..4 per country
EQ_STATES = 25            # (a, b)
RAIN_STATES_LIST = [(0, 0), (1, 0), (0, 1), (1, 1)]      # (r1, r2)
RAIN_IDX = {rs: i for i, rs in enumerate(RAIN_STATES_LIST)}
RAIN_COUNT = len(RAIN_STATES_LIST)                       # 4
ANNUAL_STATES = EQ_STATES * RAIN_COUNT                   # 100


def eq_idx(a: int, b: int) -> int:
    return a * A_STATES + b


def annual_idx(a: int, b: int, r1: int, r2: int) -> int:
    return eq_idx(a, b) * RAIN_COUNT + RAIN_IDX[(r1, r2)]


EQ_STATE_ARRAYS = np.array(
    [[a, b] for a in range(A_STATES) for b in range(A_STATES)], dtype=np.int64)
RAIN_ARR = np.array(RAIN_STATES_LIST, dtype=np.int64)


def state_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(A, B, R1, R2) arrays for all 100 observable states."""
    a = np.repeat(EQ_STATE_ARRAYS[:, 0], RAIN_COUNT)
    b = np.repeat(EQ_STATE_ARRAYS[:, 1], RAIN_COUNT)
    r1 = np.tile(RAIN_ARR[:, 0], EQ_STATES)
    r2 = np.tile(RAIN_ARR[:, 1], EQ_STATES)
    return a, b, r1, r2


# ---- earthquake CTMC -------------------------------------------------------

def eq_generator(lam1: tuple[float, float, float, float],
                 lam2: tuple[float, float, float, float],
                 c: float) -> np.ndarray:
    """25x25 generator ``G`` for the capped annual earthquake process.

    Transitions (Brief step 4):
      (a,b) -> (min(a+k,4), b)                    rate (L1-c) w1_k
      (a,b) -> (a, min(b+j,4))                    rate (L2-c) w2_j
      (a,b) -> (min(a+k,4), min(b+j,4))           rate c w1_k w2_j
    A transition is emitted only when the state actually changes; events that
    keep the (already capped) state unchanged contribute nothing. The diagonal
    is the negative of the state-changing outflow so that row sums are zero.
    """
    l1 = np.asarray(lam1, dtype=float)
    l2 = np.asarray(lam2, dtype=float)
    L1 = float(l1.sum())
    L2 = float(l2.sum())
    w1 = l1 / L1
    w2 = l2 / L2
    Q = np.zeros((EQ_STATES, EQ_STATES))
    for a in range(A_STATES):
        for b in range(A_STATES):
            s = eq_idx(a, b)
            if a < 4:
                rate = max(L1 - c, 0.0)
                for k in range(4):
                    ap = min(a + k + 1, 4)
                    Q[s, eq_idx(ap, b)] += rate * w1[k]
            if b < 4:
                rate = max(L2 - c, 0.0)
                for j in range(4):
                    bp = min(b + j + 1, 4)
                    Q[s, eq_idx(a, bp)] += rate * w2[j]
            rate = max(c, 0.0)
            for k in range(4):
                for j in range(4):
                    ap = min(a + k + 1, 4)
                    bp = min(b + j + 1, 4)
                    if (ap, bp) != (a, b):
                        Q[s, eq_idx(ap, bp)] += rate * w1[k] * w2[j]
    for s in range(EQ_STATES):
        Q[s, s] = -Q[s].sum()
    return Q


def annual_eq_distribution(Q: np.ndarray) -> np.ndarray:
    """One-year joint earthquake distribution, row convention, start (0,0)."""
    P = expm(Q)
    return np.asarray(P[0], dtype=float)


# ---- precipitation law (Brief step 5) --------------------------------------

def rain_annual_probabilities(mu1: float, mu2: float, common_rate: float) -> np.ndarray:
    """Four-state annual precipitation law ``[00,10,01,11]`` (eq. 5-6)."""
    p00 = math.exp(-(mu1 + mu2 - common_rate))
    p10 = math.exp(-mu2) - p00
    p01 = math.exp(-mu1) - p00
    p11 = 1.0 - p00 - p10 - p01
    return np.array([p00, p10, p01, p11], dtype=float)


def base_rain_rates(q: float = 0.02) -> tuple[float, float]:
    """Annual precipitation Poisson rate ``mu = -ln(1-q)`` per country."""
    mu = -math.log(1.0 - q)
    return mu, mu


# ---- annual 100-state distribution (Brief step 6) --------------------------

def annual_joint_distribution(eq25: np.ndarray, rain4: np.ndarray) -> np.ndarray:
    """100-state annual distribution (earthquake x rain, independent)."""
    return np.outer(np.asarray(eq25), np.asarray(rain4)).ravel()


# ---- capped merge table (Brief step 7) -------------------------------------

def build_merge_table() -> np.ndarray:
    """``merge[t, s]`` = terminal state after adding annual state ``s`` to ``t``.

    Capping is applied per step, which equals capping the final sum and the
    rain trigger is a Boolean OR (both order- and group-invariant merges).
    """
    ann_a, ann_b, ann_r1, ann_r2 = state_arrays()
    tab = np.zeros((ANNUAL_STATES, ANNUAL_STATES), dtype=np.int64)
    for t in range(ANNUAL_STATES):
        ta, tb, tr1, tr2 = ann_a[t], ann_b[t], ann_r1[t], ann_r2[t]
        for s in range(ANNUAL_STATES):
            tab[t, s] = annual_idx(
                min(ta + ann_a[s], 4), min(tb + ann_b[s], 4),
                int(tr1 or ann_r1[s]), int(tr2 or ann_r2[s]))
    return tab


def convolve_year(dist: np.ndarray, annual: np.ndarray,
                  merge: np.ndarray) -> np.ndarray:
    """Advances one independent year (exact finite convolution)."""
    out = np.zeros_like(dist)
    for t in range(ANNUAL_STATES):
        p = dist[t]
        if p == 0.0:
            continue
        targets = merge[t]
        out += p * np.bincount(targets, weights=annual, minlength=ANNUAL_STATES)
    return out


def three_year_terminal_distribution(annual: np.ndarray) -> np.ndarray:
    """Exact three-year capped term distribution from the zero state."""
    merge = build_merge_table()
    dist = np.zeros(ANNUAL_STATES)
    dist[annual_idx(0, 0, 0, 0)] = 1.0
    for _ in range(3):
        dist = convolve_year(dist, annual, merge)
    return dist


# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    pass