"""Independent toy oracles for cross-checking the exact engine.

These are intentionally different algorithms from ``scipy.linalg.expm`` and
from the efficient convolution in ``ctmc.py``:

* ``uniformize`` — Jensen's uniformization of the 25-state generator with a
  strictly controlled truncation remainder;
* ``compound_poisson_marginal`` — direct countable enumeration of a single
  country's compound-Poisson year using exact Poisson mass terms;
* ``capped_sum_marginal`` / ``or_terminal`` — naive per-triple enumeration of
  the three-year capped marginal.

None of these routines are used inside the production computation.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from .ctmc import EQ_STATES


def uniformize(Q: np.ndarray, t: float = 1.0, *,
               remainder_tol: float = 1e-15) -> tuple[np.ndarray, int]:
    """P(t) via uniformization using ``Pj = I + Q/lambda``.

    Returns (estimate of exp(t Q), number of terms used). The truncation
    remainder after n transitions is bounded by the surviving tail mass of the
    Poisson(lambda t) count: ``e^{-lt} sum_{k>n} (lt)^k/k!``, which we upper
    bound geometrically each step and require below ``remainder_tol``.
    """
    lam = float(np.max(-np.diag(Q)))
    if lam <= 0.0:
        return np.eye(Q.shape[0]), 0
    Pj = np.eye(EQ_STATES) + Q / lam
    lt = lam * t
    acc = np.zeros_like(Q)
    pn = math.exp(-lt)                      # n = 0 weight
    term = np.eye(EQ_STATES)
    acc += pn * term
    n = 0
    while True:
        n += 1
        pn *= lt / n
        term = term @ Pj
        acc += pn * term
        ratio = lt / (n + 1)
        if ratio < 1.0 and pn * ratio / (1.0 - ratio) < remainder_tol:
            break
    return acc, n


def compound_poisson_marginal(lam: tuple[float, float, float, float],
                              *,
                              tail_mass_tol: float = 1e-15,
                              max_events: int = 12) -> tuple[np.ndarray, float]:
    """Single-country annual payout-state law by countable enumeration.

    Returns (distribution over states 0..4, truncated event mass). Events of
    severities 1..4 are independent Poisson with rates ``lam``; the year state
    is ``min(sum of severities, 4)``. All count combinations ``n1..n4`` with
    ``n1+n2+n3+n4 <= max_events`` are accumulated exactly; the omitted
    probability (more than ``max_events`` events) is provably below the shown
    tail mass.
    """
    lamb = sum(lam)
    dist = np.zeros(5)
    acc = 0.0
    for n1 in range(max_events + 1):
        for n2 in range(max_events + 1):
            for n3 in range(max_events + 1):
                for n4 in range(max_events + 1):
                    n = n1 + n2 + n3 + n4
                    if n > max_events:
                        continue
                    mass = math.exp(-lamb)
                    for rate, count in zip(lam, (n1, n2, n3, n4)):
                        for i in range(1, count + 1):
                            mass *= rate / i
                    severity = n1 * 1 + n2 * 2 + n3 * 3 + n4 * 4
                    dist[min(severity, 4)] += mass
                    acc += mass
    return dist, 1.0 - acc


def capped_sum_marginal(year: np.ndarray, years: int = 3) -> np.ndarray:
    """Three-year capped marginal by naive triple enumeration (5 states)."""
    term = np.zeros(5)
    for combo in product(range(5), repeat=years):
        mass = 1.0
        for k in combo:
            mass *= float(year[k])
        term[min(sum(combo), 4)] += mass
    return term


def or_terminal(trigger_year: np.ndarray, years: int = 3) -> np.ndarray:
    """Probability an event triggers at least once across ``years``."""
    p_none = 1.0 - float(trigger_year)
    return 1.0 - p_none ** years