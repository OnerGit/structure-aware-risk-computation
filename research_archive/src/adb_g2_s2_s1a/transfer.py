"""Checkpoint A: independent n=2 transfer contraction and country-major
post-hoc factorization diagnostics (numpy SVD only)."""

from __future__ import annotations

import numpy as np

from .laws import (all_terminal_states, decode_terminal, merge_state,
                   terminal_code)


def transfer_matrix(annual: np.ndarray) -> np.ndarray:
    """T[s,s'] = sum_y p(y) 1{s'=merge(s,y)}. Row-stochastic (own build)."""
    T = np.zeros((100, 100))
    for s in range(100):
        for y in range(100):
            T[s, merge_state(s, y)] += annual[y]
    return T


def terminal_law(T: np.ndarray, years: int = 3) -> np.ndarray:
    pi = np.zeros(100)
    pi[0] = 1.0
    for _ in range(years):
        pi = pi @ T
    return pi


def contraction_targets(annual: np.ndarray,
                        g: dict[str, np.ndarray]) -> dict[str, float]:
    pi = terminal_law(transfer_matrix(annual))
    return {k: float(np.sum(pi * g[k])) for k in g}


def country_major_matrix(p_term: np.ndarray) -> np.ndarray:
    """Reshape the terminal distribution to the 10x10 country-major layout
    (per-country code = a*2 + r, 0..9)."""
    M = np.zeros((10, 10))
    for x in range(100):
        a1, a2, r1, r2 = decode_terminal(x)
        M[a1 * 2 + r1, a2 * 2 + r2] = p_term[x]
    return M


def numeric_rank(svals: np.ndarray, tol: float) -> int:
    if svals.size == 0:
        return 0
    return int(np.count_nonzero(svals > tol * max(svals.max(), 1e-300)))


def svd_diagnostics(p_term: np.ndarray,
                    g: dict[str, np.ndarray],
                    tol_primary: float = 1e-10,
                    tol_secondary: float = 1e-8) -> dict:
    """Post-hoc (solved-table) factorization diagnostics (Checkpoint A, part 2)."""
    M = country_major_matrix(p_term)
    um, sm, vtm = np.linalg.svd(M, full_matrices=False)
    W = {}
    for k, gv in g.items():
        W[k] = country_major_matrix(p_term * gv)
    svals = {k: np.linalg.svd(W[k], full_matrices=False)[1] for k in g}
    scalar = {k: float(np.sum(p_term * g[k])) for k in g}
    out = {
        "p_term_country_major_svals": [float(x) for x in sm],
        "p_term_rank_tol1e10": numeric_rank(sm, tol_primary),
        "p_term_rank_tol1e8": numeric_rank(sm, tol_secondary),
        "W_svals": {k: [float(x) for x in s] for k, s in svals.items()},
        "W_rank_tol1e10": {k: numeric_rank(s, tol_primary) for k, s in svals.items()},
        "W_rank_tol1e8": {k: numeric_rank(s, tol_secondary) for k, s in svals.items()},
        "W_scalar_contraction": scalar,
        "note": "post-hoc factorization of a solved exact table; not an "
                "algorithm that avoids computing the table",
    }
    return out