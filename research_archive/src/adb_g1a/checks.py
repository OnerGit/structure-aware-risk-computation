"""Invariant/validation helpers used by the test suite and diagnostics."""

from __future__ import annotations

import math

import numpy as np


def validate_generator(Q: np.ndarray, *, label: str = "G",
                       atol_neg: float = 1e-13,
                       atol_rowsum: float = 1e-10) -> None:
    """A generator must have zero row sums and nonnegative off-diagonals."""
    n = Q.shape[0]
    assert Q.shape == (n, n)
    if not np.allclose(Q.sum(axis=1), 0.0, atol=atol_rowsum):
        raise ValueError(f"{label}: row sums not zero "
                         f"(max {np.abs(Q.sum(axis=1)).max():.3e})")
    off = Q[~np.eye(n, dtype=bool)]
    if not np.all(off >= -atol_neg):
        raise ValueError(f"{label}: negative off-diagonal "
                         f"(min {off.min():.3e})")
    diag = np.diag(Q)
    if not np.all(diag <= atol_neg):
        raise ValueError(f"{label}: positive diagonal mass")


def validate_distribution(dist: np.ndarray, *, label: str = "dist",
                          atol_neg: float = 1e-13,
                          atol_sum: float = 1e-12) -> float:
    """Nonnegativity and unit-mass; returns the unnormalized sum.

    The mass check uses a strict absolute tolerance: `rtol` is explicitly 0.0
    so that the effective tolerance is `atol_sum` and nothing more (a relative
    term would inflate the allowed error near mass 1.0).
    """
    if dist.dtype == object:
        dist = np.asarray(dist, dtype=float)
    if not np.all(dist >= -atol_neg):
        raise ValueError(f"{label}: negative probability "
                         f"(min {float(np.min(dist)):.3e})")
    s = float(np.sum(dist))
    if not math.isclose(s, 1.0, rel_tol=0.0, abs_tol=atol_sum):
        raise ValueError(f"{label}: mass {s:.17f} != 1 "
                         f"(outside absolute tolerance {atol_sum:g})")
    return s


def validate_loss_bounds(losses: np.ndarray, lo: float, hi: float,
                         *, label: str = "loss", atol: float = 1e-6) -> None:
    mn, mx = float(np.min(losses)), float(np.max(losses))
    if mn < lo - atol or mx > hi + atol:
        raise ValueError(f"{label}: loss range [{mn:.6g}, {mx:.6g}] "
                         f"outside [{lo}, {hi}]")


def hazard_point_probs(t1: float, t2: float, t3: float, t4: float,
                       *, reversed_order: bool = False) -> tuple[float, ...]:
    """Choice point for the wrong-CDF-order negative control."""
    if reversed_order:
        return (1.0 - t4, t4 - t3, t3 - t2, t2 - t1, t1)
    return (1.0 - t1, t1 - t2, t2 - t3, t3 - t4, t4)


def validate_poisson_consistency(lamb: float, lam: tuple[float, ...],
                                 p: tuple[float, ...], *,
                                 atol: float = 1e-12) -> None:
    """Resynthesize the point probabilities from the Poisson rates."""
    p0_from_lambda = math.exp(-lamb)
    if not math.isclose(p0_from_lambda, p[0], rel_tol=0.0, abs_tol=atol):
        raise ValueError("p0 != exp(-Lambda)")
    lam1, lam2, lam3, lam4 = lam
    p1 = p[0] * lam1
    p2 = p[0] * (lam2 + lam1 * lam1 / 2.0)
    p3 = p[0] * (lam3 + lam1 * lam2 + lam1**3 / 6.0)
    for got, want, tag in ((p1, p[1], "p1"), (p2, p[2], "p2"), (p3, p[3], "p3")):
        if not math.isclose(got, want, rel_tol=0.0, abs_tol=atol):
            raise ValueError(f"{tag}: resynthesized {got:.17g} != {want:.17g}")


__all__ = [
    "validate_generator", "validate_distribution", "validate_loss_bounds",
    "hazard_point_probs", "validate_poisson_consistency",
]