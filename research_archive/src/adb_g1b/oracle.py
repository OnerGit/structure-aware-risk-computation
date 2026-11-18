"""Deterministic small oracles for SAMPLER correctness validation.

* ``exact_stratum_means``: per-layer conditional means computed by exact
  convolution of conditioned annual laws (independent of simulation).
* ``is_analytic_consistency``: exact enumeration identities for the IS proposal
  (E_q[W] = 1, and the single-year weighted estimator against E_p).
* ``plotcdf`` utility is intentionally absent: no plotting in a benchmark.
"""

from __future__ import annotations

import numpy as np

from adb_g1a.ctmc import build_merge_table, convolve_year
from adb_g1b.annual_law import AnnualLaw
from adb_g1b.samplers import TERM_KEYS


def _expect(term_dist: np.ndarray, law: AnnualLaw) -> dict[str, float]:
    """Exact five-metric expectation of a 100-state term distribution."""
    from adb_g1a.model import PRINCIPAL
    a, b, r1, r2 = law.a, law.b, law.r1, law.r2
    l1 = law.scenario.eq_cover * (a / 4.0) + law.scenario.rain_cover * r1
    l2 = law.scenario.eq_cover * (b / 4.0) + law.scenario.rain_cover * r2
    l = l1 + l2
    return {
        "term_expected_principal_loss_usd": float(np.sum(term_dist * l)),
        "term_joint_attachment_probability": float(np.sum(
            term_dist[(l1 > 0) & (l2 > 0)])),
        "term_portfolio_loss_exceedance_probability": float(np.sum(
            term_dist[l > PRINCIPAL])),
        "term_portfolio_stop_loss_usd": float(np.sum(
            term_dist * np.maximum(l - PRINCIPAL, 0.0))),
        "term_both_bonds_exhausted_probability": float(np.sum(
            term_dist[(a == 4) & (r1 == 1) & (b == 4) & (r2 == 1)])),
    }


def exact_stratum_means(law: AnnualLaw):
    """(layer_weights, per-layer exact metric means, weighted combination)."""
    merge = build_merge_table()
    p = law.probs
    pos = np.full(100, 0.0)
    pos[law.pos_idx] = law.probs[law.pos_idx]
    pos = pos / pos.sum()
    zero = np.zeros(100)
    zero[0] = 1.0
    layer_laws = {
        1: convolve_year(convolve_year(pos, p, merge), p, merge),
        2: convolve_year(convolve_year(zero, pos, merge), p, merge),
        3: convolve_year(convolve_year(zero, zero, merge), pos, merge),
    }
    means = {k: _expect(layer_laws[k], law) for k in (1, 2, 3)}
    weights = law.layer_weights
    combined = {}
    for m in TERM_KEYS:
        combined[m] = float(sum(weights[k - 1] * means[k][m] for k in (1, 2, 3)))
    return weights, means, combined


def is_analytic_consistency(law: AnnualLaw) -> dict:
    """Exact enumerations: E_q[W] = 1 and, for the single-year loss target,
    the IS weighted sum of p-weight equals E_p[loss]."""
    q = law.q
    W = law.probs / q                      # p(s)/q(s) over the support
    eq_w = float(np.sum(q * W))            # E_q[W] == 1 analytically
    lew = float(np.sum(law.probs * law.year_loss))   # E_p[year_loss]
    return {
        "E_q_W": eq_w,
        "E_p_year_loss": float(lew),
        "q_mass": float(q.sum()),
        "support_covered": bool(np.all(law.probs > 0) == np.all(q > 0)),
        "min_ratio": float(np.min(law.probs / q)),
        "max_ratio": float(np.max(law.probs / q)),
    }