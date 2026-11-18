"""Annual 100-state law adapter built on the frozen G1-A engine.

Rebuilds, per scenario, the exact annual 100-state distribution and every
derived object the samplers need: state-coding arrays, inverse-CDF, zero/positive
payout masks, stratified layer probabilities and the IS proposal ``q`` with the
loss-tilt. All distributions come from ``adb_g1a`` frozen functions; nothing is
re-estimated or re-scaled here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adb_g1a.ctmc import (eq_generator, annual_eq_distribution,
                          annual_joint_distribution, rain_annual_probabilities,
                          state_arrays)
from adb_g1a.model import Scenario, scenario_parameters


@dataclass(frozen=True)
class AnnualLaw:
    scenario: Scenario
    probs: np.ndarray            # (100,) exact annual probabilities
    a: np.ndarray                # (100,)
    b: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    year_loss: np.ndarray        # (100,) annual portfolio payout USD (scenario covers)
    cdf: np.ndarray              # (100,) cumulative (edges safe: last = 1.0)
    zero_mask: np.ndarray        # annual states with year_loss == 0
    pos_mask: np.ndarray         # annual states with year_loss > 0
    pos_idx: np.ndarray          # indices of positive-payout states
    pos_cdf_reduced: np.ndarray  # CDF over positive states only (len = #pos)
    zero_cdf: np.ndarray
    p0: float                    # P(annual portfolio payout == 0)
    tilt: np.ndarray = field(default=None)   # loss-tilted annual law
    q: np.ndarray = field(default=None)      # 0.5 p + 0.5 tilt
    q_cdf: np.ndarray = field(default=None)
    ep_year_loss: float = field(default=0.0)

    n_states: int = 100

    def state_from_u(self, u):
        """Invert the annual CDF on a vector of uniforms (strict boundaries)."""
        ua = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        idx = np.searchsorted(self.cdf, ua, side="left")
        return np.clip(idx, 0, self.n_states - 1).astype(np.int64)

    def state_from_q(self, u):
        ua = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        idx = np.searchsorted(self.q_cdf, ua, side="left")
        return np.clip(idx, 0, self.n_states - 1).astype(np.int64)

    def pos_state_from_u(self, u):
        ua = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        j = np.searchsorted(self.pos_cdf_reduced, ua, side="left")
        j = np.clip(j, 0, self.pos_cdf_reduced.size - 1)
        return self.pos_idx[j]

    @property
    def layer_weights(self) -> np.ndarray:
        """w_k = p0^(k-1) (1 - p0) for k = 1,2,3 (first-payout year strata)."""
        p0 = self.p0
        return np.array([(1.0 - p0), p0 * (1.0 - p0), p0 * p0 * (1.0 - p0)])


_ZERO_CACHE: dict[str, AnnualLaw] = {}


def build_annual_law(sc: Scenario) -> AnnualLaw:
    """Cache-per-scenario builder; distribution identical to G1-A (eq.5-6, eq.25)."""
    key = (sc.key, sc.m, sc.rain_common_share, sc.rain_rate_multiplier)
    cached = _ZERO_CACHE.get(key)
    if cached is not None:
        return cached
    h1, h2, c, mu1, mu2, common_rate = scenario_parameters(sc)
    Q = eq_generator(h1.lam, h2.lam, c)
    eq25 = annual_eq_distribution(Q)
    rain4 = rain_annual_probabilities(mu1, mu2, common_rate)
    probs = annual_joint_distribution(eq25, rain4)
    a, b, r1, r2 = state_arrays()
    year_loss = (sc.eq_cover * (a / 4.0) + sc.rain_cover * r1
                 + sc.eq_cover * (b / 4.0) + sc.rain_cover * r2)
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    zero_mask = np.abs(year_loss) <= 1e-9
    pos_mask = ~zero_mask
    p0 = float(probs[zero_mask].sum())
    zero_cdf = np.cumsum(probs * zero_mask)
    zero_cdf[-1] = 1.0 if p0 > 0 else 0.0
    pos_idx = np.flatnonzero(pos_mask)
    pos_probs = probs[pos_idx]
    pos_probs = pos_probs / pos_probs.sum()
    pos_cdf_reduced = np.cumsum(pos_probs)
    pos_cdf_reduced[-1] = 1.0
    ep_year_loss = float(np.sum(probs * year_loss))
    tilt = probs * year_loss / ep_year_loss
    q = 0.5 * probs + 0.5 * tilt
    q_cdf = np.cumsum(q)
    q_cdf[-1] = 1.0
    law = AnnualLaw(scenario=sc, probs=probs, a=a, b=b, r1=r1, r2=r2,
                    year_loss=year_loss, cdf=cdf, zero_mask=zero_mask,
                    pos_mask=pos_mask, pos_idx=pos_idx,
                    pos_cdf_reduced=pos_cdf_reduced, zero_cdf=zero_cdf,
                    p0=p0, tilt=tilt, q=q, q_cdf=q_cdf,
                    ep_year_loss=ep_year_loss)
    _ZERO_CACHE[key] = law
    return law


def default_layers(law: AnnualLaw, n: int) -> np.ndarray:
    """Pre-fixed proportional allocation n_k = round(n * w_k / sum(w)), 3 strata.

    The official layer probabilities ``w_k = p0^(k-1)(1-p0)`` (k=1,2,3) sum to
    ``1 - p0^3`` because the zero-payout mass contributes nothing; allocation
    is proportional over the *three* strata (normalized ``w``), min 1 per layer,
    the rounding remainder assigned in descending order of fractional part.
    Rule recorded in docs/G1_B_METHOD_AND_STATISTICAL_CONTRACT.md; never
    re-chosen after the data are seen.
    """
    w = law.layer_weights
    wnorm = w / w.sum()
    frac = n * wnorm
    n_k = np.floor(frac).astype(int)
    n_k = np.maximum(n_k, 1)
    rem = n - int(n_k.sum())
    order = np.argsort(-(frac - np.floor(frac)))
    for i in range(rem):
        n_k[order[i % 3]] += 1
    n_k[0] += int(n - n_k.sum())  # exact-sum guard
    assert int(n_k.sum()) == n
    return n_k.astype(int)