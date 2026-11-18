"""Five benchmark scenarios and the five term targets (Brief steps 8-9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ctmc import (annual_eq_distribution, annual_joint_distribution,
                   annual_idx, base_rain_rates, eq_generator,
                   rain_annual_probabilities, state_arrays,
                   three_year_terminal_distribution)
from .hazard import HAZARDS, common_shock_rate

PRINCIPAL = 80_000_000.0             # USD per country, term principal
RAIN_TRIGGER_PROB = 0.02             # annual precipitation trigger probability q
THRESHOLD = PRINCIPAL                # T = USD 80m for exceedance/stop-loss
YEARS = 3
EQ_COVER_BASE = 70_000_000.0
RAIN_COVER_BASE = 10_000_000.0

TARGET_KEYS = [
    "term_expected_principal_loss_usd",
    "term_joint_attachment_probability",
    "term_portfolio_loss_exceedance_probability",
    "term_portfolio_stop_loss_usd",
    "term_both_bonds_exhausted_probability",
]

DEFAULT_RAIN_SHARE = 0.25


@dataclass(frozen=True)
class Scenario:
    key: str
    m: float
    eq_cover: float
    rain_cover: float
    rain_common_share: float
    rain_rate_multiplier: float = 1.0
    description: str = ""

    @property
    def max_single_country_loss(self) -> float:
        return self.eq_cover + self.rain_cover


SCENARIOS = (
    Scenario("adb-base", m=1.0, eq_cover=EQ_COVER_BASE, rain_cover=RAIN_COVER_BASE,
             rain_common_share=DEFAULT_RAIN_SHARE,
             description="official base case"),
    Scenario("adb-independent", m=0.0, eq_cover=EQ_COVER_BASE,
             rain_cover=RAIN_COVER_BASE, rain_common_share=0.0,
             description="no shared events: m=0 and zero precipitation shared share"),
    Scenario("adb-common-shock", m=2.0, eq_cover=EQ_COVER_BASE,
             rain_cover=RAIN_COVER_BASE, rain_common_share=DEFAULT_RAIN_SHARE,
             description="shared earthquake rate doubled (m=2)"),
    Scenario("adb-rain-stress", m=1.0, eq_cover=EQ_COVER_BASE,
             rain_cover=RAIN_COVER_BASE, rain_common_share=DEFAULT_RAIN_SHARE,
             rain_rate_multiplier=2.0,
             description="precipitation Poisson rates doubled, 25% shared share fixed"),
    Scenario("adb-rain-cover", m=1.0, eq_cover=60_000_000.0,
             rain_cover=20_000_000.0, rain_common_share=DEFAULT_RAIN_SHARE,
             description="sub-limits changed to USD 60m earthquake + USD 20m rain"),
)


def scenario_parameters(sc: Scenario):
    """Per-scenario hazard configuration (identical probability law, covers vary)."""
    h1 = HAZARDS["Kyrgyz"]
    h2 = HAZARDS["Tajik"]
    c = common_shock_rate(h1, h2, sc.m)
    mu1, mu2 = base_rain_rates(q=RAIN_TRIGGER_PROB)
    mu1 *= sc.rain_rate_multiplier
    mu2 *= sc.rain_rate_multiplier
    common_rate = sc.rain_common_share * min(mu1, mu2)
    return h1, h2, c, mu1, mu2, common_rate


def compute_scenario(sc: Scenario) -> dict[str, float]:
    """Compute the five term targets for one scenario (deterministic, exact)."""
    h1, h2, c, mu1, mu2, common_rate = scenario_parameters(sc)
    Q = eq_generator(h1.lam, h2.lam, c)
    eq25 = annual_eq_distribution(Q)
    rain4 = rain_annual_probabilities(mu1, mu2, common_rate)
    annual = annual_joint_distribution(eq25, rain4)
    terminal = three_year_terminal_distribution(annual)
    a, b, r1, r2 = state_arrays()
    l1 = sc.eq_cover * (a / 4.0) + sc.rain_cover * r1
    l2 = sc.eq_cover * (b / 4.0) + sc.rain_cover * r2
    l = l1 + l2
    el = float(np.sum(terminal * l))
    joint = float(np.sum(terminal[(l1 > 0.0) & (l2 > 0.0)]))
    exceed = float(np.sum(terminal[l > THRESHOLD]))
    stop = float(np.sum(terminal * np.maximum(l - THRESHOLD, 0.0)))
    both_exh = float(np.sum(terminal[(l1 == THRESHOLD) & (l2 == THRESHOLD)]))
    return {
        "term_expected_principal_loss_usd": el,
        "term_joint_attachment_probability": joint,
        "term_portfolio_loss_exceedance_probability": exceed,
        "term_portfolio_stop_loss_usd": stop,
        "term_both_bonds_exhausted_probability": both_exh,
    }


def compute_all_scenarios() -> dict[str, dict[str, float]]:
    return {sc.key: compute_scenario(sc) for sc in SCENARIOS}


def zero_state_index() -> int:
    return annual_idx(0, 0, 0, 0)