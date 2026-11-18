"""Official hazard inputs and exact Poisson-rate derivation (Brief pp. 2-3).

The four tail probabilities ``t1..t4`` are direct inputs as published. The
compound-Poisson rates are derived at full double precision; the rounded
``Table 2`` print and the contextual annual expected-loss percentages are
carried here only as independently verifiable cross-checks, never as inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Official inputs
# ---------------------------------------------------------------------------

# t1..t4: annual probability that aggregate earthquake payout reaches
# at least 25%/50%/75%/100% of the sub-limit (Brief Table 1).
TAIL_PROBS = {
    "Kyrgyz": (0.03883, 0.02331, 0.01790, 0.01184),
    "Tajik": (0.03925, 0.01743, 0.01332, 0.01043),
}

# Contextual published annual EL (percent of the USD 70m sub-limit).
# Secondary cross-check only -- never used to derive rates.
PUBLISHED_ANNUAL_EL_PERCENT = {"Kyrgyz": 2.296, "Tajik": 2.011}

# Conditional trigger probabilities implied by the challenge's Brief P.2
# ("Tajikistan given Kyrgyz Republic is 7.5%; Kyrgyz given Tajikistan 7.4%").
P_TJK_GIVEN_KYR = 0.075
P_KYR_GIVEN_TJK = 0.074


@dataclass(frozen=True)
class CountryHazard:
    """All quantities for one country derived from its tail probabilities."""

    name: str
    tails: tuple[float, float, float, float]
    p: tuple[float, float, float, float, float]  # P(state 0..4), p4 = t4
    Lambda: float
    lam: tuple[float, float, float, float]       # compound-Poisson rates
    weights: tuple[float, float, float, float]   # w_k = lambda_k / Lambda


def derive_hazard(name: str, tails: tuple[float, float, float, float]) -> CountryHazard:
    """Derive point probabilities and full-precision Poisson rates (eq. 2-3)."""
    t1, t2, t3, t4 = tails
    p0 = 1.0 - t1
    p1 = t1 - t2
    p2 = t2 - t3
    p3 = t3 - t4
    p4 = t4
    lamb = -math.log(p0)
    lam1 = p1 / p0
    lam2 = p2 / p0 - lam1 * lam1 / 2.0
    lam3 = p3 / p0 - lam1 * lam2 - lam1**3 / 6.0
    lam4 = lamb - lam1 - lam2 - lam3
    lam = (lam1, lam2, lam3, lam4)
    if lam4 < 0.0:
        raise ValueError(
            f"{name}: lambda4 < 0 ({lam4:.6e}); tails would imply a "
            "negative exhaustion rate -- invalid input")
    weights = tuple(l / lamb for l in lam)
    hazard = CountryHazard(
        name=name,
        tails=tails,
        p=(p0, p1, p2, p3, p4),
        Lambda=lamb,
        lam=lam,
        weights=weights,
    )
    validate_hazard(hazard)
    return hazard


def validate_hazard(h: CountryHazard) -> None:
    """Structural validation with exact error messages (test/audit checks)."""
    problems = []
    if any(x < 0.0 or x > 1.0 for x in h.p):
        problems.append("point probabilities outside [0,1]")
    if abs(sum(h.p) - 1.0) > 1e-12:
        problems.append(f"p sums to {sum(h.p):.17g}")
    if not (h.Lambda > 0.0):
        problems.append("Lambda <= 0")
    if any(x < -1e-13 for x in h.lam):
        problems.append("negative severity rates")
    if abs(sum(h.lam) - h.Lambda) > 1e-12 * max(1.0, h.Lambda):
        problems.append("sum(lambda) != Lambda")
    if any(x < -1e-13 or x > 1.0 + 1e-13 for x in h.weights):
        problems.append("weights not in [0,1]")
    if abs(sum(h.weights) - 1.0) > 1e-12:
        problems.append("sum(weights) != 1")
    if problems:
        raise ValueError(f"{h.name}: invalid hazard: {'; '.join(problems)}")


HAZARDS = {
    "Kyrgyz": derive_hazard("Kyrgyz", TAIL_PROBS["Kyrgyz"]),
    "Tajik": derive_hazard("Tajik", TAIL_PROBS["Tajik"]),
}

KYRGYZ = HAZARDS["Kyrgyz"]
TAJIK = HAZARDS["Tajik"]


def common_shock_rate(h1: CountryHazard, h2: CountryHazard, m: float) -> float:
    """Shared earthquake rate ``c = m*(0.075*Lambda1 + 0.074*Lambda2)/2``."""
    return m * (P_TJK_GIVEN_KYR * h1.Lambda + P_KYR_GIVEN_TJK * h2.Lambda) / 2.0


def annual_expected_el_fraction(h: CountryHazard) -> float:
    """E[A/4] for a single country-year from the derived point probabilities.

    Used as a deterministic cross-check against the published annual EL percent.
    """
    p = h.p
    return (1.0 * p[1] + 2.0 * p[2] + 3.0 * p[3] + 4.0 * p[4]) / 4.0