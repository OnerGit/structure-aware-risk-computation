"""Frozen-reference laws and the synthetic n=3 law (Checkpoint B).

Frozen G1-A is used only to *construct reference fixtures*. The transfer/merge,
rain mixture and n=3 generator are independently implemented below.
Index conventions (documented): annual/terminal state index y = (a1*5+a2)*4 +
rain_idx with rain_idx order [(0,0),(1,0),(0,1),(1,1)] — matches the frozen
convention.
"""

from __future__ import annotations

import itertools as it

import numpy as np
from scipy.linalg import expm

from adb_g1a.ctmc import (annual_joint_distribution, eq_generator,
                          rain_annual_probabilities, state_arrays)
from adb_g1a.hazard import HAZARDS, common_shock_rate
from adb_g1a.model import SCENARIOS, Scenario, scenario_parameters
from adb_g1a.ctmc import annual_eq_distribution  # noqa: F401

RAIN_ORDER = [(0, 0), (1, 0), (0, 1), (1, 1)]


# ---------------------------------------------------------------------------
# own index helpers and merge (independent implementation)
# ---------------------------------------------------------------------------
def year_code(a1: int, a2: int, r1: int, r2: int) -> int:
    return (a1 * 5 + a2) * 4 + RAIN_ORDER.index((r1, r2))


def terminal_code(a1: int, a2: int, r1: int, r2: int) -> int:
    return year_code(a1, a2, r1, r2)  # same 100-layout


def decode_terminal(idx: int) -> tuple[int, int, int, int]:
    eq, ri = divmod(idx, 4)
    a1, a2 = divmod(eq, 5)
    r1, r2 = RAIN_ORDER[ri]
    return a1, a2, r1, r2


def merge_state(acc: int, year_idx: int) -> int:
    """Capped accumulation: EQ A_i <- min(A_i+a_i,4), rain R_i <- R_i OR r_i.

    Independent implementation (own cap + Boolean OR); indices follow the
    documented convention.
    """
    A1, A2, R1, R2 = decode_terminal(acc)
    a1, a2, r1, r2 = decode_terminal(year_idx)
    return terminal_code(min(A1 + a1, 4), min(A2 + a2, 4), R1 | r1, R2 | r2)


def all_terminal_states() -> np.ndarray:
    arr = np.zeros((100, 4), dtype=np.int64)
    for i in range(100):
        arr[i] = decode_terminal(i)
    return arr


# ---------------------------------------------------------------------------
# frozen n=2 reference fixtures (G1-A only)
# ---------------------------------------------------------------------------
def scenario_index() -> dict[str, Scenario]:
    return {s.key: s for s in SCENARIOS}


def frozen_n2_annual(sc: Scenario) -> np.ndarray:
    h1, h2, c, mu1, mu2, cr = scenario_parameters(sc)
    q = eq_generator(h1.lam, h2.lam, c)
    eq25 = annual_eq_distribution(q)
    rain4 = rain_annual_probabilities(mu1, mu2, cr)
    return np.asarray(annual_joint_distribution(eq25, rain4), dtype=float)


def frozen_n2_terminal(sc: Scenario) -> np.ndarray:
    from adb_g1a.ctmc import three_year_terminal_distribution
    return three_year_terminal_distribution(frozen_n2_annual(sc))


def frozen_n2_eq25(sc: Scenario) -> np.ndarray:
    h1, h2, c, *_ = scenario_parameters(sc)
    return annual_eq_distribution(eq_generator(h1.lam, h2.lam, c))


def frozen_n2_rain4(sc: Scenario) -> np.ndarray:
    *_ , mu1, mu2, cr = scenario_parameters(sc)
    return rain_annual_probabilities(mu1, mu2, cr)


def frozen_n2_targets(sc: Scenario,
                      covers: tuple[float, float] | None = None) -> dict[str, float]:
    """Normalized n=2 observable targets from the frozen exact cells."""
    from adb_g1a.model import compute_scenario
    from .observables import n2_observable_vectors
    sc_ = sc
    exact = compute_scenario(sc_)
    eq_cover, rain_cover = (covers if covers is not None
                            else (sc_.eq_cover, sc_.rain_cover))
    g = n2_observable_vectors(eq_cover, rain_cover)
    p = frozen_n2_terminal(sc_)
    return {k: float(np.sum(p * g[k])) for k in g}


# ---------------------------------------------------------------------------
# synthetic n=3 law (Checkpoint B)
# ---------------------------------------------------------------------------
def third_country_hazard() -> tuple[float, tuple[float, float, float, float], float]:
    """Synthetic third country: declared copy of the Kyrgyz first-country hazard
    (severity rates + total intensity + base rain rate). Not an official ADB
    geography/observation."""
    h = HAZARDS["Kyrgyz"]
    mu = -np.log(0.98)
    return h.Lambda, h.lam, float(mu)


def eq_generator_n3(lam1, lam2, lam3, c: float) -> np.ndarray:
    """125-state capped 3-country earthquake generator (independent marks)."""
    L = [float(sum(l)) for l in (lam1, lam2, lam3)]
    w = [np.asarray(l, dtype=float) / L[i] for i, l in enumerate((lam1, lam2, lam3))]
    Q = np.zeros((125, 125))
    for a1 in range(5):
        for a2 in range(5):
            for a3 in range(5):
                s = a1 * 25 + a2 * 5 + a3
                for i, (a, wi, Li) in enumerate(zip((a1, a2, a3), w, L)):
                    if a < 4:
                        for k in range(4):
                            ap = min(a + k + 1, 4)
                            nv = [a1, a2, a3]
                            nv[i] = ap
                            Q[s, nv[0] * 25 + nv[1] * 5 + nv[2]] += (Li - c) * wi[k]
                for k1 in range(4):
                    for k2 in range(4):
                        for k3 in range(4):
                            ap = [min(a1 + k1 + 1, 4), min(a2 + k2 + 1, 4),
                                  min(a3 + k3 + 1, 4)]
                            if tuple(ap) == (a1, a2, a3):
                                continue
                            Q[s, ap[0] * 25 + ap[1] * 5 + ap[2]] += c * w[0][k1] * w[1][k2] * w[2][k3]
    for s in range(125):
        Q[s, s] = -Q[s].sum()
    return Q


def rain_law_n3(mu: tuple[float, float, float], cr: float) -> np.ndarray:
    """2^3=8 rain-flags law (shared-Poisson mixture). Analytic, exact."""
    p_nosh = np.exp(-float(cr))
    uniq = [1.0 - np.exp(-(float(mu[i]) - float(cr))) for i in range(3)]
    out = np.zeros(8)
    for bits in it.product((0, 1), repeat=3):
        idx = int(bits[0]) * 4 + int(bits[1]) * 2 + int(bits[2])
        if bits == (1, 1, 1):
            out[idx] = (1.0 - p_nosh) + p_nosh * np.prod(uniq)
        else:
            out[idx] = p_nosh * float(np.prod(
                [uniq[i] if bits[i] == 1 else 1.0 - uniq[i] for i in range(3)]))
    return out


def annual_law_n3(sc: Scenario) -> np.ndarray:
    """1000-state annual law in country-code order c1*100+c2*10+c3, where
    c_i = a_i*2 + r_i (a_i EQ level, r_i rain flag). Years independent."""
    h1 = HAZARDS["Kyrgyz"]
    L1, lam1 = h1.Lambda, h1.lam
    mu1 = float(-np.log(0.98))
    h2 = HAZARDS["Tajik"]
    L2, lam2 = h2.Lambda, h2.lam
    mu2 = float(-np.log(0.98))
    L3, lam3, mu3 = third_country_hazard()
    c = common_shock_rate(h1, h2, sc.m)
    for name, v in (("L1-c", L1 - c), ("L2-c", L2 - c), ("L3-c", L3 - c),
                    ("c", c)):
        if v < -1e-14:
            raise ValueError(f"negative intensity {name}={v}")
    Q3 = eq_generator_n3(lam1, lam2, lam3, c)
    eq125 = np.asarray(expm(Q3)[0], dtype=float).reshape(5, 5, 5)
    share = sc.rain_common_share
    mcr = float(share * min(mu1, mu2, mu3))
    for name, v in (("mu1-cr", mu1 - mcr), ("mu2-cr", mu2 - mcr),
                    ("mu3-cr", mu3 - mcr), ("cr", mcr)):
        if v < -1e-14:
            raise ValueError(f"negative intensity {name}={v}")
    rain8 = rain_law_n3((mu1, mu2, mu3), mcr)
    out = np.zeros(1000)
    for c1 in range(10):
        a1, r1 = divmod(c1, 2)
        for c2 in range(10):
            a2, r2 = divmod(c2, 2)
            for c3 in range(10):
                a3, r3 = divmod(c3, 2)
                eq = eq125[a1, a2, a3] * rain8[r1 * 4 + r2 * 2 + r3]
                out[c1 * 100 + c2 * 10 + c3] = eq
    return out


def terminal_dp_n3(annual_cc: np.ndarray, years: int = 3) -> np.ndarray:
    """Separate exact capped-DP reference over 10^3 terminal states (country-code
    order). O(1000 x 1000) merges per year — no 1000^3 path tensor."""
    n = 1000
    merge = np.zeros((n, n), dtype=np.int64)
    for acc in range(n):
        c1, c2, c3 = acc // 100, (acc // 10) % 10, acc % 10
        AE1, AE2, AE3 = c1 // 2, c2 // 2, c3 // 2
        R1, R2, R3 = c1 % 2, c2 % 2, c3 % 2
        for y in range(n):
            d1, d2, d3 = y // 100, (y // 10) % 10, y % 10
            a1, r1 = divmod(d1, 2); a2, r2 = divmod(d2, 2); a3, r3 = divmod(d3, 2)
            n1 = min(AE1 + a1, 4) * 2 + (R1 | r1)
            n2 = min(AE2 + a2, 4) * 2 + (R2 | r2)
            n3v = min(AE3 + a3, 4) * 2 + (R3 | r3)
            merge[acc, y] = n1 * 100 + n2 * 10 + n3v
    pi = np.zeros(n); pi[0] = 1.0
    for _ in range(years):
        nxt = np.zeros(n)
        for acc in range(n):
            nxt += pi[acc] * np.bincount(merge[acc], weights=annual_cc, minlength=n)
        pi = nxt
    return pi


def marginalize_n3_to_n2(terminal1000: np.ndarray) -> np.ndarray:
    """Sum over country-3 (A3,R3): 1000 -> 100 (A1,A2,R1,R2)."""
    out = np.zeros(100)
    for acc in range(1000):
        c1, c2, _ = acc // 100, (acc // 10) % 10, acc % 10
        a1, r1 = c1 // 2, c1 % 2
        a2, r2 = c2 // 2, c2 % 2
        out[terminal_code(a1, a2, r1, r2)] += terminal1000[acc]
    return out
