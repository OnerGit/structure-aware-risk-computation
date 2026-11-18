"""G3-B-P0 — annual-law tensor factors, independent merge tables, observables.

The production contraction uses ONLY these factors; the frozen terminal
distribution functions of G1-A/S1A are used exclusively as read-only oracle
references (tests) and are never called by the production network path.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from adb_g1a.model import Scenario
from adb_g2_s2_s1a.laws import frozen_n2_annual, annual_law_n3
from adb_g2_s2_s1a.observables import (n2_observable_vectors,
                                       n3_synthetic_observable_vectors)

RAIN_ORDER = [(0, 0), (1, 0), (0, 1), (1, 1)]


# ---------------------------------------------------------------------------
# annual probability factors (read-only frozen annual joint laws)
# ---------------------------------------------------------------------------
def annual_n2(sc: Scenario) -> np.ndarray:
    return np.asarray(frozen_n2_annual(sc), dtype=float)


def annual_n3(sc: Scenario) -> np.ndarray:
    return np.asarray(annual_law_n3(sc), dtype=float)


def annual_n2_corrupted_shared_shock(sc: Scenario) -> np.ndarray:
    """Annual factor with the rain shared-shock component destroyed (cr=0)."""
    sc_bad = replace(sc, rain_common_share=0.0)
    return np.asarray(frozen_n2_annual(sc_bad), dtype=float)


# ---------------------------------------------------------------------------
# independent merge tables (n2: 100 states; n3: 1000 states)
# ---------------------------------------------------------------------------
def _decode_n2(b: int) -> (int, int, int, int):
    eq, ri = divmod(b, 4)
    a1, a2 = divmod(eq, 5)
    r1, r2 = RAIN_ORDER[ri]
    return a1, a2, r1, r2


def _encode_n2(a1, a2, r1, r2) -> int:
    return (a1 * 5 + a2) * 4 + RAIN_ORDER.index((r1, r2))


def merge_map_n2(cap: int = 4) -> np.ndarray:
    """rows[a] = length-100 array of a' = merge(a, y) for each annual y."""
    out = np.zeros((100, 100), dtype=np.int64)
    for a in range(100):
        A1, A2, R1, R2 = _decode_n2(a)
        row = np.zeros(100, dtype=np.int64)
        for y in range(100):
            a1, a2, r1, r2 = _decode_n2(y)
            row[y] = _encode_n2(min(A1 + a1, cap), min(A2 + a2, cap),
                                R1 | r1, R2 | r2)
        out[a] = row
    return out


def _decode_n3(idx: int) -> (int, int, int, int, int, int):
    c1, c2, c3 = idx // 100, (idx // 10) % 10, idx % 10
    A1, r1 = divmod(c1, 2)
    A2, r2 = divmod(c2, 2)
    A3, r3 = divmod(c3, 2)
    return A1, A2, A3, r1, r2, r3


def _encode_n3(A1, A2, A3, r1, r2, r3) -> int:
    c1 = A1 * 2 + r1
    c2 = A2 * 2 + r2
    c3 = A3 * 2 + r3
    return c1 * 100 + c2 * 10 + c3


def merge_map_n3(cap: int = 4) -> np.ndarray:
    out = np.zeros((1000, 1000), dtype=np.int64)
    for acc in range(1000):
        A1, A2, A3, R1, R2, R3 = _decode_n3(acc)
        row = np.zeros(1000, dtype=np.int64)
        for y in range(1000):
            a1, a2, a3, r1, r2, r3 = _decode_n3(y)
            row[y] = _encode_n3(min(A1 + a1, cap), min(A2 + a2, cap),
                                min(A3 + a3, cap), R1 | r1, R2 | r2, R3 | r3)
        out[acc] = row
    return out


def marginalize_n3_to_n2_terminal(term1000: np.ndarray) -> np.ndarray:
    out = np.zeros(100)
    for idx in range(1000):
        A1, A2, _A3, R1, R2, _R3 = _decode_n3(idx)
        out[_encode_n2(A1, A2, R1, R2)] += term1000[idx]
    return out


def marginalize_n3_to_n2_annual(annual1000: np.ndarray) -> np.ndarray:
    out = np.zeros(100)
    for idx in range(1000):
        A1, A2, A3, R1, R2, R3 = _decode_n3(idx)
        out[_encode_n2(A1, A2, R1, R2)] += annual1000[idx]
    return out


# ---------------------------------------------------------------------------
# observable measurement tensors (frozen semantics)
# ---------------------------------------------------------------------------
def observables_n2(sc: Scenario) -> dict[str, np.ndarray]:
    return n2_observable_vectors(sc.eq_cover, sc.rain_cover)


def observables_n2_cover(sc: Scenario, eq_cover: float, rain_cover: float):
    return n2_observable_vectors(eq_cover, rain_cover)


def observables_n3(sc: Scenario) -> dict[str, np.ndarray]:
    return n3_synthetic_observable_vectors(sc.eq_cover, sc.rain_cover)


def exceed_ge_indicator(sc: Scenario) -> np.ndarray:
    """Wrong-threshold observable tensor (L_units >= 160) for negative control."""
    from adb_g2_s2_s1b.mathdef import _by_index, COVER_CONFIGS
    a1, a2, r1, r2 = _by_index(np.arange(100))
    cfg = COVER_CONFIGS["base"] if abs(sc.eq_cover - 70e6) < 1 else COVER_CONFIGS["sublimit"]
    um1 = cfg["eq_units"] * a1 + cfg["rain_units"] * r1
    um2 = cfg["eq_units"] * a2 + cfg["rain_units"] * r2
    return ((um1 + um2) >= 160).astype(float)