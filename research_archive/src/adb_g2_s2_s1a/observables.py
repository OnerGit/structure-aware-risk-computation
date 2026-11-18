"""Normalized observables (n=2 official and n=3 synthetic) in integer USD
500,000 units. A_i/R_i are state variables; no currency is attached to them.
Exact predicates: per-country exhaustion `A_i==4 AND R_i==1`, strict exceedance
`L_units > threshold_units`.
"""

from __future__ import annotations

import numpy as np

UNIT = 500_000.0
# per-country unit maps: A_i state level -> EQ units; R flag -> rain units
EQ_UNITS = {70_000_000: 35, 60_000_000: 30}
RAIN_UNITS = {10_000_000: 20, 20_000_000: 40}


def country_loss_units(A, R, eq_cover, rain_cover) -> np.ndarray:
    return EQ_UNITS[eq_cover] * A + RAIN_UNITS[rain_cover] * R


def _n2_states():
    out = []
    for A1 in range(5):
        for A2 in range(5):
            for R1 in range(2):
                for R2 in range(2):
                    out.append((A1, A2, R1, R2))
    return np.asarray(out, dtype=np.int64)


_N2 = _n2_states()


def n2_observable_vectors(eq_cover: float, rain_cover: float) -> dict[str, np.ndarray]:
    """Five official n=2 normalized observable vectors over 100 terminal states.

    Layout matches the frozen index convention: index = (a1*5+a2)*4 + rain_idx
    with rain_idx order [(0,0),(1,0),(0,1),(1,1)].
    """
    idx = np.arange(100)
    eq = idx // 4
    A1, A2 = divmod(eq, 5)
    ri = idx % 4
    R1 = np.array([0, 1, 0, 1])[ri]
    R2 = np.array([0, 0, 1, 1])[ri]
    um1 = country_loss_units(A1, R1, eq_cover, rain_cover)   # per-country units
    um2 = country_loss_units(A2, R2, eq_cover, rain_cover)
    L_units = um1 + um2
    EL = (L_units * UNIT) / 160_000_000.0
    attach = ((um1 > 0) & (um2 > 0)).astype(float)
    exceed = (L_units > 160).astype(float)                    # L > 80m -> units > 160
    stoploss = np.maximum(L_units - 160, 0) * UNIT / 80_000_000.0
    both = ((A1 == 4) & (R1 == 1) & (A2 == 4) & (R2 == 1)).astype(float)
    return {"EL": EL, "attach": attach, "exceed": exceed,
            "stoploss": stoploss, "both": both}


_N3 = np.array([(c1 // 2, c2 // 2, c3 // 2, c1 % 2, c2 % 2, c3 % 2)
                    for c1 in range(10) for c2 in range(10) for c3 in range(10)],
                   dtype=np.int64)


def n3_synthetic_observable_vectors(eq_cover: float,
                                    rain_cover: float) -> dict[str, np.ndarray]:
    """Five synthetic n=3 normalized observables (SYNTHETIC_N3_NOT_OFFICIAL).

    Layout: index = ((A1*5+A2)*5+A3)*8 + rain_bits, rain_bits=(r1<<2|
    (r2<<1)|r3). Terminal probabilities must be in the same order (per-country
    code order used by laws.terminal_dp_n3 is derived from the same bit layout
    through annual y-codes; see tests for the mapping check).
    """
    A1 = _N3[:, 0]; A2 = _N3[:, 1]; A3 = _N3[:, 2]
    R1 = _N3[:, 3]; R2 = _N3[:, 4]; R3 = _N3[:, 5]
    n = 3
    u = []
    for A, R in ((A1, R1), (A2, R2), (A3, R3)):
        u.append(country_loss_units(A, R, eq_cover, rain_cover))
    L_units = u[0] + u[1] + u[2]
    cap = 80_000_000 * n                 # 240m for n=3
    thresh_units = 40_000_000 * n // UNIT   # 240 units for n=3 (120m/0.5m)
    L = L_units * UNIT
    return {
        "EL_synth": L / cap,
        "attach_synth": ((u[0] > 0) & (u[1] > 0) & (u[2] > 0)).astype(float),
        "exceed_synth": (L_units > thresh_units).astype(float),
        "stoploss_synth": np.maximum(L_units - thresh_units, 0) * UNIT / float(40_000_000 * n),
        "all_exhaust_synth": ((A1 == 4) & (R1 == 1) & (A2 == 4) & (R2 == 1)
                              & (A3 == 4) & (R3 == 1)).astype(float),
    }