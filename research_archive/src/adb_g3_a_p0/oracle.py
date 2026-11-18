"""G3-A-P0 — reversible good-flag oracle S_f and zero reflection S_0.

S_f: for every good 7-bit pattern g (strict L_units>160), a reversible
MCNOT(g -> flag) is applied via an X-corrected 7-control MCX ladder (5 clean
scratch qubits); then Z(flag) yields the phase (-1)^{f(b)}; the flag encoder
is then uncomputed exactly (all scratch and the flag return to |0>).
"""

from __future__ import annotations

import numpy as np

from .circuit import QGate, mcx, x
from .circuit import s0_reflection  # noqa: F401  (re-export)
from .loader import build_Adag  # noqa: F401

DATA = list(range(7))
FLAG = 7
SCRATCH = list(range(8, 13))          # q8..q12
SCRATCH_POOL = [FLAG] + SCRATCH       # flag reused as scratch in loader/S0


def _mcnt_pattern(data: list[int], pattern: int, target: int,
                  anc: list[int]) -> list[QGate]:
    n = len(data)
    g: list[QGate] = []
    for j, cq in enumerate(data):
        if not (pattern >> j) & 1:
            g.append(x(cq))
    g += mcx(list(data), target, anc)
    for j, cq in enumerate(data):
        if not (pattern >> j) & 1:
            g.append(x(cq))
    return g


def good_patterns(good_indices: np.ndarray, nbits: int = 7) -> list[int]:
    return [int(i) for i in np.sort(good_indices)]


def sf_encode(good: np.ndarray) -> list[QGate]:
    """Encode the strict-exceedance flag onto q7 for each good pattern."""
    net: list[QGate] = []
    for pat in good_patterns(good):
        net += _mcnt_pattern(DATA, pat, FLAG, SCRATCH)
    return net


def sf_net(good: np.ndarray) -> list[QGate]:
    """Phase oracle S_f = encode(f) . Z(flag) . encode(f)**-1 (exact uncompute)."""
    enc = sf_encode(good)
    net = enc + [QGate("RZ", (FLAG,), np.pi)]      # Z(flag)
    net += list(reversed(enc))
    return net


def s0_net() -> list[QGate]:
    """Zero reflection on the 7 data qubits (flag + 5 scratch as ancilla pool)."""
    return s0_reflection(DATA, SCRATCH_POOL)