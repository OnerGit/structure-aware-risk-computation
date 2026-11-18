"""G3-A-P0 — exact TABLE_LOAD amplitude loader A and its true inverse A-dagger.

Given the frozen squared-amplitude vector p (padded 100->128, sum 1), the
loader prepares |psi> = sum_x sqrt(p_x)|x> on `n` data qubits with the exact
uniformly-controlled-RY angle tree (levels 0..n-1, 127 angles for n=7). Every
multi-qubit gate is decomposed to the declared primitives (see circuit.py);
A-dagger is the exact reversed adjoint gate list (RY(theta)->RY(-theta)).
"""

from __future__ import annotations

import numpy as np

from .circuit import QGate, mcry, ry


def tree_angles(p: np.ndarray, n: int) -> list[np.ndarray]:
    """Level-ordered UCRY angles for the exact loader.

    Returns angles[l] for l=0..n-1 over 2^l control patterns j; angle =
    2*arcsin(sqrt(P(prefix(j), bit_l=1)/P(prefix(j)))). Bit i is the state of
    qubit q_i (i.e. bit i of the basis address). The bit-l=1 branch mass for a
    fixed prefix j (low l bits) is summed over the scattered addresses
    j | (1<<l) | (h << (l+1)), h in 0..2^{n-l-1}-1 (NOT a contiguous half)."""
    angles = []
    for l in range(n):
        npat = 1 << l
        hi_n = n - l - 1
        row = np.zeros(npat)
        for j in range(npat):
            P0 = 0.0
            P1 = 0.0
            for h in range(1 << hi_n):
                A = j | (h << (l + 1))
                P0 += float(p[A])
                P1 += float(p[A | (1 << l)])
            P = P0 + P1
            if P <= 0.0:
                row[j] = 0.0
            else:
                r = min(max(P1 / P, 0.0), 1.0)
                row[j] = 2.0 * np.arcsin(np.sqrt(r))
        angles.append(row)
    return angles


def build_A(angles: list[np.ndarray], data: list[int], anc: list[int]) -> list[QGate]:
    """Loader A: UCRY levels l=0..n-1 (target q_l, controls q0..q_{l-1})."""
    net: list[QGate] = []
    n = len(data)
    for l, ang in enumerate(angles):
        controls = data[:l]
        net += mcry(ang, controls, data[l], anc)
    return net


def build_Adag(A: list[QGate]) -> list[QGate]:
    """Exact reversed adjoint of A (RY(theta)->RY(-theta)); others self-adjoint."""
    out = []
    for g in reversed(A):
        if g.name == "RY":
            out.append(ry(-g.theta, g.qargs[0]))
        else:
            out.append(g)
    return out