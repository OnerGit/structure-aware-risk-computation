"""G3-A-P0 — primitive gate IR, macro decomposition, state-vector emulation,
and netlist cost/depth accounting.

All multi-control/multi-qubit operations are expanded into the declared
primitive alphabet {H, X, RY(theta), RZ(theta), T, Tdg, CX} before they are
counted or emulated. No opaque multi-qubit gates enter the reported netlists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

PRIMITIVES = ("H", "X", "RY", "RZ", "T", "Tdg", "CX")


@dataclass
class QGate:
    name: str
    qargs: Tuple[int, ...]
    theta: float | None = None

    def __post_init__(self):
        assert self.name in PRIMITIVES, self.name

    def key(self):
        return (self.name, self.qargs, None if self.theta is None else round(self.theta, 14))


# ---------------------------------------------------------------------------
# Small unitary helpers (for numeric verification only)
# ---------------------------------------------------------------------------
def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def _h():
    return (np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2))


def _x():
    return np.array([[0, 1], [1, 0]], dtype=complex)


def _cx():
    m = np.eye(4, dtype=complex)
    m[2, :] = (0, 0, 0, 1)
    m[3, :] = (0, 0, 1, 0)
    return m


# ---------------------------------------------------------------------------
# Macro decompositions (return lists of QGate primitives)
# ---------------------------------------------------------------------------
def r_gate(kind, q, theta):
    return QGate(kind, (q,), float(theta))


def ry(t, q):
    return r_gate("RY", q, t)


def rz(t, q):
    return r_gate("RZ", q, t)


def h(q):
    return QGate("H", (q,))


def x(q):
    return QGate("X", (q,))


def cx(c, t):
    return QGate("CX", (c, t))


def t_gate(q):
    return QGate("T", (q,))


def tdg(q):
    return QGate("Tdg", (q,))


def toffoli(a, b, c) -> List[QGate]:
    """15-gate Clifford+T Toffoli (Nielsen & Chuang form); unit-tested."""
    return [
        h(c), cx(b, c), tdg(c), cx(a, c), t_gate(c), cx(b, c), tdg(c),
        cx(a, c), t_gate(b), t_gate(c), h(c), cx(a, b), tdg(b), t_gate(a),
        cx(a, b),
    ]


def cry(theta, c, t) -> List[QGate]:
    """Controlled-RY: RY(theta/2)_t, CX(c,t), RY(-theta/2)_t, CX(c,t)."""
    th = float(theta)
    return [ry(th / 2, t), cx(c, t), ry(-th / 2, t), cx(c, t)]


def mcx(controls: List[int], target: int, anc: List[int]) -> List[QGate]:
    """Multi-controlled X with n controls.

    n==0: X(target); n==1: CX; n==2: Toffoli; n>=3: Barenco ladder on n-2
    clean ancillas (all returned to |0>). Uses only declared primitives.
    """
    n = len(controls)
    if n == 0:
        return [x(target)]
    if n == 1:
        return [cx(controls[0], target)]
    if n == 2:
        return toffoli(controls[0], controls[1], target)
    assert len(anc) >= n - 2, f"need {n-2} ancillas, have {len(anc)}"
    c = list(controls)
    v = anc[: n - 2]
    g: List[QGate] = []
    g += toffoli(c[0], c[1], v[0])
    for i in range(2, n - 1):
        g += toffoli(v[i - 2], c[i], v[i - 1])
    g += toffoli(v[n - 3], c[n - 1], target)
    for i in range(n - 2, 1, -1):
        g += toffoli(v[i - 2], c[i], v[i - 1])
    g += toffoli(c[0], c[1], v[0])
    return g


def mcry_pattern(theta, controls: List[int], target: int, anc: List[int],
                 pattern: int) -> List[QGate]:
    """RY(theta) on target only when controls match the bit-pattern `pattern`.

    Decomposition: X-corrections for zero-pattern controls, MCX ladder to one
    scratch, CRY(theta) scratch->target, uncompute ladder and X-corrections.
    Uses len(controls) controls + 1 scratch target + ladder ancillas.
    """
    n = len(controls)
    if n == 0:
        return [ry(theta, target)]
    g: List[QGate] = []
    for j, cq in enumerate(controls):
        if not (pattern >> j) & 1:
            g += [x(cq)]
    s = target  # placeholder; actual scratch allocated by caller below
    # we need a scratch s in `anc` plus (n-2) ladder ancillas for n>=3
    if n == 1:
        g += cry(theta, controls[0], target)
    elif n == 2:
        # control pattern all-ones after correction -> Toffoli ladder? n=2:
        # use mcr with a scratch + CRY
        used = list(anc[:1])
        g += mcx(controls, used[0], anc[1:])  # scratch unused ladder for n=2
        g += cry(theta, used[0], target)
        g += mcx(controls, used[0], anc[1:])
    else:
        used = list(anc[: n - 1])  # s + ladder
        s = used[0]
        g += mcx(controls, s, used[1:])
        g += cry(theta, s, target)
        g += mcx(controls, s, used[1:])
    for j, cq in enumerate(controls):
        if not (pattern >> j) & 1:
            g += [x(cq)]
    _ = s
    return g


def mcry(angles, controls: List[int], target: int, anc: List[int]) -> List[QGate]:
    """Uniformly-controlled RY over all 2^len(controls) control patterns."""
    n = len(controls)
    g: List[QGate] = []
    for pattern in range(1 << n):
        th = angles[pattern]
        if abs(th) > 1e-15:
            g += mcry_pattern(th, controls, target, anc, pattern)
    return g


def mcz(controls: List[int], target: int, anc: List[int]) -> List[QGate]:
    """Multi-controlled Z: H-target, MCX, H-target."""
    return [h(target)] + mcx(controls, target, anc) + [h(target)]


def s0_reflection(data: List[int], scratch_pool: List[int]) -> List[QGate]:
    """Zero-state reflection 2|0..0><0..0| - I on `data` (m qubits).

    Implemented as X^7, clean multi-controlled-Z: X(t), H(t), MCX(data,t),
    H(t), X(t) (target t and all scratch are restored; the all-ones branch
    picks up +1 relative phase from the |-> path), then X^7. The MCX ladder
    needs m-2 ancillas (m=7 -> 5 scratch); scratch_pool provides target+ladder.
    """
    m = len(data)
    xs = [x(q) for q in data]
    t = scratch_pool[0]
    ladder = scratch_pool[1:]
    return (xs + [x(t), h(t)] + mcx(data, t, ladder) + [h(t), x(t)] + xs)


# ---------------------------------------------------------------------------
# State-vector emulation of a primitive netlist
# ---------------------------------------------------------------------------
def apply_gate(st, g: QGate):
    """st: complex array of shape (2,)*nq. Mutates and returns st."""
    nq = len(st.shape)
    if g.name == "RY":
        q = g.qargs[0]
        moved = np.moveaxis(st, q, -1)
        flat = moved.reshape(-1, 2)
        nf = flat @ _ry(g.theta).T
        return np.moveaxis(nf.reshape(moved.shape).copy(), -1, q)
    if g.name == "RZ":
        q = g.qargs[0]
        moved = np.moveaxis(st, q, -1)
        flat = moved.reshape(-1, 2)
        nf = flat * np.array([np.exp(-0.5j * g.theta), np.exp(0.5j * g.theta)])
        return np.moveaxis(nf.reshape(moved.shape).copy(), -1, q)
    if g.name in ("T", "Tdg"):
        ph = np.exp(1j * np.pi / 4) if g.name == "T" else np.exp(-1j * np.pi / 4)
        q = g.qargs[0]
        moved = np.moveaxis(st, q, -1)
        flat = moved.reshape(-1, 2)
        nf = flat.copy(); nf[:, 1] *= ph
        return np.moveaxis(nf.reshape(moved.shape), -1, q)
    if g.name == "H":
        q = g.qargs[0]
        moved = np.moveaxis(st, q, -1)
        flat = moved.reshape(-1, 2)
        nf = flat @ _h().T
        return np.moveaxis(nf.reshape(moved.shape).copy(), -1, q)
    if g.name == "X":
        q = g.qargs[0]
        moved = np.moveaxis(st, q, -1)
        nf = moved[..., ::-1].copy()
        return np.moveaxis(nf, -1, q)
    if g.name == "CX":
        c, t = g.qargs
        moved = np.moveaxis(st, (c, t), (-2, -1))
        flat = moved.reshape(-1, 2, 2)
        out = flat.copy()
        out[:, 1, :] = out[:, 1, ::-1]   # control (axis -2) = 1 -> flip target (axis -1)
        nf = out.reshape(moved.shape).copy()
        return np.moveaxis(nf, (-2, -1), (c, t))
    raise NotImplementedError(g.name)


def run_netlist(st, net: List[QGate]) -> np.ndarray:
    for g in net:
        st = apply_gate(st, g)
    return st


def idx_coords(idx: int, nq: int) -> Tuple[int, ...]:
    """Coordinate tuple of an nq-qubit basis index b = sum bit_i 2^i, where
    bit_i is the state of qubit q_i (axis i)."""
    return tuple((idx >> i) & 1 for i in range(nq))


def basis_vec(nq, idx) -> np.ndarray:
    st = np.zeros((2,) * nq, dtype=complex)
    st[idx_coords(idx, nq)] = 1.0
    return st


def read_amplitude(st, idx: int) -> complex:
    """Read the amplitude of basis index b from a state tensor."""
    return st[idx_coords(idx, len(st.shape))]


# ---------------------------------------------------------------------------
# Cost / depth accounting on an actual netlist
# ---------------------------------------------------------------------------
ONE_Q = {"H", "X", "RY", "RZ", "T", "Tdg"}


def cost(net: List[QGate]) -> dict:
    """Gate counts computed from the actual expanded primitive netlist.

    All macros (Toffoli/CRY/MCX/MCRY/S0) are fully decomposed before counting,
    so the netlist contains only the declared primitives; `two_qubit_cx_gates`
    is the true 2q count and `cx_equivalent` is defined equal to it (the
    decomposition already reduces every multi-qubit gate to CX).
    """
    counts = {k: 0 for k in PRIMITIVES}
    for g in net:
        counts[g.name] += 1
    one_q = sum(counts[k] for k in ONE_Q)
    two_q = counts["CX"]
    total = sum(counts.values())
    return {
        "one_qubit_gates": one_q,
        "two_qubit_cx_gates": two_q,
        "macro_toffolis_expanded": 0,
        "total_primitive_gates": total,
        "cx_equivalent": two_q,
        "counts_by_kind": counts,
    }


def netlist_depth(net: List[QGate]) -> int:
    """Critical-path depth: each primitive gate takes one time unit; gates on
    disjoint qubits share time. (1q and 2q gates both cost 1 time unit.)"""
    last = {}
    depth = 0
    for g in net:
        t = 1 + max(last.get(q, 0) for q in g.qargs)
        for q in g.qargs:
            last[q] = t
        depth = max(depth, t)
    return depth


def netlist_two_qubit_depth(net: List[QGate]) -> int:
    """Depth counting only CX gates as consuming a time unit."""
    last = {}
    d = 0
    for g in net:
        if g.name == "CX":
            t = 1 + max(last.get(q, 0) for q in g.qargs)
            for q in g.qargs:
                last[q] = t
            d = max(d, t)
    return d


def netlist_text(net: List[QGate], maxlen: int | None = None) -> str:
    lines = []
    for i, g in enumerate(net):
        if maxlen and i >= maxlen:
            lines.append(f"... ({len(net)-maxlen} more gates omitted)")
            break
        th = "" if g.theta is None else f" {g.theta:.6f}"
        lines.append(f"{i:5d} {g.name:>3}{th}  q{tuple(sorted(g.qargs))}")
    return "\n".join(lines)