"""G3-A-P0 — Grover operator Q = A S0 A† Sf, k-traces, verification and cost.

Everything is executed on the actual expanded primitive netlists with the
small state-vector emulator in circuit.py; ideal references (analytic
sin^2((2k+1)theta)) are used only for comparison.
"""

from __future__ import annotations

import numpy as np

from adb_g1a.model import SCENARIOS
from adb_g2_s2_s1a.laws import frozen_n2_terminal
from adb_g2_s2_s1a.observables import n2_observable_vectors
from adb_g2_s2_s1a.laws import scenario_index

from .circuit import cost, netlist_depth, netlist_two_qubit_depth, run_netlist, basis_vec
from .loader import build_A, build_Adag, tree_angles
from .oracle import DATA, FLAG, SCRATCH, SCRATCH_POOL, sf_net, s0_net

NQ = 7 + 1 + 5   # 7 data + 1 flag + 5 scratch


def good_set_for_threshold(sc, threshold: int = 160, strict: bool = True) -> np.ndarray:
    """Good states (0..99) from the frozen terminal decode and covers."""
    g = n2_observable_vectors(sc.eq_cover, sc.rain_cover)["exceed"]
    if strict:
        return np.where(g > 0.5)[0]
    # wrong non-strict >=160 predicate: recompute from the vector the >= version
    idx = np.arange(100)
    from adb_g2_s2_s1b.mathdef import _by_index, COVER_CONFIGS
    cfg = COVER_CONFIGS["base"]
    a1, a2, r1, r2 = _by_index(idx)
    um1 = cfg["eq_units"] * a1 + cfg["rain_units"] * r1
    um2 = cfg["eq_units"] * a2 + cfg["rain_units"] * r2
    L = um1 + um2
    return np.where(L >= threshold)[0]


class GroverCircuit:
    def __init__(self, scenario_key: str, threshold: int = 160, strict: bool = True,
                 scrambled_good: bool = False, faulty_diffusion: bool = False):
        si = scenario_index()
        sc = si[scenario_key]
        self.scenario = scenario_key
        p = frozen_n2_terminal(sc)
        self.p = np.zeros(128)
        self.p[:100] = p
        good = good_set_for_threshold(sc, threshold, strict)
        if scrambled_good:
            rng = np.random.default_rng(20261107)
            img = rng.permutation(100)
            good = np.sort(img[: len(good)])   # different 45-subset (scrambled)
        self.good = np.asarray([int(i) for i in good])
        self.a = float(self.p[self.good].sum())
        self.angles = tree_angles(self.p, 7)
        self.netA = build_A(self.angles, DATA, SCRATCH)
        self.netAdag = build_Adag(self.netA)
        self.netSf = sf_net(self.good)
        self.netS0 = s0_net()
        if faulty_diffusion:
            self.netQ = self.netSf + self.netA + self.netS0 + self.netA
        else:
            self.netQ = self.netSf + self.netAdag + self.netS0 + self.netA

    # -- emulation --------------------------------------------------------
    def state_after(self, k: int, dtype=np.float64) -> np.ndarray:
        st = basis_vec(NQ, 0)
        st = run_netlist(st, self.netA).astype(complex)
        for _ in range(k):
            st = run_netlist(st, self.netQ)
        return st

    def good_probability(self, k: int) -> float:
        st = self.state_after(k)
        from .circuit import read_amplitude
        acc = 0.0
        for sup in self.good:
            acc += abs(read_amplitude(st, int(sup))) ** 2
        return float(acc)

    def analytic(self, k: int) -> float:
        return float(np.sin((2 * k + 1) * np.arcsin(np.sqrt(self.a))) ** 2)

    def trace(self, kmax: int = 3, tol: float = 1e-10) -> dict:
        rows = []
        for k in range(kmax + 1):
            measured = self.good_probability(k)
            analytic = self.analytic(k)
            rows.append({"k": k, "measured": measured, "analytic": analytic,
                         "abs_diff": abs(measured - analytic),
                         "pass": bool(abs(measured - analytic) <= tol)})
        return {"scenario": self.scenario, "a": self.a, "good_set_size": len(self.good),
                "tol": tol, "rows": rows,
                "all_pass": bool(all(r["pass"] for r in rows))}

    # -- loader probes -----------------------------------------------------
    def loader_prob(self) -> np.ndarray:
        from .circuit import read_amplitude
        st = run_netlist(basis_vec(NQ, 0).astype(complex), self.netA)
        probs = [abs(read_amplitude(st, i)) ** 2 for i in range(128)]
        return np.asarray(probs)

    def adjoint_identity(self, probes: list[int]) -> dict:
        """A† A |x> = |x> on declared probe basis indices (composite index:
        data bits 0..6, flag bit 7, scratch bits 8..12)."""
        from .circuit import read_amplitude
        full = self.netAdag + self.netA
        maxdiff = 0.0
        for sup in probes:
            st = basis_vec(NQ, int(sup)).astype(complex)
            st2 = run_netlist(st, full)
            target = read_amplitude(st2, int(sup))
            maxdiff = max(maxdiff, abs(target - 1.0))
            # orthogonality: total norm must stay 1 and other slots ~0
            nrm = float(np.abs(st2).reshape(-1).sum())
            nrm2 = float(np.abs(st2 * np.conj(st2)).reshape(-1).sum())
            maxdiff = max(maxdiff, abs(nrm2 - 1.0))
        return {"probes": probes, "n_probes_chemistry": len(probes),
                "max_abs_diff": maxdiff}

    # -- costs -------------------------------------------------------------
    def component_costs(self) -> dict:
        comps = {"A": self.netA, "A_dagger": self.netAdag, "S_f": self.netSf,
                 "S_0": self.netS0, "Q": self.netQ}
        out = {}
        for name, net in comps.items():
            c = cost(net)
            out[name] = {
                "one_qubit": c["one_qubit_gates"],
                "two_qubit_cx": c["two_qubit_cx_gates"],
                "total": c["total_primitive_gates"],
                "depth": netlist_depth(net),
                "two_qubit_depth": netlist_two_qubit_depth(net),
            }
        return out

    def full_costs(self, kmax: int = 3) -> dict:
        cc = self.component_costs()
        out = {}
        for k in range(kmax + 1):
            tot1 = cc["A"]["one_qubit"] + k * cc["Q"]["one_qubit"]
            tot2 = cc["A"]["two_qubit_cx"] + k * cc["Q"]["two_qubit_cx"]
            out[f"Q^1_x_A k={k}"] = {"one_qubit": tot1, "two_qubit_cx": tot2,
                                     "total": tot1 + tot2}
        return {"components": cc, "k_traces": out}