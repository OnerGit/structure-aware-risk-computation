"""G3-A-P0-R1 — independent exact reference and phase-sensitive loader checks.

R1-01: independent exact exceed reference from the FROZEN G1-B
``results/g1b_full/exact_ground_truth.json`` (never recomputed via the S1B/S1A
production helper functions), plus an independent decoder of the documented
basis-index convention implemented from first principles in this module
(no imports of ``laws.terminal_code`` / ``observables.n2_observable_vectors``).

R1-02: phase-sensitive loader verification — compare the actual complex
amplitudes of A|0> to sqrt(p) allowing only a global phase, verify relative
sign consistency, verify A-dagger·A = I on non-trivial complex
superpositions (full-vector), and provide a deliberate sign-flip negative
control that the phase checks must reject.
"""

from __future__ import annotations

import json

import numpy as np

from adb_g2_s2_s1a.laws import frozen_n2_terminal
from .circuit import basis_vec, read_amplitude, run_netlist
from .grover import GroverCircuit

# independent decoder constants (explicit, not imported from production)
_BASE_EQ_UNITS = 35
_BASE_RAIN_UNITS = 20
_RAIN_ORDER = [(0, 0), (1, 0), (0, 1), (1, 1)]
_GT = json.load(open("results/g1b_full/exact_ground_truth.json"))


def independent_exceed_indicator():
    """100-dim binary exceed vector from the documented convention only."""
    out = np.zeros(100, dtype=float)
    for b in range(100):
        eq, ri = divmod(b, 4)
        a1, a2 = divmod(eq, 5)
        r1, r2 = _RAIN_ORDER[ri]
        um1 = _BASE_EQ_UNITS * a1 + _BASE_RAIN_UNITS * r1
        um2 = _BASE_EQ_UNITS * a2 + _BASE_RAIN_UNITS * r2
        out[b] = 1.0 if (um1 + um2) > 160 else 0.0
    return out


def independent_good_set():
    return np.where(independent_exceed_indicator() > 0.5)[0]


def frozen_exceed_reference(scenario: str) -> float:
    """Frozen G1-B exact reference (certified scalar), not production."""
    return float(_GT[scenario]["term_portfolio_loss_exceedance_probability"])


def independent_reference(gc: GroverCircuit) -> dict:
    """Cross-check circuit a / good set against the independent reference."""
    good_ind = independent_good_set()
    mass_ind = float(gc.p[good_ind].sum())
    ref = frozen_exceed_reference(gc.scenario)
    return {
        "scenario": gc.scenario,
        "circuit_a": gc.a,
        "circuit_good_set_size": len(gc.good),
        "independent_good_set_size": int(len(good_ind)),
        "good_set_match": bool(set(good_ind.tolist()) == set(int(x) for x in gc.good)),
        "independent_mass_over_production_p": mass_ind,
        "frozen_exact_reference": ref,
        "diff_circuit_a_vs_frozen": abs(gc.a - ref),
        "diff_independent_mass_vs_frozen": abs(mass_ind - ref),
        "pass": bool(set(good_ind.tolist()) == set(int(x) for x in gc.good)
                     and abs(gc.a - ref) <= 1e-12
                     and abs(mass_ind - ref) <= 1e-12),
    }


def loader_amplitudes(gc: GroverCircuit) -> np.ndarray:
    st = run_netlist(basis_vec(13, 0).astype(complex), gc.netA)
    return np.asarray([read_amplitude(st, i) for i in range(128)])


def target_amplitudes(gc: GroverCircuit) -> np.ndarray:
    return np.sqrt(np.clip(gc.p, 0.0, None))[:128]


def phase_check(amp: np.ndarray, w: np.ndarray, tol: float = 1e-9) -> dict:
    """Allows one global real phase; rejects relative sign flips."""
    mag_err = float(np.abs(np.abs(amp) - w).max())
    nz = w > 1e-300
    if not nz.any():
        return {"ok": True, "max_abs_err": mag_err, "relative_sign_ok": True}
    scale = np.where(nz, amp.real / np.where(nz, w, 1.0), np.nan)
    # global-phase arrow(c): all acs share the same sign; compare each vs first
    good = scale[nz][~np.isnan(scale[nz])]
    common = good[0]
    rel_ok = bool(float(np.max(np.abs(good - common))) <= max(tol, tol * abs(common)))
    return {"ok": rel_ok and mag_err <= tol, "max_abs_err": mag_err,
            "relative_sign_ok": rel_ok}


def complex_adjoint_full_vector(gc: GroverCircuit, seed: int = 20261107,
                                tol: float = 1e-9, n_samples: int = 3) -> dict:
    """A†A = I on non-trivial complex superpositions, full-vector compare."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_samples):
        psi = (rng.standard_normal((2,) * 13)
               + 1j * rng.standard_normal((2,) * 13)).astype(complex)
        psi *= 1.0 / np.sqrt(float((np.abs(psi) ** 2).sum()))
        y = run_netlist(psi, gc.netAdag)
        z = run_netlist(y, gc.netA)
        dmax = float(np.abs(z - psi).max())
        nrm = float(np.abs(z * np.conj(z)).sum())
        out.append({"max_abs_diff_per_component": dmax, "norm": nrm,
                    "pass": bool(dmax <= tol)})
    return {"n_samples": n_samples, "samples": out,
            "all_pass": bool(all(s["pass"] for s in out))}


def sign_flip_state(gc: GroverCircuit, mode: str) -> np.ndarray:
    """Deliberately corrupted prepared state with flipped relative signs.

    mode='good_phase': flip the phase of every good-state amplitude
    (relative to the bad-state block); mode='single': flip a single nonzero
    amplitude (index 3 if present else 5).
    """
    from .circuit import idx_coords
    base = run_netlist(basis_vec(13, 0).astype(complex), gc.netA)
    st = base.copy()
    if mode == "good_phase":
        for gi in gc.good:
            st[idx_coords(int(gi), 13)] *= -1
    elif mode == "single":
        idx = 3 if float(gc.p[3]) > 1e-300 else 5
        st[idx_coords(int(idx), 13)] *= -1
    else:
        raise ValueError(mode)
    return st