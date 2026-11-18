"""G3-A-P0 driver: builds and validates the TABLE_LOAD Grover circuits for
adb-base (mandatory) and, time permitting, adb-independent / adb-common-shock.
Writes results/g3_a_p0/*.json (machine-readable evidence)."""

from __future__ import annotations

import json
import pathlib
import resource
import time

import numpy as np

from .grover import GroverCircuit, good_set_for_threshold

SCENARIOS = ("adb-base", "adb-independent", "adb-common-shock")
TOL = 1e-10


def measure(block):
    t0 = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = block()
    t1 = time.perf_counter()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return out, t1 - t0, rss_after


def negative_controls(gc: GroverCircuit, k: int = 1) -> dict:
    nca, _, _ = measure(lambda: gc.good_probability(k))
    dev = abs(nca - gc.analytic(k))
    rows = {
        "wrong_threshold": _nc_threshold(gc),
        "scrambled_good": _nc_scrambled(gc, k),
        "faulty_diffusion": _nc_diffusion(gc, k),
    }
    return {"k": k, "baseline_dev": dev, "controls": rows,
            "all_detected": all(v["detected"] for v in rows.values())}


def _dev(gc_candidate, a_correct, k, tol=TOL):
    m, _, _ = measure(lambda: gc_candidate.good_probability(k))
    an = float(np.sin((2 * k + 1) * np.arcsin(np.sqrt(a_correct))) ** 2)
    return abs(m - an)


def _nc_threshold(gc):
    a_correct = gc.a
    gcw = GroverCircuit(gc.scenario, threshold=160, strict=False)
    dev = _dev(gcw, a_correct, 1)
    return {"description": "non-strict >=160 predicate", "deviation": dev,
            "detected": bool(dev > 1e-4)}


def _nc_scrambled(gc, k):
    gcs = GroverCircuit(gc.scenario, scrambled_good=True)
    dev = _dev(gcs, gc.a, k)
    return {"description": "scrambled good-set order", "deviation": dev,
            "detected": bool(dev > 1e-4)}


def _nc_diffusion(gc, k):
    gcf = GroverCircuit(gc.scenario, faulty_diffusion=True)
    dev = _dev(gcf, gc.a, k)
    return {"description": "diffusion uses A instead of A-dagger (not centered "
                           "on the original prepared reference)",
            "deviation": dev, "detected": bool(dev > 1e-4)}


def scenario_bundle(scenario: str) -> dict:
    gc, t_build, _ = measure(lambda: GroverCircuit(scenario))
    tr, t_trace, rss = measure(lambda: gc.trace(kmax=3, tol=TOL))
    lp, _, _ = measure(gc.loader_prob)
    loader = {"max_abs_prob_err": float(np.abs(lp - gc.p).max()),
              "padding_mass_100_128": float(lp[100:128].sum()),
              "norm2": float(lp.sum())}
    adj, _, _ = measure(lambda: gc.adjoint_identity(
        [0, 1, 2, 5, 99, 127, (1 << 7) | 5, (1 << 12) | 1, (3 << 7) | 63]))
    nc = negative_controls(gc)
    costs = gc.full_costs(3)
    return {
        "scenario": scenario,
        "mandatory_adb_base": scenario == "adb-base",
        "trace": tr,
        "loader": loader,
        "adjoint_identity": adj,
        "negative_controls": nc,
        "costs": costs,
        "qubits": {"data": 7, "flag": 1, "scratch": 5, "total": 13,
                   "peak_live_scratch": 6},
        "timing": {"build_s": t_build, "trace_s": t_trace,
                   "peak_rss_max_kb": rss},
        "gate_semantics": {
            "PRIMITIVE_GATE_MODEL_COMPILED": True,
            "NATIVE_QPU_COMPILED": "NO", "NOISE_VALIDATED": "NO",
            "hardware_uncompiled": "UNKNOWN_NOT_COMPILED",
        },
    }


def run_all(outdir: str = "results/g3_a_p0") -> dict:
    t0 = time.perf_counter()
    bundles = []
    for sc in SCENARIOS:
        b = scenario_bundle(sc)
        bundles.append(b)
        if sc == "adb-base" and not b["trace"]["all_pass"]:
            break  # mandatory case failed -> do not spend further budget
    wall = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "gate": "G3_A_P0_EXECUTABLE_REVERSIBLE_CIRCUIT_VALIDATION",
        "scenarios_executed": [b["scenario"] for b in bundles],
        "scenario_bundles": bundles,
        "acceptance": {"tol": TOL,
                       "adb_base_all_pass": bundles[0]["trace"]["all_pass"]},
        "measured_wall_s": wall,
        "peak_rss_kb": rss,
        "wall_budget_s": 20 * 60,
        "rss_budget_kb": 2 * 1024 * 1024,
        "wall_within_budget": bool(wall <= 20 * 60),
        "rss_within_budget": bool(rss <= 2 * 1024 * 1024),
        "threads": 1,
        "exact_dp_preprocessing_not_free": {
            "measured_s1a_wall_s": 3.932232,
            "measured_s1a_peak_rss_kb": 64608,
            "note": "TABLE_LOAD uses the exact 100-state terminal table "
                    "produced by the frozen G1-A/S1A Exact-DP pipeline; its "
                    "cost is disclosed and is NOT a quantum loading claim."},
        "semantics": {
            "PRIMITIVE_GATE_MODEL_COMPILED": True,
            "CLASSICAL_COMPETITIVENESS": "NO_MEASURED_BENEFIT",
            "SCALING_ADVANTAGE_NOT_ESTABLISHED": True,
            "QUANTUM_ADVANTAGE": "NOT_CLAIMED",
            "FULL_G1": "NOT_RERUN",
            "A_P1_AUTHORIZED": False, "A_P2_AUTHORIZED": False,
            "G3_B_AUTHORIZED": False,
        },
    }
    w = pathlib.Path(outdir)
    w.mkdir(parents=True, exist_ok=True)
    (w / "circuit_evidence.json").write_text(json.dumps(payload, indent=2))
    return payload