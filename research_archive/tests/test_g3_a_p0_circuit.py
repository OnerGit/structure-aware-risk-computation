"""G3-A-P0 — loader exactness, padding, adjoint identity, Grover k-traces,
negative controls, cost-ledger invariants, resource budgets."""

import json
import pathlib

import numpy as np

from adb_g3_a_p0.grover import GroverCircuit
from adb_g3_a_p0 import circuit as C

TOL = 1e-10
SCEN = ("adb-base", "adb-independent", "adb-common-shock")


def _gc(sc):
    return GroverCircuit(sc)


def test_loader_exact_all_scenarios():
    for sc in SCEN:
        gc = _gc(sc)
        pr = gc.loader_prob()
        assert float(np.abs(pr - gc.p).max()) < 1e-9
        assert float(pr[100:128].sum()) < 1e-12
        assert abs(float(pr.sum()) - 1.0) < 1e-12


def test_good_set_identity_and_a():
    for sc in SCEN:
        gc = _gc(sc)
        # a equals the independent frozen target for exceed
        from adb_g2_s2_s1a.laws import frozen_n2_terminal
        from adb_g2_s2_s1a.observables import n2_observable_vectors
        from adb_g2_s2_s1a.laws import scenario_index
        scx = scenario_index()[sc]
        a = float(frozen_n2_terminal(scx) @ n2_observable_vectors(
            scx.eq_cover, scx.rain_cover)["exceed"])
        assert abs(gc.a - a) < 1e-12
        assert len(gc.good) == 45


def test_adjoint_identity():
    probes = [0, 1, 5, 99, 127, (1 << 7) | 5, (1 << 12) | 1, (3 << 7) | 63]
    for sc in SCEN:
        gc = _gc(sc)
        r = gc.adjoint_identity(probes)
        assert r["max_abs_diff"] < 1e-9, sc


def test_grover_k0_3_all_scenarios():
    for sc in SCEN:
        tr = _gc(sc).trace(3, TOL)
        assert tr["all_pass"], (sc, tr["rows"])
        assert all(r["abs_diff"] < 1e-9 for r in tr["rows"])


def test_negative_controls_detected():
    for sc in SCEN:
        gc = _gc(sc)
        nc = gc_negative(gc)
        assert nc["all_detected"], (sc, nc["controls"])


def gc_negative(gc, k=1):
    from adb_g3_a_p0 import run as R
    got = {}
    nc = R.negative_controls(gc, k)
    return nc


def test_wrong_threshold_non_strict_caught():
    gc = _gc("adb-base")
    from adb_g3_a_p0 import run as R
    nc = R.negative_controls(gc, 1)
    assert nc["controls"]["wrong_threshold"]["detected"]


def test_cost_ledger_primitive_only():
    gc = _gc("adb-base")
    costs = gc.component_costs()
    for name, c in costs.items():
        assert c["one_qubit"] > 0 and c["two_qubit_cx"] > 0
        assert c["total"] == c["one_qubit"] + c["two_qubit_cx"]
    # Q = S_f + A_dag + S_0 + A (application order): counts additive
    cA = costs["A"]; cQ = costs["Q"]
    assert cQ["one_qubit"] == (costs["S_f"]["one_qubit"] +
                               costs["A_dagger"]["one_qubit"] +
                               costs["S_0"]["one_qubit"] + cA["one_qubit"])
    assert cQ["two_qubit_cx"] == (costs["S_f"]["two_qubit_cx"] +
                                  costs["A_dagger"]["two_qubit_cx"] +
                                  costs["S_0"]["two_qubit_cx"] + cA["two_qubit_cx"])
    # only declared primitives in the netlist
    for name, net in {"A": gc.netA, "Q": gc.netQ, "S_f": gc.netSf,
                      "S_0": gc.netS0}.items():
        for g in net:
            assert g.name in C.PRIMITIVES


def test_results_evidence_reproducible():
    p = json.load(open("results/g3_a_p0/circuit_evidence.json"))
    assert p["acceptance"]["adb_base_all_pass"] is True
    assert p["wall_within_budget"] is True and p["rss_within_budget"] is True
    assert p["semantics"]["QUANTUM_ADVANTAGE"] == "NOT_CLAIMED"
    assert len(p["scenario_bundles"]) == 3


def test_frozen_inputs_untouched():
    import hashlib
    hi = {
        "src/adb_g1a/model.py": "b470f1df10a2d6334f3d3fb98f4edd4a47571db3f9d05570b6814ce47ed0001c",
        "results/g2_s2_s1a/results.json": "aaa0e7f93e4db08663c66068152c7b15647b405aa7223a56e9c06681aa6513bb",
        "docs/G2_S3_SHARED_SCIENTIFIC_CONTRACT.md": None,
    }
    for p in ("src/adb_g1a/model.py", "results/g2_s2_s1a/results.json"):
        got = hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
        assert got == hi[p], p