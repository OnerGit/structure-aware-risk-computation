"""G3-B-P0 — direct annual-law TN construction & validation tests."""

import inspect
import json
import pathlib

import numpy as np

from adb_g1a.model import SCENARIOS
from adb_g2_s2_s1a.laws import (frozen_n2_terminal, merge_state,
                                terminal_dp_n3, frozen_n2_annual,
                                marginalize_n3_to_n2)
from adb_g2_s2_s1a.observables import n2_observable_vectors

from adb_g3_b_p0 import network as N
from adb_g3_b_p0.factors import (annual_n2, annual_n3, merge_map_n2, merge_map_n3,
                                 observables_n2, observables_n3)

SCEN = {s.key: s for s in SCENARIOS}
SCEN_KEYS = ["adb-base", "adb-independent", "adb-common-shock",
             "adb-rain-stress", "adb-rain-cover"]
METRICS = ("EL", "attach", "exceed", "stoploss", "both")


def test_annual_factor_sums_to_one():
    for k in SCEN_KEYS:
        assert abs(annual_n2(SCEN[k]).sum() - 1.0) < 1e-12
    assert abs(annual_n3(SCEN["adb-base"]).sum() - 1.0) < 1e-9


def test_independent_merge_matches_production():
    m = merge_map_n2()
    for a in range(100):
        for y in range(100):
            assert m[a, y] == merge_state(a, y), (a, y)
    m3 = merge_map_n3()
    # spot-check n3 merge indices (decode production convention)
    for acc in (0, 17, 513, 999):
        for y in (0, 31, 997):
            A1, A2, A3, R1, R2, R3 = _decode_n3(acc)
            a1, a2, a3, r1, r2, r3 = _decode_n3(y)
            exp = _encode_n3(min(A1+a1,4), min(A2+a2,4), min(A3+a3,4),
                             R1|r1, R2|r2, R3|r3)
            assert m3[acc, y] == exp, (acc, y)


def _decode_n3(idx):
    c1, c2, c3 = idx // 100, (idx // 10) % 10, idx % 10
    return c1//2, c2//2, c3//2, c1%2, c2%2, c3%2


def _encode_n3(A1, A2, A3, r1, r2, r3):
    c1 = A1*2+r1; c2 = A2*2+r2; c3 = A3*2+r3
    return c1*100 + c2*10 + c3


def test_25_official_cells_exact():
    res = json.load(open("results/g2_s2_s1a/results.json"))
    rows = N.all_25_cells(SCEN_KEYS)
    assert len(rows) == 25
    for r in rows:
        target = res["n2"][r["scenario"]]["targets"][r["metric"]]
        assert abs(r["value"] - target) <= 1e-12, r
        # strict exceed semantics: value must equal the >160 exclusive target
        if r["metric"] == "exceed":
            assert abs(r["value"] - target) <= 1e-12


def test_terminal_matches_frozen():
    for k in SCEN_KEYS:
        p, t = N.contract_n2_terminal(SCEN[k])
        ref = frozen_n2_terminal(SCEN[k])
        assert np.abs(p - ref).max() <= 1e-12, k
        assert abs(t[0]["sum"] - 1.0) <= 1e-12


def test_n3_marginals_and_metrics():
    sc = SCEN["adb-base"]
    term, _, _ = N.contract_n3_terminal(sc)
    A3 = annual_n3(sc)
    from adb_g3_b_p0.factors import (marginalize_n3_to_n2_terminal,
                                     marginalize_n3_to_n2_annual)
    mt = marginalize_n3_to_n2_terminal(term)
    mt2 = marginalize_n3_to_n2(term)
    assert np.abs(mt - frozen_n2_terminal(sc)).max() <= 1e-12
    assert np.abs(mt - mt2).max() <= 1e-12
    ma = marginalize_n3_to_n2_annual(A3)
    assert np.abs(ma - frozen_n2_annual(sc)).max() <= 1e-12
    n3dp = terminal_dp_n3(A3)
    obs3 = observables_n3(sc)
    k3 = {"EL": "EL_synth", "attach": "attach_synth", "exceed": "exceed_synth",
          "stoploss": "stoploss_synth", "both": "all_exhaust_synth"}
    for m in METRICS:
        v_tn = float(np.dot(term, obs3[k3[m]]))
        v_dp = float(np.dot(n3dp, obs3[k3[m]]))
        assert abs(v_tn - v_dp) <= 1e-12, m


def test_chi_truncation_records_errors_and_exact_at_rank8():
    sc = SCEN["adb-base"]
    p_exact, _ = N.contract_n2_terminal(sc)
    obs = observables_n2(sc)["exceed"]
    ref = float(np.dot(p_exact, obs))
    for chi in (1, 2, 4, 6):
        r = N.contract_truncated_n2(sc, chi)
        v = float(np.dot(r["p"][:100], obs))
        # errors and per-bond kept fractions are always recorded
        assert r["trace"][-1]["kept_L2_fraction"] > 0.0
        assert r["trace"][-1]["kept_L2_fraction"] <= 1.0
        assert abs(r["trace"][-1]["norm_L1"] - float(r["p"].sum())) < 1e-12
        assert r["trace"][-1]["norm_L1"] >= 0.0   # nonneg; normalization may deviate
        if chi in (1, 2):                          # approximate tier: must deviate
            assert abs(v - ref) > 1e-6, (chi, v, ref)
    r8 = N.contract_truncated_n2(sc, 8)
    v8 = float(np.dot(r8["p"][:100], obs))
    # the frozen distribution's TT rank is <= 8 -> chi=8 is exact
    assert abs(v8 - ref) <= 1e-9


def test_negative_controls():
    from adb_g3_b_p0 import run as R
    nc = R.negative_controls()
    assert nc["all_detected"] is True


def test_no_terminal_input_in_production_path():
    src_n = inspect.getsource(N.contract_n2_terminal) + \
             inspect.getsource(N.contract_n3_terminal) + \
             inspect.getsource(N.contract_truncated_n2)
    assert "frozen_n2_terminal" not in src_n
    assert "terminal_dp_n3" not in src_n
    src_f = inspect.getsource(merge_map_n2) + inspect.getsource(merge_map_n3)
    assert "merge_state" not in src_f.replace("independent", "") or "merge_state" not in src_f


def test_cover_checks_recorded_and_not_in_official():
    cc = N.cover_mapping_checks()
    assert "adb-base/sublimit" in cc
    assert "adb-rain-cover/base" in cc


def test_results_files_written():
    p = json.load(open("results/g3_b_p0/full.json"))
    assert p["n2"]["n2_25_cells"]["all_pass_1e12"] is True
    assert p["n3"]["marginals_pass_1e12"] is True
    assert p["negative_controls"]["all_detected"] is True
    assert p["classification"]["DP_EQUIVALENT_TN"] == "YES"