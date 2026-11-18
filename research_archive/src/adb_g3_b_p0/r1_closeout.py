"""G3-B-P0-R1 — synthetic-fixture & diagnostic closeout (new R1-namespaced
results; original B-P0 results are left untouched)."""

from __future__ import annotations

import json
import pathlib
import resource
import time

import numpy as np

from adb_g1a.model import SCENARIOS
from adb_g2_s2_s1a.laws import terminal_dp_n3, frozen_n2_terminal, frozen_n2_annual

from .factors import (annual_n3, observables_n3, merge_map_n3,
                      marginalize_n3_to_n2_terminal,
                      marginalize_n3_to_n2_annual)
from . import network as N

SCEN = {s.key: s for s in SCENARIOS}
METRICS = ("EL", "attach", "exceed", "stoploss", "both")
N3_KEY = {"EL": "EL_synth", "attach": "attach_synth",
          "exceed": "exceed_synth", "stoploss": "stoploss_synth",
          "both": "all_exhaust_synth"}

FIXTURES = [
    {"fixture_id": "F1_base", "scenario": "adb-base",
     "source": "S1A F1/F2/F3 synthetic n3 law (annual_law_n3) + exact n3 DP "
               "(terminal_dp_n3); SYNTHETIC_N3_NOT_OFFICIAL"},
    {"fixture_id": "F2_independent", "scenario": "adb-independent",
     "source": "S1A F1/F2/F3 synthetic n3 law (annual_law_n3) + exact n3 DP "
               "(terminal_dp_n3); SYNTHETIC_N3_NOT_OFFICIAL"},
    {"fixture_id": "F3_common_shock", "scenario": "adb-common-shock",
     "source": "S1A F1/F2/F3 synthetic n3 law (annual_law_n3) + exact n3 DP "
               "(terminal_dp_n3); SYNTHETIC_N3_NOT_OFFICIAL"},
]


def n3_fixtures() -> list[dict]:
    rows = []
    for fx in FIXTURES:
        sc = SCEN[fx["scenario"]]
        A3 = annual_n3(sc)
        term, A3_alt, _ = N.contract_n3_terminal(sc)
        n3dp = terminal_dp_n3(A3, years=3)
        obs3 = observables_n3(sc)
        for m in METRICS:
            v_tn = float(np.dot(term, obs3[N3_KEY[m]]))
            v_dp = float(np.dot(n3dp, obs3[N3_KEY[m]]))
            rows.append({
                "fixture_id": fx["fixture_id"],
                "scenario": fx["scenario"],
                "metric": m,
                "tag": "SYNTHETIC_N3_NOT_OFFICIAL",
                "source": fx["source"],
                "tn_value": v_tn,
                "n3_exact_dp_reference": v_dp,
                "normalized_abs_error": abs(v_tn - v_dp),
                "pass_1e12": bool(abs(v_tn - v_dp) <= 1e-12),
            })
        marg_rows = {}
        marg_rows["annual_to_n2"] = float(np.abs(
            marginalize_n3_to_n2_annual(A3) - frozen_n2_annual(sc)).max())
        marg_rows["terminal_to_n2"] = float(np.abs(
            marginalize_n3_to_n2_terminal(term) - frozen_n2_terminal(sc)).max())
        rows.append({
            "fixture_id": fx["fixture_id"], "scenario": fx["scenario"],
            "tag": "SYNTHETIC_N3_NOT_OFFICIAL",
            "source": fx["source"],
            "marginals": marg_rows,
            "marginals_pass_1e12": bool(marg_rows["annual_to_n2"] <= 1e-12
                                        and marg_rows["terminal_to_n2"] <= 1e-12),
        })
    return rows


def chi_diagnostics(chi_list=(1, 2, 4, 6, 8, 10)) -> dict:
    sc = SCEN["adb-base"]
    p_exact, _ = N.contract_n2_terminal(sc)
    obs = observables_n3 if False else __import__(
        "adb_g2_s2_s1a.observables", fromlist=["n2_observable_vectors"]
    ).n2_observable_vectors(sc.eq_cover, sc.rain_cover)
    FIVE = {m: float(np.dot(p_exact, obs[m])) for m in METRICS}
    exact_l1 = float(p_exact.sum())
    exact_l2 = float(np.dot(p_exact, p_exact))
    runs = []
    for chi in chi_list:
        t0 = time.perf_counter()
        r = N.contract_truncated_n2_diagnostic(sc, chi, years=3)
        dt = time.perf_counter() - t0
        p_tr = r["p"][:100]
        vals = {m: float(np.dot(p_tr, obs[m])) for m in METRICS}
        runs.append({
            "chi": chi,
            "seconds_total": dt,
            "bonds_last_year": r["trace"][2]["bonds"],
            "actual_kept_l2_last_year": r["trace"][2]["actual_kept_l2_fraction"],
            "stages_last_year": r["trace"][2]["stages"],
            "values_vs_exact": {
                m: {"approx": vals[m], "exact": FIVE[m],
                    "abs_err": abs(vals[m] - FIVE[m]),
                    "rel_err_nonzero":
                        (abs(vals[m] - FIVE[m]) / FIVE[m]) if FIVE[m] != 0 else None}
                for m in METRICS},
            "final_p_l1": float(p_tr.sum()),
            "final_effective_100mass_fraction": float(p_tr.sum() / exact_l1)
                if exact_l1 else None,
        })
    return {
        "tt_max_rank_note": "7-bit TT has maximum bond dimension 8; "
                            "chi=8/10 are exact by construction, NOT a "
                            "scaling-advantage evidence",
        "exact_reference": {"L1": exact_l1, "L2_sq": exact_l2,
                            "metrics_adb_base": FIVE},
        "runs": runs,
    }


def cost_r1() -> dict:
    out = {"entries": {}}
    # actually-constructed index tables (2-D), not the theoretical 3-D dense M
    out["entries"]["merge_index_table_n2_elements"] = 100 * 100
    out["entries"]["merge_index_table_n3_elements"] = 1000 * 1000
    out["entries"]["theoretical_dense_M_tensor_n2"] = "not materialized (100^3=1e6 theoretical)"
    out["entries"]["theoretical_dense_M_tensor_n3"] = "not materialized (1000^3=1e9 theoretical)"
    out["entries"]["annual_prob_tensor_n2_elements"] = 100
    out["entries"]["annual_prob_tensor_n3_elements"] = 1000
    out["entries"]["terminal_peak_n2_elements"] = 100
    out["entries"]["terminal_peak_n3_elements"] = 1000
    out["entries"]["measurement_tensor_peak_elements"] = 1000  # n3 observables
    # timing
    sc = SCEN["adb-base"]
    phases = {}
    t0 = time.perf_counter(); merge_map_n3(); phases["merge_table_build_n3"] = time.perf_counter() - t0
    t0 = time.perf_counter(); annual_n3(sc); phases["annual_factor_build_n3"] = time.perf_counter() - t0
    t0 = time.perf_counter(); N.contract_n3_terminal(sc); phases["exact_contraction_n3"] = time.perf_counter() - t0
    t0 = time.perf_counter(); terminal_dp_n3(annual_n3(sc)); phases["matched_n3_exact_dp"] = time.perf_counter() - t0
    t0 = time.perf_counter(); N.contract_truncated_n2_diagnostic(sc, 4); phases["diagnostic_truncated_chi4"] = time.perf_counter() - t0
    out["phases_seconds"] = phases
    out["mode"] = "single-thread CPU, float64, same machine"
    out["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return out


def run_all(outdir: str = "results/g3_b_p0") -> dict:
    n3 = n3_fixtures()
    chi = chi_diagnostics()
    cost = cost_r1()
    payload = {"gate": "G3_B_P0_R1_SYNTHETIC_FIXTURE_AND_DIAGNOSTIC_CLOSEOUT",
               "n3_fixtures": n3,
               "n3_fixtures_all_pass": bool(
                   all(x["pass_1e12"] for x in n3 if "pass_1e12" in x)
                   and all(x["marginals_pass_1e12"]
                           for x in n3 if "marginals_pass_1e12" in x)),
               "chi_diagnostics": chi,
               "cost_ledger_r1": cost}
    w = pathlib.Path(outdir)
    w.mkdir(parents=True, exist_ok=True)
    (w / "r1_n3_fixtures.json").write_text(json.dumps({"fixtures": n3}, indent=2))
    (w / "r1_chi_diagnostics.json").write_text(json.dumps(chi, indent=2))
    (w / "r1_cost_ledger.json").write_text(json.dumps(cost, indent=2))
    (w / "r1_full.json").write_text(json.dumps(payload, indent=2))
    return payload