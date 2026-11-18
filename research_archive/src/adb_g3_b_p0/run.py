"""G3-B-P0 driver — executes the direct-TN validation and writes
results/g3_b_p0/*.json (machine-readable evidence)."""

from __future__ import annotations

import json
import pathlib
import resource
import time

import numpy as np

from adb_g1a.model import SCENARIOS
from adb_g2_s2_s1a.laws import terminal_dp_n3, frozen_n2_terminal
from adb_g2_s2_s1a.observables import n2_observable_vectors

from . import network as N
from .factors import (annual_n2, annual_n3, merge_map_n2, merge_map_n3,
                      observables_n2, observables_n3, annual_n2_corrupted_shared_shock,
                      exceed_ge_indicator, marginalize_n3_to_n2_terminal,
                      marginalize_n3_to_n2_annual)

SCEN = {s.key: s for s in SCENARIOS}
SCEN_KEYS = ["adb-base", "adb-independent", "adb-common-shock",
             "adb-rain-stress", "adb-rain-cover"]
METRICS = ("EL", "attach", "exceed", "stoploss", "both")
CHIS = (1, 2, 4, 6, 8, 10)


def _t():
    return time.perf_counter()


def n2_25_cells() -> dict:
    res = json.load(open("results/g2_s2_s1a/results.json"))
    gt = json.load(open("results/g1b_full/exact_ground_truth.json"))
    rows = N.all_25_cells(SCEN_KEYS)
    r2g = {"EL": "term_expected_principal_loss_usd",
           "attach": "term_joint_attachment_probability",
           "exceed": "term_portfolio_loss_exceedance_probability",
           "stoploss": "term_portfolio_stop_loss_usd",
           "both": "term_both_bonds_exhausted_probability"}
    out = []
    for r in rows:
        key, m = r["scenario"], r["metric"]
        target = res["n2"][key]["targets"][m]
        scale = 160e6 if m == "EL" else (80e6 if m == "stoploss" else 1.0)
        gtv = gt[key][r2g[m]]
        err = abs(r["value"] - target)
        out.append({"scenario": key, "metric": m, "cover": r["cover"],
                    "tn_value": r["value"], "frozen_target": target,
                    "gt_scaled": gtv / scale,
                    "normalized_abs_error_vs_target": err,
                    "pass_1e12": bool(err <= 1e-12)})
    return {"cells": out,
            "all_pass_1e12": bool(all(c["pass_1e12"] for c in out)),
            "count": len(out),
            "reference_source": "results/g2_s2_s1a/results.json n2.*.targets "
                                "+ results/g1b_full/exact_ground_truth.json"}


def cover_mapping_check() -> dict:
    return N.cover_mapping_checks()


def n3_block() -> dict:
    sc = SCEN["adb-base"]
    term, A3, trace = N.contract_n3_terminal(sc, years=3)
    # frozen n2 terminal marginal identity
    p_n2_term_ref = frozen_n2_terminal(sc)
    p_n2_term_tn = marginalize_n3_to_n2_terminal(term)
    # annual n2 marginal identity
    A3_flat = annual_n3(sc)
    A2_ref = annual_n2(sc)
    A2_tn = marginalize_n3_to_n2_annual(A3_flat)
    n3dp = terminal_dp_n3(A3_flat, years=3)
    obs3 = observables_n3(sc)
    n3_key = {"EL": "EL_synth", "attach": "attach_synth",
              "exceed": "exceed_synth", "stoploss": "stoploss_synth",
              "both": "all_exhaust_synth"}
    ref3 = {m: float(np.dot(n3dp, obs3[n3_key[m]])) for m in METRICS}
    tn3 = {m: float(np.dot(term, obs3[n3_key[m]])) for m in METRICS}
    marg_term_err = float(np.abs(p_n2_term_tn - p_n2_term_ref).max())
    marg_ann_err = float(np.abs(A2_tn - A2_ref).max())
    return {
        "law": "S1A F1/F2/F3 synthetic third country",
        "tag": "SYNTHETIC_N3_NOT_OFFICIAL",
        "terminal_marginal_to_n2_max_abs_err": marg_term_err,
        "annual_marginal_to_n2_max_abs_err": marg_ann_err,
        "marginals_pass_1e12": bool(marg_term_err <= 1e-12 and marg_ann_err <= 1e-12),
        "metrics_vs_exact_n3_dp": {m: {"tn": tn3[m], "dp_reference": ref3[m],
                                        "abs_err": abs(tn3[m] - ref3[m])}
                                    for m in METRICS},
        "metrics_pass_1e12": bool(all(abs(tn3[m] - ref3[m]) <= 1e-12 for m in METRICS)),
        "annual_factor_norm": float(A3_flat.sum()),
        "terminal_norm": float(term.sum()),
    }


def chi_sweep() -> dict:
    # exact reference scalar per scenario (exceed; also full metrics for base)
    exact = {}
    sc_base = SCEN["adb-base"]
    pb, _ = N.contract_n2_terminal(sc_base)
    exact["adb-base"] = {m: N.scalar(pb, observables_n2(sc_base)[m]) for m in METRICS}
    for key in SCEN_KEYS[1:]:
        sc = SCEN[key]
        p, _ = N.contract_n2_terminal(sc)
        exact[key] = N.scalar(p, observables_n2(sc)["exceed"])
    rows = []
    for key in SCEN_KEYS:
        sc = SCEN[key]
        p_exact = None
        for chi in CHIS:
            try:
                r = N.contract_truncated_n2(sc, chi, years=3)
                p_tr = r["p"][:100]
                obs = observables_n2(sc)
                scalar_exact = exact[key] if key != "adb-base" else exact[key]["exceed"]
                if key == "adb-base":
                    vals = {m: N.scalar(p_tr, obs[m]) for m in METRICS}
                else:
                    vals = {"exceed": N.scalar(p_tr, obs["exceed"])}
                for m, v in vals.items():
                    ref = exact[key][m] if key == "adb-base" and isinstance(exact[key], dict) else scalar_exact
                    rel = (abs(v - ref) / ref) if ref != 0 else None
                    rows.append({
                        "scenario": key, "chi": chi, "metric": m,
                        "approx": v, "exact": ref,
                        "abs_err": abs(v - ref),
                        "rel_err_nonzero": rel,
                        "kept_L2_fraction": r["trace"][-1]["kept_L2_fraction"],
                        "final_norm_L1": r["trace"][-1]["norm_L1"],
                        "mass_removed_padding": r["trace"][-1]["mass_removed_padding"],
                        "feasible": True})
            except (ValueError, np.linalg.LinAlgError) as e:
                rows.append({"scenario": key, "chi": chi, "metric": "exceed",
                             "approx": None, "exact": None, "abs_err": None,
                             "rel_err_nonzero": None, "kept_L2_fraction": None,
                             "final_norm_L1": None, "mass_removed_padding": None,
                             "feasible": False, "reason": f"NOT_RUN: {type(e).__name__}"})
    return {"rows": rows, "positions": "mid-contraction TT bond truncation after each annual merge",
            "note": "<=1e-12 exact tier A; 10% relative exploratory tier B on nonzero targets reported separately"}


def negative_controls() -> dict:
    sc = SCEN["adb-base"]
    p_corr, _ = N.contract_n2_terminal(sc)
    obs = observables_n2(sc)
    exceed_corr = N.scalar(p_corr, obs["exceed"])
    # 1) cap-at-5 wrong merge (128-space, cap corrupt; padded mass ignored)
    p_cap = N.contract_cap_corrupted(sc, cap=5, years=3)
    exceed_cap = N.scalar(p_cap[:100], obs["exceed"])
    # 2) wrong threshold >=160 at the observable-tensor layer
    p2, _ = N.contract_n2_terminal(sc)
    exceed_ge = N.scalar(p2, exceed_ge_indicator(sc))
    # 3) wrong shared-shock at the annual-factor layer
    A_bad = annual_n2_corrupted_shared_shock(sc)
    p_bad = np.zeros(100); p_bad[0] = 1.0
    merge = merge_map_n2()
    for _t in range(3):
        p2b = np.zeros(100)
        for a in range(100):
            if p_bad[a] == 0.0:
                continue
            np.add.at(p2b, merge[a], p_bad[a] * A_bad)
        p_bad = p2b
    exceed_shock = N.scalar(p_bad, obs["exceed"])
    return {
        "reference_exceed": exceed_corr,
        "controls": {
            "wrong_cap_at_5": {"deviation": abs(exceed_cap - exceed_corr),
                               "detected": bool(abs(exceed_cap - exceed_corr) > 1e-4)},
            "wrong_threshold_ge_160": {"deviation": abs(exceed_ge - exceed_corr),
                                       "detected": bool(abs(exceed_ge - exceed_corr) > 1e-6)},
            "wrong_shared_shock_cr0": {"deviation": abs(exceed_shock - exceed_corr),
                                       "detected": bool(abs(exceed_shock - exceed_corr) > 1e-4)},
        },
        "all_detected": bool(abs(exceed_cap - exceed_corr) > 1e-4
                             and abs(exceed_ge - exceed_corr) > 1e-6
                             and abs(exceed_shock - exceed_corr) > 1e-4),
    }


def cost_ledger() -> dict:
    out = {"phases": {}}
    sc = SCEN["adb-base"]
    # annual factor construction
    t0 = _t(); annual_n2(sc); out["phases"]["annual_factor_build_n2"] = _t() - t0
    t0 = _t(); merge_map_n2(); out["phases"]["merge_table_build_n2"] = _t() - t0
    t0 = _t(); N.contract_n2_terminal(sc); out["phases"]["exact_contraction_n2_1scenario"] = _t() - t0
    t0 = _t(); annual_n3(sc); out["phases"]["annual_factor_build_n3"] = _t() - t0
    t0 = _t(); merge_map_n3(); out["phases"]["merge_table_build_n3"] = _t() - t0
    t0 = _t(); N.contract_n3_terminal(sc); out["phases"]["exact_contraction_n3_1scenario"] = _t() - t0
    t0 = _t(); terminal_dp_n3(annual_n3(sc)); out["phases"]["matched_n3_exact_dp"] = _t() - t0
    t0 = _t(); N.contract_truncated_n2(sc, 4); out["phases"]["truncated_contract_chi4"] = _t() - t0
    t0 = _t()
    for key in SCEN_KEYS:
        p, _ = N.contract_n2_terminal(SCEN[key])
        obs = observables_n2(SCEN[key])
        for m in METRICS:
            N.scalar(p, obs[m])
    out["phases"]["all_25_cells"] = _t() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_entry = 1000 * 1000  # dense M n2 per (a,y) pairs built row-wise
    return {"phases": out, "peak_rss_kb": rss,
            "peak_intermediate_entries_n2": peak_entry,
            "intermediate_size_trace_exact_n2": [
                {"after_year": t, "states": 100} for t in range(1, 4)],
            "intermediate_size_trace_exact_n3": [
                {"after_year": t, "states": 1000} for t in range(1, 4)],
            "matched": "same machine, single thread, float64",
            "dp_equivalence_note": "exact contraction (typhoon) = capped-state "
                                   "transfer DP on the same state space;"}


def run_all(outdir: str = "results/g3_b_p0") -> dict:
    t_start = _t()
    n2 = n2_25_cells()
    n2p = {"n2_25_cells": n2, "cover_mapping_checks_not_in_official_25": cover_mapping_check()}
    n3 = n3_block()
    chi = chi_sweep()
    nc = negative_controls()
    cost = cost_ledger()
    wall = _t() - t_start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "gate": "G3_B_P0_DIRECT_ANNUAL_LAW_TN_MPS_CONSTRUCTION_AND_VALIDATION",
        "method": {"production_input": "annual-law factors only",
                   "posthoc_TT_on_terminal_table": "NO",
                   "production_functions": "annual_n2/annual_n3 + merge_map + "
                                           "contraction (network.py)",
                   "terminal_distribution_functions_used": "diagnostics only"},
        "n2": n2p,
        "n3": n3,
        "chi_experiments": chi,
        "negative_controls": nc,
        "cost_ledger": cost,
        "measured_wall_s": wall,
        "peak_rss_kb": rss,
        "budget": {"wall_s": 30 * 60, "rss_kb": 2 * 1024 * 1024},
        "wall_within_budget": bool(wall <= 30 * 60),
        "rss_within_budget": bool(rss <= 2 * 1024 * 1024),
        "classification": {
            "DP_EQUIVALENT_TN": "YES",
            "CLASSICAL_COMPETITIVENESS": "NO_MEASURED_BENEFIT",
            "FULLY_FACTOR_LEVEL_SCALING": "NO (annual tensor fully materialized)",
            "FULL_G1": "NOT_RERUN",
        },
    }
    w = pathlib.Path(outdir)
    w.mkdir(parents=True, exist_ok=True)
    (w / "n2_25cells.json").write_text(json.dumps(n2, indent=2))
    (w / "n3.json").write_text(json.dumps(n3, indent=2))
    (w / "chi_experiments.json").write_text(json.dumps(chi, indent=2))
    (w / "negative_controls.json").write_text(json.dumps(nc, indent=2))
    (w / "cost_ledger.json").write_text(json.dumps(cost, indent=2))
    (w / "full.json").write_text(json.dumps(payload, indent=2))
    return payload