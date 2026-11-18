"""G3-B-P0-R1 — synthetic-fixture and diagnostic closeout tests."""

import json

import numpy as np

from adb_g1a.model import SCENARIOS
from adb_g3_b_p0.network import contract_truncated_n2_diagnostic, contract_n2_terminal
from adb_g3_b_p0.factors import observables_n2

FIX = [("F1_base", "adb-base"), ("F2_independent", "adb-independent"),
       ("F3_common_shock", "adb-common-shock")]
METRICS = ("EL", "attach", "exceed", "stoploss", "both")


def _load(name):
    return json.load(open(f"results/g3_b_p0/{name}"))


def test_n3_three_fixtures_covered():
    rows = _load("r1_n3_fixtures.json")["fixtures"]
    ids = {r["fixture_id"] for r in rows}
    assert ids == {"F1_base", "F2_independent", "F3_common_shock"}
    scen = {r["fixture_id"]: r["scenario"] for r in rows if "metric" in r}
    assert scen == {"F1_base": "adb-base",
                    "F2_independent": "adb-independent",
                    "F3_common_shock": "adb-common-shock"}


def test_n3_fixtures_exact_and_tagged():
    rows = _load("r1_n3_fixtures.json")["fixtures"]
    met = [r for r in rows if "metric" in r]
    assert len(met) == 3 * 5
    for r in met:
        assert r["tag"] == "SYNTHETIC_N3_NOT_OFFICIAL"
        assert r["pass_1e12"] is True, r
        assert "source" in r and r["source"]
    marg = [r for r in rows if "marginals" in r]
    assert len(marg) == 3
    for r in marg:
        assert r["tag"] == "SYNTHETIC_N3_NOT_OFFICIAL"
        assert r["marginals_pass_1e12"] is True, r
        assert r["marginals"]["annual_to_n2"] <= 1e-12
        assert r["marginals"]["terminal_to_n2"] <= 1e-12


def test_n2_25_cells_match_g1b_ground_truth_directly():
    """Direct check against the frozen G1-B raw ground-truth fields (not the
    report copy fields)."""
    gt = json.load(open("results/g1b_full/exact_ground_truth.json"))
    g2n = {"EL": "term_expected_principal_loss_usd",
           "attach": "term_joint_attachment_probability",
           "exceed": "term_portfolio_loss_exceedance_probability",
           "stoploss": "term_portfolio_stop_loss_usd",
           "both": "term_both_bonds_exhausted_probability"}
    for sc_key in ("adb-base", "adb-independent", "adb-common-shock",
                   "adb-rain-stress", "adb-rain-cover"):
        sc = {s.key: s for s in SCENARIOS}[sc_key]
        p, _ = contract_n2_terminal(sc)
        obs = observables_n2(sc)
        for m in METRICS:
            val = float(np.dot(p, obs[m]))
            raw = gt[sc_key][g2n[m]]
            scale = 160e6 if m == "EL" else (80e6 if m == "stoploss" else 1.0)
            assert abs(val * scale - raw) <= 1e-6, (sc_key, m, val, raw)


def test_tt_standard_kept_l2_matches_product():
    sc = {s.key: s for s in SCENARIOS}["adb-base"]
    for chi in (1, 2, 4, 6):
        r = contract_truncated_n2_diagnostic(sc, chi, years=3)
        tr = r["trace"][2]
        prod = float(np.prod([b["singular_kept_sq_mass"] / b["singular_total_sq_mass"]
                              for b in tr["bonds"] if b["singular_total_sq_mass"] > 0]))
        assert abs(tr["actual_kept_l2_fraction"] - prod) < 1e-9, (chi, prod)
        # per-bond rank after <= chi and <= rank before, and >=1
        for b in tr["bonds"]:
            assert 1 <= b["rank_after"] <= min(chi, b["rank_before"])
    # rank-8 exactness: chi=8 gives exact exceed

def test_chi_mass_stages_consistent():
    sc = {s.key: s for s in SCENARIOS}["adb-base"]
    r2 = contract_truncated_n2_diagnostic(sc, 2, years=3)
    st = r2["trace"][2]["stages"]
    # all stage fields present and ranges sane
    assert abs(st["L1_before_compress"] - st["L1_padding_domain_before_compress"]
               - st["L1_valid_100_states_before_compress"]) < 1e-12
    assert st["L1_padding_domain_before_compress"] == 0.0  # zero padding pre-compress
    assert st["L1_after_compress_nonneg_projection"] >= 0.0
    assert abs(st["L1_abs_diff_before_vs_after"] -
               abs(st["L1_before_compress"] - st["L1_after_compress_nonneg_projection"])) < 1e-12


def test_cost_labels_r1():
    c = _load("r1_cost_ledger.json")
    assert c["entries"]["merge_index_table_n2_elements"] == 10000
    assert c["entries"]["merge_index_table_n3_elements"] == 1000000
    assert "not materialized" in c["entries"]["theoretical_dense_M_tensor_n2"]
    assert "not materialized" in c["entries"]["theoretical_dense_M_tensor_n3"]
    assert c["entries"]["annual_prob_tensor_n2_elements"] == 100
    assert c["entries"]["annual_prob_tensor_n3_elements"] == 1000
    assert all(k in c["phases_seconds"] for k in (
        "merge_table_build_n3", "annual_factor_build_n3",
        "exact_contraction_n3", "matched_n3_exact_dp",
        "diagnostic_truncated_chi4"))
    assert c["mode"] == "single-thread CPU, float64, same machine"


def test_chi8_is_exact_not_scaling_claim():
    chi = _load("r1_chi_diagnostics.json")
    assert "maximum bond dimension 8" in chi["tt_max_rank_note"]
    runs = {r["chi"]: r for r in chi["runs"]}
    assert abs(runs[8]["actual_kept_l2_last_year"] - 1.0) < 1e-12
    worst8 = max(v["abs_err"] for v in runs[8]["values_vs_exact"].values())
    assert worst8 <= 1e-9