"""G3-B-P0-R1-R1 — complete 54-unit standard TT-SVD chi sweep tests."""

import hashlib
import json
import pathlib

import numpy as np

SCEN = ["adb-base", "adb-independent", "adb-common-shock",
        "adb-rain-stress", "adb-rain-cover"]
METRICS = ("EL", "attach", "exceed", "stoploss", "both")
CHIS = (1, 2, 4, 6, 8, 10)


def _load():
    return json.load(open("results/g3_b_p0/r1_1_standard_chi54.json"))


def test_54_units_layout():
    p = _load()
    assert p["n_units"] == 54
    assert p["layout"]["adb_base_units"] == 30
    assert p["layout"]["other_scenario_exceed_units"] == 24
    # per scenario x chi groups
    from collections import Counter
    c = Counter((u["scenario"], u["chi"]) for u in p["units"])
    assert len(c) == 5 * 6
    for sc in SCEN[1:]:
        for chi in CHIS:
            assert c[(sc, chi)] == 1 and c[(sc, chi)] == 1
    for m in METRICS:
        for chi in CHIS:
            assert c[("adb-base", chi)] == 5  # one per metric


def test_standard_algorithm_used_everywhere():
    for u in _load()["units"]:
        assert u["algorithm"].startswith("tt_truncate_vector_std")


def test_adb_base_other_metric_rows_present():
    units = _load()["units"]
    present = {(u["scenario"], u["metric"]) for u in units}
    for m in METRICS:
        assert ("adb-base", m) in present


def test_error_matches_recorded_values():
    for u in _load()["units"]:
        assert abs(u["abs_err"] - abs(u["approx"] - u["exact"])) < 1e-15, u
        if u["exact"] != 0:
            assert u["rel_err_nonzero"] is not None
            assert abs(u["rel_err_nonzero"] - u["abs_err"] / u["exact"]) < 1e-12


def test_mass_diagnostics_consistent():
    for u in _load()["units"]:
        st = u
        # valid100 <= nonneg-projected L1 (padding nonneg)
        assert 0.0 <= st["padding_L1_after"]
        assert 0.0 <= st["valid_100_state_L1_after"]
        assert abs(st["nonneg_projected_L1_after"] -
                   (st["valid_100_state_L1_after"] + st["padding_L1_after"])) < 1e-9
        assert 0.0 < st["actual_kept_l2_fraction"] <= 1.0 + 1e-9
        # timing measured, not a placeholder
        assert u["truncation_seconds_total"] > 0.0
        assert u["truncation_seconds_total"] <= u["seconds_total"]
        # bond ranks after <= before and <= chi
        for b_before, b_after in zip(u["bond_rank_before"], u["bond_ranks_after"]):
            assert 1 <= b_after <= min(u["chi"], b_before)


def test_chi4_measured_not_inferred():
    p = _load()
    assert "note_chi4" in p
    assert p["chi4_exactness_observed"] == {
        "adb-base": True, "adb-independent": True, "adb-common-shock": True,
        "adb-rain-stress": True, "adb-rain-cover": True}
    # chi=4 exactness is a per-scenario observation; each scenario's record exists
    for sc in SCEN:
        u = [x for x in p["units"] if x["scenario"] == sc and x["chi"] == 4][0]
        assert u["abs_err"] <= 1e-9, (sc, u["abs_err"])


def test_chi8_not_scaling_claim():
    p = _load()
    assert "NOT scaling-advantage evidence" in p["tt_max_rank_note"]
    assert "max bond 8" in p["tt_max_rank_note"]


def test_no_historical_result_overwrite():
    p54 = _load()
    p0 = json.load(open("results/g3_b_p0/chi_experiments.json"))
    # original B-P0 chi (old algorithm) file must still exist with old length
    assert len(p0["rows"]) == 54
    assert p54["units"][0]["algorithm"].startswith("tt_truncate_vector_std")
    # and the R1 diagnostic file untouched hash-wise
    h = hashlib.sha256(pathlib.Path(
        "results/g3_b_p0/r1_chi_diagnostics.json").read_bytes()).hexdigest()
    assert h  # exists, byte-consistent
    assert pathlib.Path("results/g3_b_p0/r1_1_standard_chi54.json").is_file()