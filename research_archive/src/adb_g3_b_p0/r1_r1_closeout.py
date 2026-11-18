"""G3-B-P0-R1-R1 — complete 54-unit STANDARD TT-SVD chi sweep across the five
official scenarios (no historical result overwritten; new R1-R1 namespace)."""

from __future__ import annotations

import json
import pathlib
import resource
import time

import numpy as np

from adb_g1a.model import SCENARIOS
from adb_g2_s2_s1a.observables import n2_observable_vectors

from . import network as N

SCEN = {s.key: s for s in SCENARIOS}
SCEN_KEYS = ["adb-base", "adb-independent", "adb-common-shock",
             "adb-rain-stress", "adb-rain-cover"]
METRICS = ("EL", "attach", "exceed", "stoploss", "both")
CHIS = (1, 2, 4, 6, 8, 10)


def _exact_metric(sc, metric):
    p, _ = N.contract_n2_terminal(sc)
    o = n2_observable_vectors(sc.eq_cover, sc.rain_cover)[metric]
    return float(np.dot(p, o))


def standard_unit(sc_key: str, chi: int, metric: str = "exceed") -> dict:
    sc = SCEN[sc_key]
    exact = _exact_metric(sc, metric)
    t0 = time.perf_counter()
    r = N.contract_truncated_n2_diagnostic(sc, chi, years=3)
    wall = time.perf_counter() - t0
    p_tr = np.maximum(r["p"][:100], 0.0)
    obs = n2_observable_vectors(sc.eq_cover, sc.rain_cover)[metric]
    approx = float(np.dot(p_tr, obs))
    last = r["trace"][2]
    return {
        "scenario": sc_key, "chi": int(chi), "metric": metric,
        "algorithm": "tt_truncate_vector_std (standard TT-SVD, S*Vt carried)",
        "exact": exact, "approx": approx,
        "abs_err": float(abs(approx - exact)),
        "rel_err_nonzero": (float(abs(approx - exact) / exact) if exact != 0 else None),
        "actual_kept_l2_fraction": last["actual_kept_l2_fraction"],
        "bond_ranks_after": [b["rank_after"] for b in last["bonds"]],
        "bond_rank_before": [b["rank_before"] for b in last["bonds"]],
        "valid_100_state_L1_after": last["stages"]["L1_valid_100_states_after_compress"],
        "padding_L1_after": last["stages"]["L1_padding_domain_after_compress"],
        "nonneg_projected_L1_after": last["stages"]["L1_after_compress_nonneg_projection"],
        "L1_before_compress": last["stages"]["L1_before_compress"],
        "truncation_seconds_total": float(sum(x["truncation_seconds"] for x in r["trace"])),
        "seconds_total": wall,
    }


def build_54() -> dict:
    t_start = time.perf_counter()
    units = []
    # adb-base: all five metrics (6 chi x 5 = 30 units)
    for chi in CHIS:
        for m in METRICS:
            units.append(standard_unit("adb-base", chi, m))
    # other four scenarios: exceed only (6 chi x 4 = 24 units)
    for key in SCEN_KEYS[1:]:
        for chi in CHIS:
            units.append(standard_unit(key, chi, "exceed"))
    wall = time.perf_counter() - t_start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    chi4 = {}
    for u in units:
        if u["chi"] == 4:
            prev = chi4.get(u["scenario"], -1.0)
            chi4[u["scenario"]] = max(prev, u["abs_err"])
    return {
        "gate": "G3_B_P0_R1_R1_STANDARD_CHI_SWEEP_CLOSEOUT",
        "algorithm": "tt_truncate_vector_std (standard TT-SVD)",
        "units": units,
        "n_units": len(units),
        "layout": {"adb_base_units": 6 * len(METRICS),
                   "other_scenario_exceed_units": (len(SCEN_KEYS) - 1) * len(CHIS),
                   "total_expected": 54},
        "chi4_exactness_observed": {k: (float(v) <= 1e-9)
                                    for k, v in chi4.items()},
        "note_chi4": "chi=4 exactness is a MEASURED observation per scenario, "
                     "not inferred from adb-base alone",
        "tt_max_rank_note": "7-bit TT max bond 8; chi=8/10 exact by "
                            "construction, NOT scaling-advantage evidence",
        "measured_wall_s": wall, "peak_rss_kb": rss,
        "threads": 1, "float": "float64",
    }


def write(outdir: str = "results/g3_b_p0") -> str:
    payload = build_54()
    w = pathlib.Path(outdir)
    w.mkdir(parents=True, exist_ok=True)
    out = w / "r1_1_standard_chi54.json"
    out.write_text(json.dumps(payload, indent=2))
    return str(out)


if __name__ == "__main__":
    print(write())