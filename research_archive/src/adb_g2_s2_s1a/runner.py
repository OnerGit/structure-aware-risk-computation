"""G2-S2-S1A-R2 measurement runner (CPU-only, 1 thread).

R2 contract: (01) per-metric W_j minima computed and reported separately from
all-five simultaneous minima; (02) honest e2e timing starts before the annual
law and ends after all fixture production, and isolated stage timings use
precomputed cores so SVD, reconstruction+contraction and threshold passes are
not conflated.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np

from .laws import (annual_law_n3, frozen_n2_annual, frozen_n2_terminal,
                   marginalize_n3_to_n2, scenario_index, terminal_dp_n3)
from .observables import n3_synthetic_observable_vectors
from .tensor import (minimal_chi_per_metric, minimal_chi_per_metric_rel,
                     minimal_chi_relative, minimal_chi_threshold,
                     mode1_spectrum, numeric_rank, p_row_from_cores,
                     to_tt_svd, w_mode1_diagnostics, w_row_from_cores)
from .transfer import (contraction_targets, svd_diagnostics, terminal_law,
                       transfer_matrix)
from .observables import n2_observable_vectors  # noqa: F401

CHIS = [1, 2, 4, 6, 8, 10]
ABSTOL = 1e-12
RELTOL = 0.10

E2E_BOUNDARY = ("run_fixture e2e starts before annual-law computation and stops "
               "after the complete result payload is assembled; the "
               "self-referential run_fixture_e2e_s is inserted last")
ALL_SIX = ("minimal_chi_threshold(P,abs)", "minimal_chi_relative(P,rel)",
           "minimal_chi_per_metric(W,abs)", "minimal_chi_per_metric_rel(W,rel)",
           "minimal_chi_threshold(W,abs)", "minimal_chi_relative(W,rel)")


def run_minima(p_rows, w_rows) -> dict:
    """The complete set of six minima evaluations (C-R3-01). Pure minima: no
    row construction, no to_tt_svd, no numpy.linalg.svd."""
    return {
        "P_abs": minimal_chi_threshold(p_rows, ABSTOL),
        "P_rel": minimal_chi_relative(p_rows, RELTOL),
        "W_abs_per": minimal_chi_per_metric(w_rows, ABSTOL),
        "W_rel_per": minimal_chi_per_metric_rel(w_rows, RELTOL),
        "W_abs_all": minimal_chi_threshold(w_rows, ABSTOL),
        "W_rel_all": minimal_chi_relative(w_rows, RELTOL),
    }


def mass_flags(dist) -> dict:
    return {"mass": float(np.sum(dist)),
            "min": float(np.min(dist)),
            "max": float(np.max(dist))}


def run_n2(sc_key: str) -> dict:
    sc = scenario_index()[sc_key]
    annual = frozen_n2_annual(sc)
    T = transfer_matrix(annual)
    term = terminal_law(T)
    ref = frozen_n2_terminal(sc)
    g = n2_observable_vectors(sc.eq_cover, sc.rain_cover)
    targets = contraction_targets(annual, g)
    from adb_g1a.model import compute_scenario
    exact = compute_scenario(sc)
    oracle = {
        "EL": exact["term_expected_principal_loss_usd"] / 160e6,
        "attach": exact["term_joint_attachment_probability"],
        "exceed": exact["term_portfolio_loss_exceedance_probability"],
        "stoploss": exact["term_portfolio_stop_loss_usd"] / 80e6,
        "both": exact["term_both_bonds_exhausted_probability"],
    }
    errs = {k: abs(targets[k] - oracle[k]) for k in oracle}
    return {
        "scenario": sc_key,
        "term_max_abs_dev": float(np.max(np.abs(term - ref))),
        "targets": {k: float(v) for k, v in targets.items()},
        "frozen_oracle_computed_from_model": oracle,
        "vs_frozen_oracle_abs_errors": {k: float(v) for k, v in errs.items()},
        "max_vs_frozen_oracle_abs_error": max(errs.values()),
        "transfer_rowsum_max": float(np.max(np.abs(T.sum(axis=1) - 1.0))),
        "mass": mass_flags(term),
        "svd": svd_diagnostics(term, g),
    }


def run_fixture(sc_key: str, tag: str) -> dict:
    sc = scenario_index()[sc_key]
    g = n3_synthetic_observable_vectors(sc.eq_cover, sc.rain_cover)
    t_e2e = time.perf_counter()          # R3-T03: e2e starts before annual law
    t0 = time.perf_counter(); annual = annual_law_n3(sc)
    t_annual = time.perf_counter() - t0
    t0 = time.perf_counter(); term = terminal_dp_n3(annual)
    t_dp = time.perf_counter() - t0
    P = term.reshape(10, 10, 10)
    W = {k: (term * gg.ravel()).reshape(10, 10, 10) for k, gg in g.items()}

    # 1) SVD-only: TT core decompositions (no reconstruction, no raw spectra)
    t0 = time.perf_counter()
    p_cores = {chi: to_tt_svd(P, chi) for chi in CHIS}
    t_p_svd = time.perf_counter() - t0
    t0 = time.perf_counter()
    w_cores = {chi: {k: to_tt_svd(W[k], chi) for k in W} for chi in CHIS}
    t_w_svd = time.perf_counter() - t0

    # 2) RAW-mode-1 diagnostics (spectra + numeric ranks) — separate SVD stage
    t0 = time.perf_counter()
    w1p = mode1_spectrum(P)
    p_diag = {"mode1_spectrum": [float(x) for x in w1p],
              "mode1_rank_tol1e10": numeric_rank(w1p, 1e-10),
              "mode1_rank_tol1e8": numeric_rank(w1p, 1e-8)}
    w_diag = {k: w_mode1_diagnostics(Wk) for k, Wk in W.items()}
    t_svd_diag = time.perf_counter() - t0

    # 3) PURE reconstruction+contraction (precomputed cores AND diag): no SVD
    t0 = time.perf_counter()
    p_rows = [p_row_from_cores(chi, *p_cores[chi], P, g) for chi in CHIS]
    w_rows = []
    for chi in CHIS:
        per = {k: w_row_from_cores(k, chi, *w_cores[chi][k], W[k], w_diag[k])
               for k in W}
        w_rows.append({"strategy": "W_TT_THEN_SUM", "chi": int(chi),
                       "per_metric": per})
    t_recon = time.perf_counter() - t0

    # 4) minima: exactly ONE shared pure pass over precomputed rows (C-R3-04/P-01)
    t0 = time.perf_counter()
    minima = run_minima(p_rows, w_rows)
    t_thresh = time.perf_counter() - t0
    min_p_abs, min_p_rel = minima["P_abs"], minima["P_rel"]
    w_abs_per, w_rel_per = minima["W_abs_per"], minima["W_rel_per"]
    allfive_abs, allfive_rel = minima["W_abs_all"], minima["W_rel_all"]

    # 6) assemble the complete result payload FIRST (C-R3-03), then measure e2e
    payload = {
        "tag": tag, "scenario": sc_key,
        "annual_mass": mass_flags(annual),
        "term_mass": mass_flags(term),
        "observables_synth": {k: float(np.sum(term * gg.ravel()))
                              for k, gg in g.items()},
        "timing_one_full": {
            "annual_build_s": t_annual, "dp_s": t_dp,
            "p_svd_allchi_s": t_p_svd, "w_svd_allchi_s": t_w_svd,
            "svd_diagnostics_s": t_svd_diag,
            "recon_contract_allchi_s": t_recon, "threshold_s": t_thresh,
            "stages_sum_s": t_annual + t_dp + t_p_svd + t_w_svd + t_svd_diag
                            + t_recon + t_thresh,
        },
        "P_TT_THEN_CONTRACT": {
            "rows": p_rows,
            "minimal_chi_abs_1e12": int(min_p_abs),
            "minimal_chi_rel_10pct": int(min_p_rel),
            "mode1_spectrum": list(p_diag["mode1_spectrum"]),
            "mode1_rank_tol1e10": int(p_diag["mode1_rank_tol1e10"]),
            "mode1_rank_tol1e8": int(p_diag["mode1_rank_tol1e8"]),
        },
        "W_TT_THEN_SUM": {
            "rows": w_rows,
            "minimal_chi_abs_1e12_per_metric": {k: int(v) for k, v in w_abs_per.items()},
            "minimal_chi_rel_10pct_per_metric": {k: int(v) for k, v in w_rel_per.items()},
            "minimal_chi_abs_1e12_all_five": int(allfive_abs),
            "minimal_chi_rel_10pct_all_five": int(allfive_rel),
        },
    }
    # e2e covers payload assembly; insert the self-referential field last
    payload["timing_one_full"]["run_fixture_e2e_s"] =         time.perf_counter() - t_e2e
    return payload


def annual_marginal_to_n2(annual_cc: np.ndarray) -> np.ndarray:
    from .laws import terminal_code
    out = np.zeros(100)
    for c1 in range(10):
        a1, r1 = divmod(c1, 2)
        for c2 in range(10):
            a2, r2 = divmod(c2, 2)
            ii = terminal_code(a1, a2, r1, r2)
            for c3 in range(10):
                out[ii] += annual_cc[c1 * 100 + c2 * 10 + c3]
    return out


def warm_fixture(sc_key: str, tag: str) -> dict:
    """Three-rep warm timings for one fixture (C-R3-01/02): all six minima in
    the threshold stage; mode-1 diagnostics computed exactly once per rep inside
    the dedicated timer and reused downstream; no duplicated W diagnostics."""
    sc = scenario_index()[sc_key]
    g = n3_synthetic_observable_vectors(sc.eq_cover, sc.rain_cover)
    rep = dict(annual_build=[], dp=[], p_svd=[], w_svd=[],
               svd_diagnostics=[], recon_contract=[], threshold=[])
    for _ in range(3):
        t = time.perf_counter(); annual = annual_law_n3(sc)
        rep["annual_build"].append(time.perf_counter() - t)
        t = time.perf_counter(); term = terminal_dp_n3(annual)
        rep["dp"].append(time.perf_counter() - t)
        P = term.reshape(10, 10, 10)
        W = {k: (term * gg.ravel()).reshape(10, 10, 10) for k, gg in g.items()}
        p_cores = {}; t = time.perf_counter()
        for chi in CHIS: p_cores[chi] = to_tt_svd(P, chi)
        rep["p_svd"].append(time.perf_counter() - t)
        w_cores = {}; t = time.perf_counter()
        for chi in CHIS:
            w_cores[chi] = {k: to_tt_svd(W[k], chi) for k in W}
        rep["w_svd"].append(time.perf_counter() - t)
        # diagnostics: exactly one P + five W mode-1 spectra, computed once here
        t = time.perf_counter()
        w1p = mode1_spectrum(P)
        p_diag = {"mode1_spectrum": [float(x) for x in w1p],
                  "mode1_rank_tol1e10": numeric_rank(w1p, 1e-10),
                  "mode1_rank_tol1e8": numeric_rank(w1p, 1e-8)}
        w_diag = {k: w_mode1_diagnostics(Wk) for k, Wk in W.items()}
        rep["svd_diagnostics"].append(time.perf_counter() - t)
        # pure recon+contract using cores AND the same cached diagnostics
        t = time.perf_counter()
        p_rows = [p_row_from_cores(chi, *p_cores[chi], P, g) for chi in CHIS]
        w_rows = []
        for chi in CHIS:
            per = {k: w_row_from_cores(k, chi, *w_cores[chi][k], W[k], w_diag[k])
                   for k in W}
            w_rows.append({"strategy": "W_TT_THEN_SUM", "chi": chi,
                           "per_metric": per})
        rep["recon_contract"].append(time.perf_counter() - t)
        # threshold: all six minima only (rows already built)
        t = time.perf_counter()
        run_minima(p_rows, w_rows)
        rep["threshold"].append(time.perf_counter() - t)
    return {k: {"reps_s": v, "median_s": float(np.median(v))}
            for k, v in rep.items()}


def run_all(out_dir: Path) -> dict:
    t_start = time.perf_counter()
    n2 = {k: run_n2(k) for k in scenario_index()}
    fixtures = [("adb-base", "F1_base"), ("adb-independent", "F2_independent"),
                ("adb-common-shock", "F3_common_shock")]
    n3 = {tag: run_fixture(sc, tag) for sc, tag in fixtures}

    warm = {}
    for sc_key, tag in fixtures:
        warm[tag] = warm_fixture(sc_key, tag)
    results = {
        "gate": "G2_S2_S1A_R3_R2_MINIMAL_CLOSEOUT_PATCH",
        "n2": n2,
        "n3": n3,
        "warm_timings": warm,
        "timing_labels": {
            "annual_build": "annual_law_n3(...) only",
            "dp": "terminal_dp_n3(annual) only",
            "p_svd": "to_tt_svd(P, chi) core decomposition only (no reconstruction)",
            "w_svd": "to_tt_svd(W_k, chi) core decomposition only (no reconstruction)",
            "svd_diagnostics": "RAW numpy.linalg.svd of mode-1 unfolding for P and each W_j (spectra + numeric ranks); separate from core decomposition",
            "recon_contract": "tt_reconstruct + scalar contract/sum using PRECOMPUTED cores AND PRECOMPUTED mode-1 diagnostics; zero to_tt_svd and zero numpy.linalg.svd inside",
            "threshold": "only minimal_chi_* minima functions on precomputed rows; row construction/reconstruction/SVD occur before the timer",
            "run_fixture_e2e": "entire run_fixture from before annual-law computation to AFTER the complete result payload is assembled; self-referential e2e inserted last",
            "e2e_boundary": E2E_BOUNDARY,
            "PURE_CONTRACTION_TIMING": "MEASURED",
        },
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "wall_s": time.perf_counter() - t_start,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "summary.txt").write_text(_summarize(results) + "\n")
    return results


def _summarize(r) -> str:
    lines = []
    for k, v in r["n2"].items():
        lines.append(f"n2 {k}: max_vs_frozen_oracle={v['max_vs_frozen_oracle_abs_error']:.3e}")
    for tag, v in r["n3"].items():
        P = v["P_TT_THEN_CONTRACT"]; W = v["W_TT_THEN_SUM"]
        lines.append(
            f"{tag}: P min_chi_abs={P['minimal_chi_abs_1e12']} rel={P['minimal_chi_rel_10pct']} | "
            f"W per-metric abs={W['minimal_chi_abs_1e12_per_metric']} rel={W['minimal_chi_rel_10pct_per_metric']} | "
            f"W all-five abs={W['minimal_chi_abs_1e12_all_five']} rel={W['minimal_chi_rel_10pct_all_five']} | "
            f"e2e={v['timing_one_full']['run_fixture_e2e_s']:.4f}s")
    lines.append(f"gate_wall_s={r['wall_s']:.4f} peak_rss_kb={r['peak_rss_kb']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/g2_s2_s1a")
    a = ap.parse_args()
    res = run_all(Path(a.out))
    print(_summarize(res))