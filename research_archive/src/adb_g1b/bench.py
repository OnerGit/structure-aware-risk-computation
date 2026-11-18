"""Benchmark runner: pilot + full experiment grid + statistical evaluation."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from adb_g1a.model import SCENARIOS, Scenario, compute_scenario
from adb_g1b.annual_law import AnnualLaw, build_annual_law
from adb_g1b.config import (BUDGETS, SOBOL_BATCHES, SOBOL_SCRAMBLES,
                            load_or_create_config)
from adb_g1b.samplers import (TERM_KEYS, estimate_trial, is_trial, is_weight_stats,
                              mc_trial, sobol_trial, stratified_layer_means,
                              stratified_trial)


def rss_kb() -> int:
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def scenario_by_key(key: str) -> Scenario:
    return next(s for s in SCENARIOS if s.key == key)


def law_for(key: str) -> AnnualLaw:
    return build_annual_law(scenario_by_key(key))


# ---- single-trial evaluation -------------------------------------------------

def run_trial(method: str, law: AnnualLaw, n: int, seed: int) -> dict:
    from adb_g1b.samplers import (interval_sufficient_stats,
                                  metric_contributions)
    rng = default_rng(int(seed))
    t0 = time.perf_counter()
    # ---- B: sampling ONLY (state generation, RNG/QMC, inverse CDF, and the
    #      logically required sampling-weight construction for IS). No metric,
    #      interval or diagnostic computation enters B (R2R1-01). ----
    if method == "mc":
        trial = mc_trial(law, n, rng)
        extra = {}
    elif method == "stratified":
        trial = stratified_trial(law, n, rng)
        extra = {}
    elif method == "is":
        trial = is_trial(law, n, rng)
        extra = {}
    else:
        raise ValueError(method)
    t_samp = time.perf_counter() - t0
    # ---- C: metrics, estimates, interval statistics, diagnostics, events ----
    t_m0 = time.perf_counter()
    if method == "mc" or method == "is":
        est = estimate_trial(law, trial)
    elif method == "stratified":
        _, layer_means, est = stratified_layer_means(law, trial)
        extra = {"layer_means": {k: v.tolist() for k, v in layer_means.items()}}
    else:
        raise ValueError(method)
    if method == "is":
        extra = {"weight_stats": is_weight_stats(trial)}
    interval = interval_sufficient_stats(law, trial)
    cont = metric_contributions(trial.states, law)
    prob_keys = ("term_joint_attachment_probability",
                 "term_portfolio_loss_exceedance_probability",
                 "term_both_bonds_exhausted_probability")
    events = {k: int(np.count_nonzero(cont[k])) for k in prob_keys}
    t_end = time.perf_counter()
    out = {"method": method, "n": int(n), "seed": int(seed),
           "paths": trial.n_paths,
           "random_draws": trial.random_draws,
           "invcdf_calls": trial.invcdf_calls,
           "deterministic_assignments": trial.deterministic_assignments,
           # R2R1-01 timing boundaries (perf_counter), per trial:
           "sampling_wall_s": t_samp,          # B (pure sampling, no metrics)
           "metrics_wall_s": t_end - t_m0,     # C (metrics + estimates + intervals + events)
           "trial_wall_s": t_end - t0,         # D = B + C + in-trial overhead
           "event_counts": events,
           "zero_event": {k: int(events[k] == 0) for k in prob_keys},
           "interval": interval}
    out.update(est)
    out.update(extra)
    return out


def run_sobol_batch(law: AnnualLaw, n_perm: int, batch: int, seed_base: int,
                    scenario_key: str, budget: int) -> dict:
    """One 8-scramble Sobol experiment: 8 estimates, mean, t(df=7) interval."""
    from adb_g1b.samplers import metric_contributions
    from scipy import stats as _st

    t0 = time.perf_counter()
    # ---- B: QMC point generation + inverse CDF only (no metrics) ----
    trials = []
    for j in range(SOBOL_SCRAMBLES):
        scramble_seed = int(seed_base) + batch * SOBOL_SCRAMBLES + j
        _, trial = sobol_trial(law, n_perm, scramble_seed)
        trials.append(trial)
    t_samp = time.perf_counter() - t0
    # ---- C: estimates, contribution sums, and the t(df=7) batch interval ----
    t_m0 = time.perf_counter()
    estimates = []
    sums = {k: [] for k in TERM_KEYS}
    for trial in trials:
        est = estimate_trial(law, trial)
        estimates.append(est)
        cont = metric_contributions(trial.states, law)
        for k in TERM_KEYS:
            sums[k].append(float(np.sum(cont[k])))
    arr = {k: np.array([e[k] for e in estimates]) for k in TERM_KEYS}
    mean = {k: float(v.mean()) for k, v in arr.items()}
    sd = {k: float(v.std(ddof=1)) if v.size > 1 else 0.0 for k, v in arr.items()}
    tv = float(_st.t.ppf(0.975, df=SOBOL_SCRAMBLES - 1))
    ci = {k: (mean[k] - tv * sd[k] / np.sqrt(SOBOL_SCRAMBLES),
              mean[k] + tv * sd[k] / np.sqrt(SOBOL_SCRAMBLES))
          for k in TERM_KEYS}
    t_end = time.perf_counter()
    rec = {
        "method": "sobol", "n_per_scramble": int(n_perm),
        "n_scrambles": SOBOL_SCRAMBLES,
        "n_total": int(n_perm) * SOBOL_SCRAMBLES,
        "batch": int(batch), "seed_base": int(seed_base),
        "paths": int(n_perm) * SOBOL_SCRAMBLES,
        "random_draws": 3 * int(n_perm) * SOBOL_SCRAMBLES,
        "invcdf_calls": 3 * int(n_perm) * SOBOL_SCRAMBLES,
        "deterministic_assignments": 0,
        # R2R1-01 timing boundaries (perf_counter), per batch:
        "sampling_wall_s": t_samp,          # B: QMC + inverse CDF only
        "metrics_wall_s": t_end - t_m0,     # C: estimates + sums + t(df=7) interval
        "trial_wall_s": t_end - t0,         # D: complete batch (B + C + overhead)
        "scramble_estimates": {k: v.tolist() for k, v in arr.items()},
        "sum_sums_per_scramble": {k: v for k, v in sums.items()},
        "t_quantile": float(tv),
    }
    rec.update(mean)
    for k in TERM_KEYS:
        rec[f"{k}_sd"] = sd[k]
        rec[f"{k}_ci_lo"] = ci[k][0]
        rec[f"{k}_ci_hi"] = ci[k][1]
    return rec


def degree_freedom_t_interval(estimates: list[float]) -> tuple[float, float, float]:
    """t(df = k-1) interval for a set of k independent estimates (Sobol scrambles)."""
    from scipy import stats as _st
    arr = np.asarray(estimates, dtype=float)
    k = arr.size
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    tv = float(_st.t.ppf(0.975, df=k - 1))
    return mean - tv * sd / np.sqrt(k), mean + tv * sd / np.sqrt(k), tv


def run_matched_mc(out: Path, config: dict, exact_gt: dict, *,
                   scenario_keys: list[str] | None = None,
                   rep_count: int | None = None,
                   budgets: list[int] | None = None):
    """MC at N = 8*N_per_scramble: same real path count as a Sobol experiment
    (fair per-total-path comparison). Uses pre-fixed derived seeds."""
    from adb_g1b.config import _seed
    scenario_keys = scenario_keys or [s.key for s in SCENARIOS]
    rep_count = rep_count or config["reps"]
    budgets = budgets or BUDGETS
    for si, sc in enumerate(scenario_keys):
        law = law_for(sc)
        for bi, n8 in enumerate([8 * b for b in budgets]):
            rows = []
            for rep in range(rep_count):
                seed = _seed(config["master_seed"], 99, si, bi, rep)
                r = run_trial("mc", law, n8, seed)
                r["rep"] = rep
                r["scenario"] = sc
                r["matched_to_sobol"] = 1
                rows.append(r)
            _write_raw(out, "mc", sc, n8, rows)
    return out


def write_cost_fairness(out: Path, config: dict, exact_gt: dict,
                        scenario_keys: list[str] | None = None,
                        budgets: list[int] | None = None,
                        scenarios_meta: dict | None = None,
                        e2e_wall: float | None = None) -> dict:
    """Cost/fairness table with explicit timing boundaries (R2-02 / R2R1-02).

    Measured with ``time.perf_counter``. Per-method/scenario/budget rows are
    sums over the trials/batches of that cell:

    * ``preprocess_s``        : A – scenario annual-law build (once per
                                scenario; measured when the law is first built).
    * ``sampling_wall_s_total``: B – state/rng/inverse-CDF generation only.
    * ``metrics_wall_s_total`` : C – estimates + five metrics + weight/interval
                                diagnostics + event counts (+ Sobol t(df=7)).
    * ``trial_wall_s_total``  : D – complete trial time (B + C + in-trial
                                overhead). Never called "end-to-end".
    * ``e2e_wall_s``          : E – real end-to-end experiment wall time from
                                run_meta (measured over grid + aggregation,
                                see run()); the same value on every row (global).
    Exactly one row per (method, scenario, budget_nominal) — 75 unique rows for
    the full grid (R2R1-02); Sobol contributes 3 budgets x 5 scenarios = 15 rows
    and matched-cost MC another 15, with no duplicates.
    """
    from adb_g1b.stats_eval import _load_rows
    raw = out / "raw"
    table = []
    scenario_keys = scenario_keys or list(exact_gt)
    budgets = budgets or config["budgets"]
    if scenarios_meta is None:
        scenarios_meta = {}
    if e2e_wall is None:
        e2e_wall = float("nan")
    # R2R1-02: deduplicated (method, budget) construction.
    methods_budgets: list[tuple[str, int]] = []
    for n in budgets:
        for method in ("mc", "stratified", "is"):
            methods_budgets.append((method, int(n)))
        methods_budgets.append(("sobol", int(n)))      # once per budget
    for n8 in [8 * b for b in budgets]:
        methods_budgets.append(("mc", int(n8)))        # matched-path-count rows
    for method, n in methods_budgets:
        for sc in scenario_keys:
            path = raw / f"{method}--{sc}--N{n}.csv"
            if not path.is_file():
                continue
            rows = _load_rows(path)
            if not rows:
                continue
            n_trials = len(rows)
            total_paths = sum(int(float(r.get("paths", 0))) for r in rows)
            total_draws = sum(int(float(r.get("random_draws", 0))) for r in rows)
            total_invcdf = sum(int(float(r.get("invcdf_calls", 0))) for r in rows)
            total_det = sum(int(float(r.get("deterministic_assignments", 0))) for r in rows)
            samp = sum(float(r.get("sampling_wall_s", 0.0)) for r in rows)
            metr = sum(float(r.get("metrics_wall_s", 0.0)) for r in rows)
            trw = sum(float(r.get("trial_wall_s", 0.0)) for r in rows)
            pre = scenarios_meta.get(sc, {}).get("preprocess_s")
            matched = bool(rows and rows[0].get("matched_to_sobol", ""))
            rec = {
                "method": method, "scenario": sc, "budget_nominal": n,
                "budget_unit": "N_PER_SCRAMBLE * 8" if method == "sobol" else "N",
                "matched_to_sobol": int(matched),
                "total_paths": total_paths,
                "paths_per_trial": total_paths / max(n_trials, 1),
                "n_trials": n_trials,
                "random_draws_total": total_draws,
                "invcdf_calls_total": total_invcdf,
                "deterministic_assignments_total": total_det,
                # R2R1 timing boundaries (perf_counter), B/C/D per cell:
                "preprocess_s": pre,
                "sampling_wall_s_total": samp,        # B only
                "metrics_wall_s_total": metr,         # C only
                "trial_wall_s_total": trw,            # D = B+C+overhead (per trial sum)
                "e2e_wall_s": e2e_wall,               # E (global, not summed)
                "trial_wall_s_per_100k_paths": trw / max(total_paths / 1e5, 1e-12),
                # explicit compatibility aliases (R2-02: no silent redefinition):
                "alias_wall_s_total_EQ_trial_wall_s_total": trw,
                "alias_sampling_and_metric_wall_s_total_EQ_B_plus_C": samp + metr,
                "alias_wall_s_per_100k_paths_EQ_trial": trw / max(total_paths / 1e5, 1e-12),
            }
            table.append(rec)
    schema = {
        "time_measurement": "time.perf_counter, same boundaries for all methods",
        "field_definitions": {
            "preprocess_s": {"label": "A", "scope": "once per scenario",
                             "includes_preprocessing": "yes, annual-law build only",
                             "shared_across_methods": "yes, NOT re-summed per trial"},
            "sampling_wall_s_total": {"label": "B",
                                      "scope": "sum over trials/batches of cell",
                                      "includes": "state generation, RNG/QMC points, inverse CDF; IS weight construction",
                                      "includes_metrics": "no",
                                      "includes_preprocessing": "no"},
            "metrics_wall_s_total": {"label": "C", "scope": "sum over trials/batches",
                                     "includes": "estimates, five metrics, weight diagnostics, interval stats (incl. Sobol t(df=7) interval), event counts",
                                     "includes_preprocessing": "no"},
            "trial_wall_s_total": {"label": "D", "scope": "sum over trials/batches",
                                   "includes": "B + C + in-trial overhead",
                                   "includes_preprocessing": "no",
                                   "note": "complete per-trial time; NOT the experiment wall"},
            "e2e_wall_s": {"label": "E", "scope": "whole experiment run()",
                           "includes": "grid simulation, matched-cost MC, summary/coverage/rare-tail aggregation and cost-table construction",
                           "excludes": "nothing except the final run_meta.json write that embeds E itself (documented self-reference exception)",
                           "note": "measured with time.perf_counter across the entire run(); not derived from random numbers or path counts"},
        },
        "comparison_notes": (
            "matched-path-count comparison: Sobol (8*N paths) vs mc row with "
            "budget_nominal = 8*N and matched_to_sobol=1. This controls the "
            "number of simulated paths, NOT runtime; runtime is compared by "
            "measured trial_wall/e2e fields separately."),
        "row_count_contract": (
            "exactly one row per (method, scenario, budget_nominal). Full grid: "
            "per scenario, methods MC/Stratified/IS x 3 budgets (9 rows) + "
            "Sobol x 3 budgets (3) + mc-matched x 3 (3) = 15 rows; across the 5 "
            "scenarios this is 75 unique rows. Sobol rows = 15, mc-matched rows"
            " = 15. No duplicate (method, scenario, budget_nominal)."),
    }
    (out / "cost_fairness.json").write_text(
        json.dumps({"rows": table, "schema": schema}, indent=2) + "\n")
    return {"rows": table, "schema": schema}


def run(experiment: str = "full", *, out_dir: Path | None = None,
        summarize_now: bool = True,
        scenario_keys: list[str] | None = None,
        budgets_override: list[int] | None = None,
        batch_count_override: int | None = None,
        rep_count_override: int | None = None,
        with_matched_mc: bool | None = None) -> Path:
    out = out_dir or Path("results/g1b")
    scenario_keys = scenario_keys or [s.key for s in SCENARIOS]
    config = load_or_create_config(out, scenario_keys)
    exact = {s.key: compute_scenario(s) for s in SCENARIOS}
    (out / "exact_ground_truth.json").write_text(json.dumps(exact, indent=2) + "\n")

    budgets = budgets_override or ([2**12] if experiment == "pilot" else BUDGETS)
    batch_count = batch_count_override or (2 if experiment == "pilot" else SOBOL_BATCHES)
    rep_count = rep_count_override or (3 if experiment == "pilot" else config["reps"])
    do_matched = (experiment == "full") if with_matched_mc is None else with_matched_mc

    (out / "raw").mkdir(parents=True, exist_ok=True)
    started_wall = time.time()
    e2e_t0 = time.perf_counter()          # E anchor: start of every measured step
    meta = {"experiment": experiment, "started": started_wall,
            "os": os.uname().sysname, "cpu_count": os.cpu_count(),
            "numpy": np.__version__, "master_seed": config["master_seed"],
            "budgets": [int(x) for x in budgets]}
    scen_meta = {}
    for sc in scenario_keys:
        t_pre = time.perf_counter()
        law = law_for(sc)
        pre_s = time.perf_counter() - t_pre
        scen_meta[sc] = {
            "p0": law.p0,
            "layer_weights": law.layer_weights.tolist(),
            "annual_mass": float(law.probs.sum()),
            "ep_year_loss": law.ep_year_loss,
            "preprocess_s": pre_s,
        }
        for method in config["methods"]:
            for n in budgets:
                n_key = str(n)
                out_row = []
                if method == "sobol":
                    seed_base = config["seeds"][method][sc][n_key]["seed_base"]
                    n_perm = n
                    log = []
                    for b in range(batch_count):
                        r = run_sobol_batch(law, n_perm, b, seed_base, sc, n)
                        log.append(r)
                    out_row = log
                else:
                    seeds = config["seeds"][method][sc][n_key][:rep_count]
                    for rep, seed in enumerate(seeds):
                        r = run_trial(method, law, n, seed)
                        r["rep"] = rep
                        r["scenario"] = sc
                        out_row.append(r)
                _write_raw(out, method, sc, n, out_row)
    if do_matched:
        run_matched_mc(out, config, exact, scenario_keys=scenario_keys,
                       rep_count=rep_count, budgets=budgets)
    if summarize_now:
        from adb_g1b.stats_eval import summarize
        summarize(out, exact, config, scenario_keys=scenario_keys,
                  budgets=budgets, matched_mc=do_matched)
    # E is measured AFTER grid + matched MC + summary aggregation + cost-table
    # construction, so E truly includes the full run's work (R2R1-01). The only
    # operations excluded are (i) filling E's own value into the cost rows and
    # run_meta (documented self-reference exception) and (ii) the final JSON
    # serializations.
    write_cost_fairness(out, config, exact, scenario_keys=scenario_keys,
                        budgets=budgets, scenarios_meta=scen_meta,
                        e2e_wall=None)                 # placeholder (nan)
    e2e_wall = time.perf_counter() - e2e_t0
    _fill_cost_e2e(out, e2e_wall)
    meta["scenarios"] = scen_meta
    meta["finished"] = time.time()
    meta["e2e_wall_s"] = e2e_wall
    meta["peak_rss_kb"] = rss_kb()
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


def _fill_cost_e2e(out: Path, e2e: float) -> None:
    """Post-fill the measured E value into the cost rows (self-reference pass).

    This pass is excluded from E itself; it only replaces the placeholder nan
    with the already-measured end-to-end value so that cost_fairness.json and
    run_meta.json agree on E.
    """
    path = out / "cost_fairness.json"
    blob = json.loads(path.read_text())
    for r in blob["rows"]:
        r["e2e_wall_s"] = e2e
    path.write_text(json.dumps(blob, indent=2) + "\n")


def _write_raw(out: Path, method: str, sc: str, n: int, rows: list[dict]) -> None:
    path = out / "raw" / f"{method}--{sc}--N{n}.csv"
    fields = sorted({k for r in rows for k in r})
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)

        def clean(r):
            return {k: (json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list))
                        else v) for k, v in r.items()}

        w.writeheader()
        for r in rows:
            w.writerow({k: clean(r).get(k, "") for k in fields})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["pilot", "full"], default="full")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    print("OUTPUT_DIR =", run(args.experiment, out_dir=Path(args.out_dir) if args.out_dir else None))