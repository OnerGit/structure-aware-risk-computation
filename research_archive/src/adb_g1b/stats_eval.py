"""Statistical evaluation (G1-B R1): method-correct intervals and coverage.

Three distinct statistics are reported separately:

* **single-trial CI**: within-trial, method-correct variance
    - MC:     var(contrib) / n
    - IS:     var(W·g) / n   (never Bernoulli on the weighted estimate)
    - Strat:  sum_k w_k^2 s_k^2 / n_k
    - Sobol:  8-scramble t-interval with df=7 (official)
    A probability metric with zero observed events yields a degenerate (zero
    width) normal interval; that is flagged, never reported as high precision.
* **estimate-mean interval**: mean over the R repetitions +/- t(df=R-1) SE.
* **empirical coverage**: the mean (over the R repetitions, or over the Sobol
  batches) of the indicator that the *single-trial* CI contains the exact value.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from adb_g1b.samplers import TERM_KEYS

PROB_KEYS = ("term_joint_attachment_probability",
             "term_portfolio_loss_exceedance_probability",
             "term_both_bonds_exhausted_probability")
_Z = 1.96


def _load_rows(path: Path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _get(row, key, default=None):
    v = row.get(key, "")
    if v in ("", None):
        return default
    return v


def _float(row, key) -> float:
    v = _get(row, key)
    return float(v) if v is not None else float("nan")


def _json(row, key) -> list | dict:
    v = _get(row, key)
    if v is None or v == "":
        return []
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        return json.loads(v)
    return []


def _t_quantile(df: float) -> float:
    from scipy import stats as _st
    return float(_st.t.ppf(0.975, df=df))


def per_trial_interval_fields(row, exact_metric, method: str):
    """Method-correct within-trial (est, se, CI) for the five metrics."""
    interval = _json(row, "interval")
    by_metric = {it.get("metric"): it for it in interval if isinstance(it, dict)}
    out = {}
    for k in TERM_KEYS:
        it = by_metric.get(k)
        if not it:
            out[k] = {"est": float("nan"), "se": float("nan"), "ci_lo": float("nan"),
                      "ci_hi": float("nan"), "covered": None,
                      "degenerate_ci": False, "warn": ["missing_interval"]}
            continue
        est = float(it.get("est", float("nan")))
        se = float(it.get("se", float("nan")))
        lo, hi = est - _Z * se, est + _Z * se
        warnings = list(it.get("warn") or [])
        degenerate = False
        if se == 0.0:
            degenerate = True
            warnings.append("zero_variance_normal_CI_degenerate")
        covered = None if math.isnan(lo) else bool(lo <= exact_metric[k] <= hi)
        out[k] = {"est": est, "se": se, "ci_lo": lo, "ci_hi": hi,
                  "covered": covered, "degenerate_ci": degenerate,
                  "warn": warnings}
    return out


def eval_stochastic(method, sc, n, exact_metric, rows, rep_label="rep") -> dict:
    per_metric = {}
    n_reps = len(rows)
    ci_rows = []
    for r in rows:
        ci_rows.append(per_trial_interval_fields(r, exact_metric, method))
    for k in TERM_KEYS:
        vals = np.array([_float(r, k) for r in rows])
        est = float(np.nanmean(vals))
        err = est - exact_metric[k]
        rmse = float(math.sqrt(float(np.nanmean((vals - exact_metric[k]) ** 2))))
        n_valid = int(np.count_nonzero(~np.isnan(vals)))
        # estimate-mean interval over the R repetitions
        mean, lo, hi, tv = _t_ci(vals)
        sd_of_trials = float(np.nanstd(vals, ddof=1)) if n_valid > 1 else float("nan")
        se_mean = (sd_of_trials / math.sqrt(n_valid) if sd_of_trials == sd_of_trials
                   else float("nan"))
        # empirical coverage from single-trial CIs
        cov_flags = [c[k]["covered"] for c in ci_rows
                     if c[k].get("covered") is not None]
        cov = float(np.mean(cov_flags)) if cov_flags else float("nan")
        deg = int(np.sum([1 for c in ci_rows if c[k].get("degenerate_ci")]))
        warns = set()
        for c in ci_rows:
            warns.update(c[k].get("warn") or [])
        per_metric[k] = {
            "estimate": est, "exact": exact_metric[k], "signed_error": err,
            "abs_error": abs(err),
            "rel_error": (abs(err) / exact_metric[k] if exact_metric[k] else float("nan")),
            "bias": err, "rmse": rmse,
            # R2-01: unambiguous SD vs SE_mean of the R repetition estimates
            "sd_repeated_trials": sd_of_trials,          # s = sample_std(theta_1..R)
            "se_mean": se_mean,                          # SE_mean = s / sqrt(R)
            "se_across_reps": sd_of_trials,
            "_field_notes": {
                "sd_repeated_trials": "sample standard deviation of the R "
                                      "repetition estimates (empirical spread)",
                "se_mean": "standard error of the mean of the R estimates "
                           "= sd_repeated_trials / sqrt(R)",
                "se_across_reps": "DEPRECATED_ALIAS_EQ_sd_repeated_trials; "
                                  "kept for compatibility, value == sd",
            },
            "estimate_mean_ci_lo": lo, "estimate_mean_ci_hi": hi,
            "t_quantile_df": n_valid - 1,
            "empirical_coverage": cov,
            "n_trials": n_reps,
            "degenerate_ci_trials": deg,
            "warnings": sorted(warns),
            "min_rep": float(np.nanmin(vals)), "max_rep": float(np.nanmax(vals)),
            "per_trial": [{"est": c[k]["est"], "ci_lo": c[k]["ci_lo"],
                           "ci_hi": c[k]["ci_hi"],
                           "covered": c[k]["covered"],
                           "degenerate": c[k]["degenerate_ci"],
                           "se": c[k]["se"]} for c in ci_rows],
        }
    return {"method": method, "scenario": sc, "budget": n,
            "per_metric": per_metric}


def _t_ci(values):
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    tv = _t_quantile(arr.size - 1)
    return mean, mean - tv * sd / math.sqrt(arr.size), mean + tv * sd / math.sqrt(arr.size), tv


def eval_sobol(sc, n, exact_metric, rows) -> dict:
    per_metric = {}
    for k in TERM_KEYS:
        means = np.array([_float(r, k) for r in rows])
        est = float(np.nanmean(means))
        err = est - exact_metric[k]
        rmse = float(math.sqrt(float(np.nanmean((means - exact_metric[k]) ** 2))))
        n_batches = means.size
        sd_batch = float(np.nanstd(means, ddof=1)) if n_batches > 1 else float("nan")
        se_batch_mean = (sd_batch / math.sqrt(n_batches)
                         if sd_batch == sd_batch else float("nan"))
        covered_flags = []
        widths = []
        degrad = 0
        for r in rows:
            lo, hi = _float(r, f"{k}_ci_lo"), _float(r, f"{k}_ci_hi")
            if math.isnan(lo):
                continue
            widths.append(hi - lo)
            covered_flags.append(lo <= exact_metric[k] <= hi)
            if hi - lo == 0.0:
                degrad += 1
        per_metric[k] = {
            "estimate": est, "exact": exact_metric[k], "signed_error": err,
            "abs_error": abs(err),
            "rel_error": (abs(err) / exact_metric[k] if exact_metric[k] else float("nan")),
            "bias": err, "rmse": rmse,
            # R2-01: batch-level SD vs repetition-level SE of the 20 batches
            "sd_batch_means": sd_batch,              # s over 20 batch means
            "se_mean_of_batches": se_batch_mean,     # = s / sqrt(20)
            "se_of_batch_means": sd_batch,
            "_field_notes": {
                "sd_batch_means": "sample SD of the 20 independent 8-scramble "
                                  "batch means",
                "se_mean_of_batches": "SE of the grand mean over batches "
                                      "= sd_batch_means / sqrt(n_batches)",
                "se_of_batch_means": "DEPRECATED_ALIAS_EQ_sd_batch_means; kept "
                                     "for compatibility, value == sd",
            },
            "within_batch_interval": "t(df=7) over 8 scrambles (official method)",
            "mean_ci_width": float(np.mean(widths)) if widths else float("nan"),
            "empirical_coverage": (float(np.mean(covered_flags)) if covered_flags
                                   else float("nan")),
            "degenerate_ci_batches": degrad,
            "n_batches": int(n_batches),
        }
    return {"method": "sobol", "scenario": sc, "budget": n,
            "per_batch_estimates": {k: [_float(r, k) for r in rows] for k in TERM_KEYS},
            "per_metric": per_metric}


def rare_tail_summary(out_dir: Path, exact: dict, config: dict,
                      scenario_keys: list[str]) -> dict:
    raw = out_dir / "raw"
    out = {}
    combos = []
    for method in ("mc", "stratified", "is"):
        for n in config["budgets"]:
            combos.append((method, int(n)))
    for n8 in [8 * b for b in config["budgets"]]:
        combos.append(("mc", int(n8)))
    key = "term_both_bonds_exhausted_probability"
    for sc in scenario_keys:
        for method, n in combos:
            path = raw / f"{method}--{sc}--N{n}.csv"
            if not path.is_file():
                continue
            rows = _load_rows(path)
            n_reps = len(rows)
            ests = np.array([_float(r, key) for r in rows])
            per_rep_events = np.array([
                int(json.loads(r.get("event_counts") or "{}").get(key, 0))
                for r in rows], dtype=int)
            zero_event_reps = int(np.sum(per_rep_events == 0))
            total_events = int(np.sum(per_rep_events))
            sample_var = float(np.var(ests, ddof=1)) if n_reps > 1 else float("nan")
            sd_trials = float(math.sqrt(sample_var)) if sample_var == sample_var else float("nan")
            se_mean = (sd_trials / math.sqrt(n_reps)
                       if sd_trials == sd_trials and n_reps > 0 else float("nan"))
            warnings = ["zero_event_repetitions_present"] if zero_event_reps else []
            out[f"{method}::{sc}::N{n}"] = {
                "exact": exact[sc][key],
                "estimate": float(np.mean(ests)) if n_reps else float("nan"),
                "n_reps": n_reps,
                "zero_event_reps": zero_event_reps,
                "observed_events_total": total_events,
                "mean_events_per_rep": total_events / n_reps if n_reps else float("nan"),
                "sample_variance_of_reps": sample_var,
                "sd_repeated_trials": sd_trials,
                "se_mean": se_mean,
                "se_across_reps": sd_trials,   # deprecated alias
                "_field_notes": {
                    "sd_repeated_trials": "sample SD of the R repetition "
                                          "estimates (empirical spread)",
                    "se_mean": "SE of the mean of the R estimates = SD/sqrt(R)",
                    "se_across_reps": "DEPRECATED_ALIAS_EQ_sd_repeated_trials",
                },
                "note": ("zero estimate is a data limit, not proof the true "
                         "probability is zero"),
                "warnings": warnings,
            }
    return out


def summarize(out_dir: Path, exact: dict, config: dict,
              scenario_keys: list[str] | None = None,
              budgets: list[int] | None = None,
              matched_mc: bool = True) -> dict:
    raw = out_dir / "raw"
    scenario_keys = scenario_keys or list(exact)
    budgets = budgets or config["budgets"]
    summary = {}
    coverage_rows = []
    for method in ("mc", "stratified", "is"):
        for sc in scenario_keys:
            for n in budgets:
                path = raw / f"{method}--{sc}--N{n}.csv"
                if not path.is_file():
                    continue
                rows = _load_rows(path)
                summary[f"{method}::{sc}::N{n}"] = eval_stochastic(
                    method, sc, int(n), exact[sc], rows)
                pm = summary[f"{method}::{sc}::N{n}"]["per_metric"][
                    "term_both_bonds_exhausted_probability"]
                coverage_rows.append({
                    "method": method, "scenario": sc, "budget": n,
                    "empirical_coverage": pm["empirical_coverage"],
                    "mean_width": np.nanmean([(c["ci_hi"] - c["ci_lo"])
                                              for c in pm["per_trial"]
                                              if not math.isnan(c["ci_lo"])]),
                    "degenerate_trials": pm["degenerate_ci_trials"],
                    "n_trials": pm["n_trials"],
                })
    if matched_mc:
        for sc in scenario_keys:
            for n8 in [8 * b for b in budgets]:
                path = raw / f"mc--{sc}--N{n8}.csv"
                if not path.is_file():
                    continue
                rows = _load_rows(path)
                summary[f"mc-matched::{sc}::N{n8}"] = eval_stochastic(
                    "mc-matched", sc, int(n8), exact[sc], rows)
                pm = summary[f"mc-matched::{sc}::N{n8}"]["per_metric"][
                    "term_both_bonds_exhausted_probability"]
                coverage_rows.append({
                    "method": "mc-matched", "scenario": sc, "budget": n8,
                    "empirical_coverage": pm["empirical_coverage"],
                    "mean_width": np.nanmean([(c["ci_hi"] - c["ci_lo"])
                                              for c in pm["per_trial"]
                                              if not math.isnan(c["ci_lo"])]),
                    "degenerate_trials": pm["degenerate_ci_trials"],
                    "n_trials": pm["n_trials"],
                })
    for sc in scenario_keys:
        for n in budgets:
            path = raw / f"sobol--{sc}--N{n}.csv"
            if not path.is_file():
                continue
            rows = _load_rows(path)
            summary[f"sobol::{sc}::N{n}"] = eval_sobol(sc, int(n), exact[sc], rows)
            pm = summary[f"sobol::{sc}::N{n}"]["per_metric"][
                "term_both_bonds_exhausted_probability"]
            coverage_rows.append({
                "method": "sobol", "scenario": sc, "budget": n,
                "empirical_coverage": pm["empirical_coverage"],
                "mean_width": pm["mean_ci_width"],
                "degenerate_trials": pm["degenerate_ci_batches"],
                "n_trials": pm["n_batches"],
            })

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(out_dir / "coverage_report.csv", coverage_rows)
    (out_dir / "rare_tail.json").write_text(
        json.dumps(rare_tail_summary(out_dir, exact, config, scenario_keys),
                   indent=2) + "\n")
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)