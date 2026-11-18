"""The four official samplers (Brief step 9) and path metric contributions.

Sampling contract (see docs/G1_B_METHOD_AND_STATISTICAL_CONTRACT.md):

* Plain MC: ``n_paths`` x 3 iid uniform draws, annual inverse CDF.
* Scrambled Sobol: d=3 scrambled Sobol, N_PER_SCRAMBLE power of two, 8
  independent scrambles averaged; Student-t (df=7) interval computed by the
  caller from the 8 estimates.
* Stratified: layers = first payout year 1/2/3, ``w_k = p0^(k-1)(1-p0)``;
  pre-fixed proportional allocation; years before first payout are zero-payout
  states, the first payout year is drawn from the conditional positive law,
  later years from the unconditional law; final estimate = layer-weighted
  mean of the conditional layer means.
* Importance Sampling: ``q(s) = 0.5 p(s) + 0.5 tilt(s)`` with
  ``tilt(s) = p(s) loss(s)/E_p[loss]`` (scenario's own covers); paths iid from
  q; unnormalized (unbiased in E_q) weights ``W = prod p/q``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.random import default_rng

from .annual_law import AnnualLaw, default_layers


@dataclass(frozen=True)
class Trial:
    states: np.ndarray                       # (n, 3) annual state indices
    weights: np.ndarray = field(default=None)   # IS path weights
    component: np.ndarray = field(default=None)  # layer id per path (stratified)

    # honest cost counters (R1-02): deterministic zero-year assignments are
    # NOT random sampling and are counted separately
    random_draws: int = field(default=None)      # number of uniform/QMC variates used
    invcdf_calls: int = field(default=None)      # number of inverse-CDF lookups
    deterministic_assignments: int = field(default=0)

    n_paths: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "n_paths", int(self.states.shape[0]))
        if self.random_draws is None:
            object.__setattr__(self, "random_draws", 3 * self.n_paths)
        if self.invcdf_calls is None:
            object.__setattr__(self, "invcdf_calls", 3 * self.n_paths)


def mc_trial(law: AnnualLaw, n: int, rng: np.random.Generator) -> Trial:
    u = rng.random((n, 3))
    return Trial(states=law.state_from_u(u))


def sobol_trial(law: AnnualLaw, n: int, scramble_seed: int):
    """One scramble = ``n`` d=3 scrambled-Sobol paths (n a power of two)."""
    if n < 1 or (n & (n - 1)):
        raise ValueError(f"Sobol budget must be a positive power of two, got {n}")
    from scipy.stats.qmc import Sobol
    sobol = Sobol(d=3, scramble=True, seed=int(scramble_seed))
    pts = sobol.random(n).astype(np.float64)
    states = law.state_from_u(np.clip(pts, 0.0, 1.0 - np.finfo(np.float64).eps))
    return n, Trial(states=states)


def stratified_trial(law: AnnualLaw, n: int, rng: np.random.Generator,
                     n_layer: np.ndarray | None = None) -> Trial:
    n_layer = default_layers(law, n) if n_layer is None else n_layer
    assert int(n_layer.sum()) == n
    zero_states = np.flatnonzero(law.zero_mask)
    assert zero_states.size == 1, "annual model has exactly one zero-payout state"
    zero_state = int(zero_states[0])

    blocks = []
    comps = []
    draws = 0
    det = 0
    for k, m in enumerate(n_layer, start=1):
        # layer H_k: k-1 deterministic zero-payout years + one random positive
        # first-payout year + (3-k) random unconditional years -> 4-k random
        draws += m * (4 - k)
        det += m * (k - 1)
        u_first = rng.random((m,))
        first = law.pos_state_from_u(u_first)
        if k == 1:
            year0 = first
            year1 = law.state_from_u(rng.random((m,)))
            year2 = law.state_from_u(rng.random((m,)))
            st = np.stack([year0, year1, year2], axis=1)
        elif k == 2:
            year0 = np.full(m, zero_state, dtype=np.int64)
            year1 = first
            year2 = law.state_from_u(rng.random((m,)))
            st = np.stack([year0, year1, year2], axis=1)
        else:
            year0 = np.full(m, zero_state, dtype=np.int64)
            year1 = np.full(m, zero_state, dtype=np.int64)
            year2 = first
            st = np.stack([year0, year1, year2], axis=1)
        blocks.append(st)
        comps.append(np.full(m, k - 1, dtype=np.int64))
    states = np.concatenate(blocks, axis=0)
    component = np.concatenate(comps, axis=0)
    trial = Trial(states=states, component=component,
                  random_draws=draws, invcdf_calls=draws,
                  deterministic_assignments=det)
    return trial


def is_trial(law: AnnualLaw, n: int, rng: np.random.Generator) -> Trial:
    u = rng.random((n, 3))
    states = law.state_from_q(u)
    pv = law.probs[states]
    qv = law.q[states]
    log_w = np.sum(np.log(pv) - np.log(qv), axis=1)
    w = np.exp(log_w)
    return Trial(states=states, weights=w)


# ---- terminal states and metric contributions -----------------------------

TERM_KEYS = ("term_expected_principal_loss_usd",
             "term_joint_attachment_probability",
             "term_portfolio_loss_exceedance_probability",
             "term_portfolio_stop_loss_usd",
             "term_both_bonds_exhausted_probability")


def terminal_from_states(states: np.ndarray, law: AnnualLaw):
    s0, s1, s2 = states[:, 0], states[:, 1], states[:, 2]
    A = np.minimum(law.a[s0] + law.a[s1] + law.a[s2], 4)
    B = np.minimum(law.b[s0] + law.b[s1] + law.b[s2], 4)
    R1 = law.r1[s0] | law.r1[s1] | law.r1[s2]
    R2 = law.r2[s0] | law.r2[s1] | law.r2[s2]
    return A, B, R1.astype(np.int64), R2.astype(np.int64)


def metric_contributions(states: np.ndarray, law: AnnualLaw) -> dict[str, np.ndarray]:
    """Per-path contribution arrays for the five term metrics (raw probability/USD)."""
    from adb_g1a.model import PRINCIPAL
    A, B, R1, R2 = terminal_from_states(states, law)
    l1 = law.scenario.eq_cover * (A / 4.0) + law.scenario.rain_cover * R1
    l2 = law.scenario.eq_cover * (B / 4.0) + law.scenario.rain_cover * R2
    l = l1 + l2
    return {
        "term_expected_principal_loss_usd": l.astype(np.float64),
        "term_joint_attachment_probability": ((l1 > 0.0) & (l2 > 0.0)).astype(np.float64),
        "term_portfolio_loss_exceedance_probability": (l > PRINCIPAL).astype(np.float64),
        "term_portfolio_stop_loss_usd": np.maximum(l - PRINCIPAL, 0.0).astype(np.float64),
        "term_both_bonds_exhausted_probability": ((A == 4) & (R1 == 1) &
                                                  (B == 4) & (R2 == 1)).astype(np.float64),
    }


def estimate_trial(law: AnnualLaw, trial: Trial) -> dict:
    """Point estimate of the five metrics for one trial."""
    cont = metric_contributions(trial.states, law)
    out = {}
    if trial.weights is None:
        for k in TERM_KEYS:
            out[k] = float(cont[k].mean())
    else:
        for k in TERM_KEYS:
            out[k] = float(np.sum(trial.weights * cont[k]) / trial.n_paths)
    return out


def stratified_layer_means(law: AnnualLaw, trial: Trial) -> tuple[np.ndarray, dict, np.ndarray]:
    """Per-layer conditional means (raw) plus the weighted overall estimate."""
    comp = trial.component
    cont = metric_contributions(trial.states, law)
    layer_means = {k: np.array([float(cont[k][comp == j].mean())
                                for j in range(3)]) for k in TERM_KEYS}
    est = {}
    for k in TERM_KEYS:
        est[k] = float(np.sum(law.layer_weights * layer_means[k]))
    return law.layer_weights, layer_means, est


def is_weight_stats(trial: Trial) -> dict:
    w = trial.weights
    n = w.size
    ess = float((w.sum() ** 2) / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0
    return {
        "weight_mean": float(w.mean()),
        "weight_min": float(w.min()),
        "weight_q25": float(np.quantile(w, 0.25)),
        "weight_q50": float(np.median(w)),
        "weight_q75": float(np.quantile(w, 0.75)),
        "weight_q99": float(np.quantile(w, 0.99)),
        "weight_max": float(w.max()),
        "weight_concentration": float(w.max() / w.mean()) if w.mean() > 0 else float("nan"),
        "ess": ess,
        "ess_per_path": float(ess / n),
    }


# ---- method-correct interval sufficient statistics (R1-01) ------------------
#
# The sampling variance of theta_hat is computed from the *contributions*:
#   MC         :  Var(mean_contrib)          = var(contrib) / n
#   IS         :  theta_hat = mean(W·g); Var  = var(W·g) / n   (NOT Bernoulli on
#                the final weighted estimate)
#   Stratified :  theta_hat = sum_k w_k mean_k; Var = sum_k w_k^2 s_k^2 / n_k
# All five metrics get the correct estimator; indicators are just special cases.

CONTRIB_COLUMNS = TERM_KEYS


def trial_contribution_matrix(law: AnnualLaw, trial: Trial) -> np.ndarray:
    cont = metric_contributions(trial.states, law)
    return np.column_stack([cont[k] for k in CONTRIB_COLUMNS])


def interval_sufficient_stats(law: AnnualLaw, trial: Trial) -> dict:
    """Per-metric (est, se) with the method-correct variance estimator and flags.

    Returns a list of dicts keyed as ``interval`` for raw JSON storage.
    """
    M = trial_contribution_matrix(law, trial)
    n = int(M.shape[0])
    out = []
    if trial.component is not None:
        w = np.asarray(law.layer_weights, dtype=float)
        comp = trial.component
        n_v = 3
        layer_n = np.array([np.count_nonzero(comp == j) for j in range(n_v)])
        layer_mean = np.zeros((n_v, len(CONTRIB_COLUMNS)))
        layer_var = np.zeros((n_v, len(CONTRIB_COLUMNS)))
        for j in range(n_v):
            sel = M[comp == j]
            layer_mean[j] = sel.mean(axis=0) if sel.shape[0] else np.zeros(len(CONTRIB_COLUMNS))
            ddof = 1 if sel.shape[0] > 1 else 0
            layer_var[j] = sel.var(axis=0, ddof=ddof) if sel.shape[0] else np.zeros(len(CONTRIB_COLUMNS))
        est = w @ layer_mean
        var_est = np.sum((w[:, None] ** 2) * layer_var / np.maximum(layer_n[:, None], 1),
                         axis=0)
        for m, key in enumerate(CONTRIB_COLUMNS):
            out.append({
                "metric": key,
                "est": float(est[m]),
                "se": float(np.sqrt(max(var_est[m], 0.0))),
                "layer_n": [int(x) for x in layer_n],
                "layer_mean": [float(x) for x in layer_mean[:, m]],
                "layer_var": [float(x) for x in layer_var[:, m]],
                "warn": [] if n > 1 else ["layer_n1"],
            })
        return out
    if trial.weights is not None:
        Y = trial.weights[:, None] * M
        est = Y.mean(axis=0)
        ddof = 1 if n > 1 else 0
        vmean = Y.var(axis=0, ddof=ddof) / n
    else:
        est = M.mean(axis=0)
        ddof = 1 if n > 1 else 0
        vmean = M.var(axis=0, ddof=ddof) / n
    for m, key in enumerate(CONTRIB_COLUMNS):
        se = float(np.sqrt(max(vmean[m], 0.0)))
        warn = []
        if se == 0.0 and key in (
                "term_joint_attachment_probability",
                "term_portfolio_loss_exceedance_probability",
                "term_both_bonds_exhausted_probability"):
            warn.append("zero_variance_normal_CI_degenerate")
        if key.endswith("_probability") and float(est[m]) == 0.0:
            warn.append("zero_event_prct_estimate_0")
        out.append({"metric": key, "est": float(est[m]), "se": se,
                    "warn": warn})
    return out