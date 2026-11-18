"""G3-B-P0 — direct network contraction (exact) and mid-contraction TT
truncation. The production path never calls the frozen terminal distribution
functions; exact terminal probabilities are assembled only as diagnostics.
"""

from __future__ import annotations

import numpy as np

from adb_g1a.model import SCENARIOS, Scenario
from adb_g2_s2_s1a.observables import n2_observable_vectors

from .factors import (annual_n2, annual_n3, merge_map_n2, merge_map_n3,
                      observables_n2, observables_n3)

SCEN = {s.key: s for s in SCENARIOS}
ALL_SCEN = ["adb-base", "adb-independent", "adb-common-shock",
            "adb-rain-stress", "adb-rain-cover"]
METRICS = ("EL", "attach", "exceed", "stoploss", "both")


# ---------------------------------------------------------------------------
# exact contraction (time-major; documented factor order)
# ---------------------------------------------------------------------------
def contract_cap_corrupted(sc: Scenario, cap: int = 5, years: int = 3) -> np.ndarray:
    """Wrong-cap merge contraction on the padded 128 basis (cap-at-5 control).
    Values beyond 99 are admitted by the corrupt cap but contribute zero to any
    official observable (padded region). """
    A = annual_n2(sc)
    merge = merge_map_n2(cap)
    p = np.zeros(128)
    p[0] = 1.0
    for _t in range(years):
        p2 = np.zeros(128)
        for a in range(100):
            if p[a] == 0.0:
                continue
            np.add.at(p2, merge[a], p[a] * A)
        p = p2
    return p


def contract_n2_terminal(sc: Scenario, cap: int = 4,
                         years: int = 3) -> tuple[np.ndarray, list[dict]]:
    A = annual_n2(sc)
    merge = merge_map_n2(cap)
    p = np.zeros(100)
    p[0] = 1.0
    trace = []
    for t in range(years):
        p2 = np.zeros(100)
        for a in range(100):
            pa = p[a]
            if pa == 0.0:
                continue
            np.add.at(p2, merge[a], pa * A)
        p = p2
        trace.append({"year": t + 1, "size": 100,
                      "nnz": int((p > 0).sum()), "sum": float(p.sum()),
                      "max": float(p.max())})
    return p, trace


def contract_n3_terminal(sc: Scenario, cap: int = 4,
                         years: int = 3) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Returns (terminal1000, annual_n3_factor, trace)."""
    A = annual_n3(sc)
    merge = merge_map_n3(cap)
    p = np.zeros(1000)
    p[0] = 1.0
    trace = []
    for t in range(years):
        p2 = np.zeros(1000)
        for a in range(1000):
            pa = p[a]
            if pa == 0.0:
                continue
            np.add.at(p2, merge[a], pa * A)
        p = p2
        trace.append({"year": t + 1, "size": 1000,
                      "nnz": int((p > 0).sum()), "sum": float(p.sum()),
                      "max": float(p.max())})
    return p, A, trace


def scalar(p: np.ndarray, obs: np.ndarray) -> float:
    return float(np.dot(p, obs))


def all_25_cells(sc_keys=ALL_SCEN) -> list[dict]:
    rows = []
    for key in sc_keys:
        sc = SCEN[key]
        p, _ = contract_n2_terminal(sc)
        obs = observables_n2(sc)
        for m in METRICS:
            rows.append({"scenario": key, "metric": m, "value": scalar(p, obs[m]),
                         "cover": (35, 20) if abs(sc.eq_cover - 70e6) < 1
                                  else (30, 40)})
    return rows


def cover_mapping_checks() -> dict:
    """Both-cover observable mapping audit (NOT part of the official 25 cells)."""
    out = {}
    for key in ("adb-base", "adb-rain-cover"):
        sc = SCEN[key]
        p, _ = contract_n2_terminal(sc)
        for cover in ("base", "sublimit"):
            eq, rain = (70e6, 10e6) if cover == "base" else (60e6, 20e6)
            obs = n2_observable_vectors(eq, rain)
            out[f"{key}/{cover}"] = {m: scalar(p, obs[m]) for m in METRICS}
    return out


# ---------------------------------------------------------------------------
# TT truncation (mid-contraction location)
# ---------------------------------------------------------------------------
def _to_tensor(v: np.ndarray, nq: int = 7) -> np.ndarray:
    """Flat vector (index b = sum bit_i*2^i) -> tensor with axis i = bit i."""
    T = np.zeros((2,) * nq)
    for b in range(2 ** nq):
        T[tuple((b >> i) & 1 for i in range(nq))] = v[b]
    return T


def _from_tensor(T: np.ndarray, nq: int = 7) -> np.ndarray:
    nq = T.ndim
    v = np.zeros(2 ** nq)
    for b in range(2 ** nq):
        v[b] = T[tuple((b >> i) & 1 for i in range(nq))]
    return v


def tt_cores(v: np.ndarray, nq: int) -> list[np.ndarray]:
    """Left-canonical MPS cores of a vector (exact, no truncation).

    cores[q] has shape (alpha_{q-1}, 2, alpha_q) with alpha_{-1}=1,
    alpha_{nq-1}=1; v[i0..i_{nq-1}] = product of cores along the chain
    (axis q = bit q, i.e. v is a tensor with axis i = bit i)."""
    res = _to_tensor(v, nq).reshape((1,) + (2,) * nq)
    lr = 1
    cores = []
    for q in range(nq - 1):
        R = 2 ** (nq - q - 1)
        m = res.reshape(lr * 2, R)
        U, S, Vt = np.linalg.svd(m, full_matrices=False)
        chi = U.shape[1]
        cores.append(U.reshape(lr, 2, chi))
        if q == nq - 2:
            cores.append((S[:, None] * Vt).reshape(chi, 2, 1))
            break
        res = (S[:, None] * Vt).reshape((chi,) + (2,) * (nq - q - 1))
        lr = chi
    return cores


def tt_expand(cores: list[np.ndarray]) -> np.ndarray:
    """Contract MPS cores to the full flat vector (index b = sum bit_i*2^i)."""
    out = cores[0]                       # (1,2,chi)
    for c in cores[1:]:
        out = np.einsum('...ab,bcd->...acd', out, c)
    T = np.squeeze(np.asarray(out))
    return _from_tensor(T)


def tt_truncate_vector(v: np.ndarray, chi_max: int, nq: int = 7):
    """Left-canonical MPS of a flat vector with every bond truncated to
    chi_max. Returns (cores, kept_L2_fraction, lost_L2_fraction)."""
    res = _to_tensor(v, nq).reshape((1,) + (2,) * nq)
    lr = 1
    cores = []
    kept = 1.0
    for q in range(nq - 1):
        R2 = 2 ** (nq - q - 1)
        m = res.reshape(lr * 2, R2)
        U, S, Vt = np.linalg.svd(m, full_matrices=False)
        total = float((S ** 2).sum())
        ck = min(chi_max, int(S.size))
        kept *= float((S[:ck] ** 2).sum() / total) if total > 0 else 0.0
        cores.append((U[:, :ck] * S[:ck]).reshape(lr, 2, ck))
        if q == nq - 2:
            cores.append(Vt[:ck].reshape(ck, 2, 1))
            break
        res = Vt[:ck].reshape((ck,) + (2,) * (nq - q - 1))
        lr = ck
    return cores, kept, 1.0 - kept


def tt_truncate_vector_std(v: np.ndarray, chi_max: int, nq: int = 7):
    """STANDARD left-canonical TT-SVD truncation of a flat vector.

    Every bond carries the singular values in the *right* residual:
    core_i = U[:, :chi] (unscaled), residual = (S@Vt)[:chi] — the standard
    S*Vt propagation. Returns (cores, per_bond_info, actual_kept_l2) where
    actual_kept_l2 = ||reconstruct||^2 / ||v||^2 measured directly.
    """
    res = _to_tensor(v, nq).reshape((1,) + (2,) * nq)
    lr = 1
    cores = []
    bonds = []
    for q in range(nq - 1):
        R2 = 2 ** (nq - q - 1)
        m = res.reshape(lr * 2, R2)
        U, S, Vt = np.linalg.svd(m, full_matrices=False)
        total = float((S ** 2).sum())
        sigma_rank = int(S.size)
        ck = min(chi_max, sigma_rank)
        bonds.append({"bond": q, "rank_before": int(sigma_rank),
                      "rank_after": int(ck),
                      "singular_kept_sq_mass": float((S[:ck] ** 2).sum()),
                      "singular_total_sq_mass": total,
                      "product_fraction_this_bond":
                          float((S[:ck] ** 2).sum() / total) if total > 0 else 0.0})
        cores.append(U[:, :ck].reshape(lr, 2, ck))
        if q == nq - 2:
            # last bond absorbed into the final core (2 legs -> 1)
            cores.append((S[:ck, None] * Vt[:ck]).reshape(ck, 2, 1))
            break
        res = (S[:ck, None] * Vt[:ck]).reshape((ck,) + (2,) * (nq - q - 1))
        lr = ck
    w = np.real(tt_expand(cores))
    actual_kept_l2 = float(np.dot(w, w) / np.dot(v, v)) if float(np.dot(v, v)) > 0 else 0.0
    return cores, bonds, actual_kept_l2


def _mass_stages(p2_exact: np.ndarray, p_expand: np.ndarray, nq: int = 7):
    """L1 masses at each declared stage of one truncated merge step."""
    l1 = float(p2_exact.sum())
    pad_exact = float(p2_exact[100:].sum())
    valid_exact = float(p2_exact[:100].sum())
    p_comp = np.maximum(np.real(p_expand), 0.0)
    return {
        "L1_before_compress": l1,
        "L1_padding_domain_before_compress": pad_exact,
        "L1_valid_100_states_before_compress": valid_exact,
        "L1_after_compress_nonneg_projection": float(p_comp.sum()),
        "L1_padding_domain_after_compress": float(p_comp[100:].sum()),
        "L1_valid_100_states_after_compress": float(p_comp[:100].sum()),
        "L1_abs_diff_before_vs_after": float(abs(l1 - p_comp.sum())),
    }


def contract_truncated_n2_diagnostic(sc: Scenario, chi: int,
                                     years: int = 3) -> dict:
    """Corrected/diagnostic version of the mid-contraction TT truncation.

    Uses the STANDARD TT-SVD (S*Vt carried); records bond ranks, per-stage
    L1 masses (before compress / padding / valid-100 / after-compress
    nonneg-projection), actual L2 kept fraction, per-year measured truncation
    wall time, and per-year absolute error budget. Production input = annual
    factors only.
    """
    import time as _time
    A = annual_n2(sc)
    merge = merge_map_n2()
    p = np.zeros(128)
    p[0] = 1.0
    rec = []
    for t in range(years):
        p2 = np.zeros(128)
        for a in range(100):
            pa = p[a]
            if pa == 0.0:
                continue
            np.add.at(p2, merge[a], pa * A)
        _t0 = _time.perf_counter()
        cores, bonds, kept_l2 = tt_truncate_vector_std(p2, chi, 7)
        p_tr = np.maximum(np.real(tt_expand(cores)), 0.0)
        _dt = _time.perf_counter() - _t0
        rec.append({
            "year": t + 1,
            "bonds": bonds,
            "actual_kept_l2_fraction": float(kept_l2),
            "stages": _mass_stages(p2, p_tr),
            "truncation_seconds": float(_dt),
        })
        p = p_tr
    return {"p": p, "trace": rec, "tt_max_rank_note": "7-bit TT max bond 8"}


def contract_truncated_n2(sc: Scenario, chi: int,
                          years: int = 3) -> dict:
    """Exact direct contraction with chi-truncation of the running mode after
    each annual merge (declared mid-contraction location)."""
    A = annual_n2(sc)
    merge = merge_map_n2()
    p = np.zeros(128)
    p[0] = 1.0
    rec = []
    for t in range(years):
        p2 = np.zeros(128)
        for a in range(100):
            pa = p[a]
            if pa == 0.0:
                continue
            np.add.at(p2, merge[a], pa * A)
        padding_mass = float(p2[100:].sum())
        p2[100:] = 0.0
        cores, kept, lost = tt_truncate_vector(p2, chi, 7)
        p_tr = np.maximum(tt_expand(cores).real, 0.0)
        rec.append({"year": t + 1,
                    "nnz": int((p_tr > 1e-300).sum()),
                    "norm_L1": float(p_tr.sum()),
                    "mass_removed_padding": float(padding_mass),
                    "kept_L2_fraction": float(kept),
                    "lost_L2_fraction": float(lost)})
        p = p_tr
    return {"p": p, "trace": rec}