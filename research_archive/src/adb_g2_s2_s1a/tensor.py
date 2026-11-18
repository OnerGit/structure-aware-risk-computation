"""Checkpoint C machinery (S1A-R1): TT-SVD with internal bond ranks, singular
spectra, numeric ranks and the two separate contraction strategies
(P_TT_THEN_CONTRACT vs W_TT_THEN_SUM). numpy.linalg.svd only, no new deps.
"""

from __future__ import annotations

import numpy as np


def to_tt_svd(tensor: np.ndarray, chi: int):
    """TT-SVD of a 3-mode tensor; returns (cores list, internal ranks (r1,r2),
    spectra (s1, s2 truncated to kept ranks), stored_elements).
    The last mode size (n3=10) is a *physical* dimension, not a bond rank.
    """
    A = np.asarray(tensor, dtype=float)
    n1, n2, n3 = A.shape
    M = A.reshape(n1, n2 * n3)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r1 = int(min(S.size, int(chi)))
    U1 = U[:, :r1]
    G1 = U1.reshape(n1, r1)
    B = (S[:r1, None] * Vt[:r1, :]).reshape(r1 * n2, n3)
    U2, S2, V2t = np.linalg.svd(B, full_matrices=False)
    r2 = int(min(S2.size, int(chi)))
    G2 = U2[:, :r2].reshape(r1, n2, r2)
    G3 = (S2[:r2, None] * V2t[:r2, :])
    cores = [G1, G2, G3]
    stored = int(sum(c.size for c in cores))
    return cores, (r1, r2), (S[:r1].copy(), S2[:r2].copy()), stored


def tt_reconstruct(cores: list[np.ndarray]) -> np.ndarray:
    G1, G2, G3 = cores
    n1, r1 = G1.shape
    r1b, n2, r2 = G2.shape
    r2b, n3 = G3.shape
    assert r1 == r1b and r2 == r2b
    tmp = np.einsum("ir, rjk -> ijk", G1, G2)
    return np.einsum("ijk, kl -> ijl", tmp, G3)


def frobenius_residual(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def numeric_rank(svals: np.ndarray, tol: float) -> int:
    if svals.size == 0:
        return 0
    return int(np.count_nonzero(svals > tol * max(float(np.max(svals)), 1e-300)))


def mode1_spectrum(tensor3: np.ndarray) -> np.ndarray:
    """Singular values of the mode-1 (country-1) unfolding (10 x 100)."""
    A = np.asarray(tensor3, dtype=float)
    return np.linalg.svd(A.reshape(10, 100), compute_uv=False)


def P_TT_THEN_CONTRACT(p_term: np.ndarray, g: dict[str, np.ndarray],
                       chi: int) -> dict:
    """Strategy 1: TT-SVD on P=p_term (10x10x10), then scalar = P_approx*g."""
    tensor = p_term.reshape(10, 10, 10)
    cores, ranks, spectra, stored = to_tt_svd(tensor, chi)
    rec = tt_reconstruct(cores)
    p_approx = rec.ravel()
    exact = {k: float(np.sum(p_term * gg.ravel())) for k, gg in g.items()}
    tt = {k: float(np.sum(p_approx * gg.ravel())) for k, gg in g.items()}
    return {
        "strategy": "P_TT_THEN_CONTRACT", "chi": int(chi),
        "internal_ranks": list(ranks), "stored_elements": stored,
        "p_1_spec_kept": [float(x) for x in spectra[0]],
        "p_2_spec_kept": [float(x) for x in spectra[1]],
        "p_frobenius_residual": frobenius_residual(tensor, rec),
        "per_metric": {k: _row(exact[k], tt[k]) for k in exact},
    }


def W_TT_THEN_SUM(p_term: np.ndarray, g: dict[str, np.ndarray],
                  chi: int) -> dict:
    """Strategy 2: per-observable W_j = p_term*g_j (10x10x10), scalar = sum(TT(W_j))."""
    out = {"strategy": "W_TT_THEN_SUM", "chi": int(chi)}
    per = {}
    for k, gg in g.items():
        W = (p_term * gg.ravel()).reshape(10, 10, 10)
        cores, ranks, spectra, stored = to_tt_svd(W, chi)
        rec = tt_reconstruct(cores)
        w1 = mode1_spectrum(W)
        exact = float(np.sum(W))
        tt = float(np.sum(rec))
        per[k] = {
            "scalar_exact": exact,
            "scalar_tt": tt,
            "abs_err": abs(tt - exact),
            "rel_err": (abs(tt - exact) / abs(exact) if exact != 0.0 else None),
            "internal_ranks": list(ranks),
            "stored_elements": int(stored),
            "w_frobenius_residual": frobenius_residual(W, rec),
            "mode1_spec": [float(x) for x in w1],
            "mode1_rank_tol1e10": numeric_rank(w1, 1e-10),
            "mode1_rank_tol1e8": numeric_rank(w1, 1e-8),
        }
    out["per_metric"] = per
    return out


def _row(exact, approx) -> dict:
    rel = abs(approx - exact) / abs(exact) if exact != 0.0 else None
    return {"scalar_exact": float(exact), "scalar_tt": float(approx),
            "abs_err": float(abs(approx - exact)), "rel_err": rel}


def minimal_chi_threshold(rows: list[dict], tol: float) -> int:
    """Smallest tested chi with per-metric abs_err <= tol for a fixed strategy."""
    for r in rows:
        worst = max((m["abs_err"] for m in r["per_metric"].values()))
        if worst <= tol:
            return int(r["chi"])
    raise ValueError("no chi satisfies threshold")


def minimal_chi_relative(rows: list[dict], target: float) -> int:
    """Smallest tested chi with all nonzero targets' rel_err <= target; a
    target with exact==0 or rel_err None is reported, not silently dropped."""
    for r in rows:
        ok = True
        for m in r["per_metric"].values():
            if m["rel_err"] is None:
                ok = ok and m["abs_err"] == 0.0
            else:
                ok = ok and m["rel_err"] <= target
        if ok:
            return int(r["chi"])
    raise ValueError("no chi satisfies relative threshold")


def minimal_chi_per_metric(rows: list[dict], tol: float) -> dict[str, int]:
    """R2-01: smallest tested chi per metric with abs_err <= tol (metric-keyed)."""
    metrics = list(rows[0]["per_metric"])
    out = {}
    for k in metrics:
        got = None
        for r in rows:
            if r["per_metric"][k]["abs_err"] <= tol:
                got = int(r["chi"])
                break
        if got is None:
            raise ValueError(f"no chi satisfies abs threshold for {k}")
        out[k] = got
    return out


def minimal_chi_per_metric_rel(rows: list[dict], target: float) -> dict[str, int]:
    """R2-01: smallest tested chi per metric with rel_err <= target
    (exact==0 handled via abs). Metric-keyed."""
    metrics = list(rows[0]["per_metric"])
    out = {}
    for k in metrics:
        got = None
        for r in rows:
            m = r["per_metric"][k]
            ok = (m["rel_err"] is None and m["abs_err"] == 0.0) or \
                 (m["rel_err"] is not None and m["rel_err"] <= target)
            if ok:
                got = int(r["chi"])
                break
        if got is None:
            raise ValueError(f"no chi satisfies relative threshold for {k}")
        out[k] = got
    return out


# --- isolated SVD vs reconstruct+contract separation (R2-02) ----------------
def p_row_from_cores(chi, cores, ranks, spectra_kept, stored,
                     P: np.ndarray, g: dict[str, np.ndarray]) -> dict:
    """P strategy row built from precomputed cores (no SVD inside)."""
    rec = tt_reconstruct(cores)
    pa = rec.ravel()
    exact = {k: float(np.sum(P.ravel() * gg.ravel())) for k, gg in g.items()}
    tt = {k: float(np.sum(pa * gg.ravel())) for k, gg in g.items()}
    return {
        "strategy": "P_TT_THEN_CONTRACT", "chi": int(chi),
        "internal_ranks": list(ranks),
        "stored_elements": int(stored),
        "p_1_spec_kept": [float(x) for x in spectra_kept[0]],
        "p_2_spec_kept": [float(x) for x in spectra_kept[1]],
        "p_frobenius_residual": frobenius_residual(P.reshape(10,10,10), rec),
        "per_metric": {k: _row(exact[k], tt[k]) for k in exact},
    }


def w_mode1_diagnostics(W: np.ndarray) -> dict:
    """RAW numpy.linalg.svd diagnostics for W_j (mode-1 spectrum + ranks).
    Kept outside any pure reconstruction/contraction timed region."""
    w1 = mode1_spectrum(W)
    return {"mode1_spec": [float(x) for x in w1],
            "mode1_rank_tol1e10": numeric_rank(w1, 1e-10),
            "mode1_rank_tol1e8": numeric_rank(w1, 1e-8)}


def w_row_from_cores(k, chi, cores, ranks, spectra_kept, stored,
                     W: np.ndarray, diag: dict) -> dict:
    """W_j row built from precomputed cores + precomputed mode-1 diagnostics.

    Pure reconstruction + contraction only: no to_tt_svd and no
    numpy.linalg.svd are executed inside this function (R3-T02).
    """
    rec = tt_reconstruct(cores)
    exact = float(np.sum(W))
    tt = float(np.sum(rec))
    return {
        "strategy": "W_TT_THEN_SUM", "chi": int(chi),
        "metric": k,
        "scalar_exact": exact, "scalar_tt": tt,
        "abs_err": abs(tt - exact),
        "rel_err": (abs(tt - exact) / abs(exact) if exact != 0.0 else None),
        "internal_ranks": list(ranks),
        "stored_elements": int(stored),
        "w_frobenius_residual": frobenius_residual(W, rec),
        "mode1_spec": list(diag["mode1_spec"]),
        "mode1_rank_tol1e10": int(diag["mode1_rank_tol1e10"]),
        "mode1_rank_tol1e8": int(diag["mode1_rank_tol1e8"]),
    }
