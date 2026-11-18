"""Checkpoint A — mathematics and resource definitions for Route A (QAE).

Quantum amplitude a_j = E_p[f_j] = sum_i p_i f_j(i) over the frozen n=2
terminal distribution p (100 states, index y=(a1*5+a2)*4+rain_idx). The five
official metrics split into binary indicators and bounded continuous payoffs
in [0,1]. All values are computed in integer USD 500k units; nothing here
loads or compiles a circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np

from adb_g1a.model import Scenario
from adb_g2_s2_s1a.laws import frozen_n2_terminal, scenario_index
from adb_g2_s2_s1a.observables import n2_observable_vectors

UNIT = 500_000.0
N_TERMINAL = 100
N_BASIS_PADDED = 128          # 2^7 data qubits (100 occupied, 28 zero states)
DATA_QUBITS = 7
PAYOFF_BITS = 9               # L_units <= 320 < 512 -> 9-bit portfolio loss
STRICT_EXCEED_UNITS = 160     # L > 80 m USD strictly  <=>  L_units > 160

# two cover configurations (units per A-level / rain flag)
COVER_CONFIGS = {
    "base":     {"eq_cover": 70_000_000.0, "rain_cover": 10_000_000.0,
                 "eq_units": 35, "rain_units": 20},
    "sublimit": {"eq_cover": 60_000_000.0, "rain_cover": 20_000_000.0,
                 "eq_units": 30, "rain_units": 40},
}

INDICATOR_METRICS = ("attach", "exceed", "both")
PAYOFF_METRICS = ("EL", "stoploss")

OFFICIAL_SCENARIOS = ("adb-base", "adb-independent", "adb-common-shock")


def terminal_p(sc: Scenario) -> np.ndarray:
    """Frozen 100-state terminal probability vector (read-only)."""
    return frozen_n2_terminal(sc)


def metric_vectors(sc: Scenario) -> dict[str, np.ndarray]:
    """Five official 100-dim observable vectors for the scenario covers."""
    return n2_observable_vectors(sc.eq_cover, sc.rain_cover)


def amplitudes(sc: Scenario) -> dict[str, float]:
    """a_j = E_p[f_j] for the five official metrics (float64)."""
    p = frozen_n2_terminal(sc)
    g = n2_observable_vectors(sc.eq_cover, sc.rain_cover)
    return {k: float(np.dot(p, g[k])) for k in g}


def _by_index(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eq, ri = np.divmod(idx, 4)
    a1, a2 = np.divmod(eq, 5)
    r1 = np.take([0, 1, 0, 1], ri)
    r2 = np.take([0, 0, 1, 1], ri)
    return a1, a2, r1, r2


def country_loss_units_by_cover(
        cover: str, a1, a2, r1, r2) -> tuple[np.ndarray, np.ndarray]:
    cfg = COVER_CONFIGS[cover]
    um1 = cfg["eq_units"] * a1 + cfg["rain_units"] * r1
    um2 = cfg["eq_units"] * a2 + cfg["rain_units"] * r2
    return um1, um2


def cover_config_matrix(cover: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """100-dim vectors {EL,attach,exceed,stoploss,both} + L_units for a cover."""
    idx = np.arange(100)
    a1, a2, r1, r2 = _by_index(idx)
    um1, um2 = country_loss_units_by_cover(cover, a1, a2, r1, r2)
    L = um1 + um2
    EL = (L * UNIT) / 160_000_000.0
    attach = ((um1 > 0) & (um2 > 0)).astype(float)
    exceed = (L > STRICT_EXCEED_UNITS).astype(float)
    stoploss = np.maximum(L - STRICT_EXCEED_UNITS, 0) * UNIT / 80_000_000.0
    both = ((a1 == 4) & (r1 == 1) & (a2 == 4) & (r2 == 1)).astype(float)
    return {"EL": EL, "attach": attach, "exceed": exceed,
            "stoploss": stoploss, "both": both}, L


def exceed_good_set(sc: Scenario) -> np.ndarray:
    """Indices (0..99) of the good set for strict exceedance."""
    g = metric_vectors(sc)
    return np.where(g["exceed"] > 0.5)[0]


def zero_padded_amplitude_vector(p: np.ndarray) -> np.ndarray:
    """sqrt(p), zero-padded 100 -> 128 basis states."""
    assert len(p) == N_TERMINAL
    out = np.zeros(N_BASIS_PADDED, dtype=float)
    out[:N_TERMINAL] = np.sqrt(np.clip(p, 0.0, None))
    return out


def payoff_fixed_point(precision_bits: int = 8) -> dict[str, float]:
    """Fixed-point error analysis for the continuous payoff channels (R2-01).

    Correct statement — three *separate* error channels must not be conflated:

    1. payoff value quantization f -> z: worst-case |z-f| <= 2^{-(bits+1)}
       (``max_abs_payoff_encoding_error``). The observable bias it introduces
       is |E_p[z] - E_p[f]| <= 2^{-(bits+1)} because the ancilla-probability
       channel q = E[1_ancilla] = E[f] is *linear* in f (the actual value may
       be smaller; see ``exact_payoff_representability``).
    2. angular pixel effect of that deviation: theta(f) = arcsin(sqrt(f)) is
       NOT globally linear in f. The tight worst case on [0,1] is
       |theta(x) - theta(y)| <= arcsin(sqrt(|x-y|)) (R2-01 proof), attained
       both near f=0 and near f=1, so the worst-case single-pixel effect for a
       deviation delta is ``max_angular_effect = arcsin(sqrt(delta))``
       (strictly larger than the earlier sqrt(delta) underestimate).
    3. controlled-rotation synthesis of theta: UNKNOWN (no compiled circuit);
       it must never be inferred to be 0.
    """
    f_err = 2.0 ** (-(precision_bits + 1))
    return {
        "payoff_precision_bits": precision_bits,
        "max_abs_payoff_encoding_error": float(f_err),
        "max_observable_bias": float(f_err),
        "max_angular_effect": float(np.arcsin(np.sqrt(f_err))),
        "global_linear_angle_bound_invalid": True,
        "angular_worst_case_formula": "arcsin(sqrt(delta)), delta=2^-(bits+1)",
        "rotation_synthesis_error": "UNKNOWN_NOT_COMPILED",
        "channel_note": "3 separate channels: value quantization / angular "
                        "pixel effect / rotation synthesis (never conflated)",
    }


def exact_payoff_representability() -> dict:
    """Proof + enumeration that EL and stoploss are exactly representable in
    8-bit (and finer) fixed point under BOTH cover configs.

    Arithmetic: EL = L_units/320; stoploss = max(L_units-160,0)/160.
      base     : L_units = 35(A1+A2)+20(R1+R2)  ->  L_units == 0 (mod 5)
                 => EL grid 2^-6 (<=6 frac bits), stoploss grid 2^-5.
      sublimit : L_units = 30(A1+A2)+40(R1+R2)  ->  L_units == 0 (mod 10)
                 => EL grid 2^-5, stoploss grid 2^-4.
    Hence all values lie on the 2^-8 grid: quantization error is exactly 0.
    """
    idx = np.arange(100)
    a1, a2, r1, r2 = _by_index(idx)
    out = {}
    for cover in COVER_CONFIGS:
        cfg = COVER_CONFIGS[cover]
        um1 = cfg["eq_units"] * a1 + cfg["rain_units"] * r1
        um2 = cfg["eq_units"] * a2 + cfg["rain_units"] * r2
        L = um1 + um2
        EL = (L * UNIT) / 160_000_000.0
        SL = np.maximum(L - STRICT_EXCEED_UNITS, 0) * UNIT / 80_000_000.0
        row = {}
        for name, vals in (("EL", EL), ("stoploss", SL)):
            minbits = 9
            for bits in range(0, 9):
                grid = np.round(vals * (2 ** bits)) / (2 ** bits)
                if np.array_equal(grid, vals):
                    minbits = bits
                    break
            q_err8 = float(np.abs(vals - np.round(vals * 256) / 256).max())
            row[name] = {
                "min_fractional_bits_exact": minbits,
                "exact_in_8bit": bool(minbits <= 8),
                "quantization_error_8bit": q_err8,
                "max_abs_value": float(vals.max()),
            }
        out[cover] = row
    return {
        "theorem": ("EL and stoploss lie on the 2^-8 grid under both covers "
                    "(divisibility of L_units), hence 8-bit fixed point is "
                    "EXACT: quantization error = 0"),
        "per_cover": out,
    }


@dataclass
class LoadScheme:
    """Register taxonomy (R1-02 corrected).

    Distinguishes the terminal output data register (7 bits = 128 basis),
    the three-year annual input registers (GENERATIVE_LOAD only), the
    reversible-arithmetic workspace, the flag/phase ancillas, and unbounded
    garbage/uncomputation. total_qubits() is a conservative *architectural
    estimate* (sum of explicitly-held registers); it is NOT a proven upper
    bound because no complete reversible construction was compiled — see
    ``reliable_upper_bound`` and ``construction_note``.
    """

    name: str
    data_qubits: int                  # terminal output register (7)
    input_annual_qubits: int          # 3 x 7, or 24 fieldwise (GENERATIVE only)
    work_ancilla_estimate: int        # reversible capped-merge / comparator est.
    flag_ancilla: int
    phase_ancilla: int
    oracle_queries_per_shot_at_k1: int
    arithmetic_kind: str
    gate_count: Literal["UNKNOWN_NOT_COMPILED"] = "UNKNOWN_NOT_COMPILED"
    depth: Literal["UNKNOWN_NOT_COMPILED"] = "UNKNOWN_NOT_COMPILED"
    reliable_upper_bound: str = "NO (no complete reversible construction)"
    construction_note: str = "architectural estimate; not compiled"

    def total_qubits(self) -> int:
        return (self.data_qubits + self.input_annual_qubits +
                self.work_ancilla_estimate + self.flag_ancilla +
                self.phase_ancilla)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_qubits"] = self.total_qubits()
        return d


def load_schemes() -> dict[str, LoadScheme]:
    """Two load schemes with corrected, honestly-labelled register accounting.

    TABLE_LOAD        : A|0> = sum_x sqrt(p_T(x))|x>|0>  (7 data qubits, S0).
    GENERATIVE_LOAD   : A|0> = sum sqrt(p1 p1 p1)|y1 y2 y3>|0> — the three
        annual 7-bit y-registers are *input data* (21 qubits; 24 with
        per-country fields 3+3+1+1 per year), the terminal accumulation and
        predicate need a further 7-bit output register, reversible capped-DP
        merge arithmetic adds workspace (estimate), and garbage is only
        cleaned by explicit uncomputation (A-dagger per Grover iteration).
        The earlier "7 data + 24 work + 2 ~= 33" record conflated the S0
        21/24 *input* registers with workspace and therefore is **not** a
        reliable total bound — see reliable_upper_bound.
    """
    table = LoadScheme(
        name="TABLE_LOAD",
        data_qubits=DATA_QUBITS,
        input_annual_qubits=0,
        work_ancilla_estimate=10,
        flag_ancilla=1,
        phase_ancilla=1,
        oracle_queries_per_shot_at_k1=3,
        arithmetic_kind="QROM/index decode + 9-bit comparator",
        construction_note="QROM/DSP over precomputed 100-state table; "
                          "loader/ancilla count UNKNOWN_NOT_COMPILED (S0 A2-a)",
    )
    generative = LoadScheme(
        name="GENERATIVE_LOAD",
        data_qubits=DATA_QUBITS,
        input_annual_qubits=21,       # 3 x 7-bit y-registers; 24 fieldwise
        work_ancilla_estimate=24,     # reversible capped-merge est. (not proof)
        flag_ancilla=1,
        phase_ancilla=1,
        oracle_queries_per_shot_at_k1=3,
        arithmetic_kind="3x capped-merge (add/OR) + 9-bit comparator",
        construction_note="3-year annual input registers held; terminal "
                          "accumulator separate; garbage uncleaned unless "
                          "A-dagger uncomputed (S0 A2-b)",
    )
    return {"TABLE_LOAD": table, "GENERATIVE_LOAD": generative}


def scenario_list() -> dict[str, str]:
    return {k: v for k, v in scenario_index().items()
            if k in OFFICIAL_SCENARIOS}