"""Fixed, machine-readable experiment configuration and seed derivation.

All seeds are derived from a fixed master integer and recorded explicitly in
``config.json`` on first run; reruns reuse the same file so every trial is fully
deterministic. Allocation rule for stratification is fixed in
``docs/G1_B_METHOD_AND_STATISTICAL_CONTRACT.md`` (proportional to layer weight).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.random import SeedSequence

MASTER_SEED = 20261105
BUDGETS = [2**12, 2**14, 2**16]           # N nominal paths per trial
SOBOL_SCRAMBLES = 8                        # official number of independent scrambles
REPS_ESTIMATES = 20                        # independent repetitions (MC/Strat/IS)
SOBOL_BATCHES = 20                         # independent 8-scramble batches (coverage)
SOBOL_SCRAMBLE_SEED_BASE = 1000001         # scramble seed = base + batch*s + j

METHOD_ORDER = ["mc", "sobol", "stratified", "is"]


def _seed(master: int, *parts: int) -> int:
    return int(SeedSequence([master, *parts]).generate_state(1)[0])


def build_config(scenario_keys: list[str], master: int = MASTER_SEED) -> dict:
    """Explicit seed list for every (method, scenario, budget, rep/batch)."""
    seeds = {}
    for mi, method in enumerate(METHOD_ORDER):
        for si, sc in enumerate(scenario_keys):
            for bi, n in enumerate(BUDGETS):
                if method == "sobol":
                    seeds.setdefault(method, {})
                    seeds[method].setdefault(sc, {})
                    seeds[method][sc][str(n)] = {
                        "n_scrambles": SOBOL_SCRAMBLES,
                        "seed_base": _seed(master, mi, si, bi, 90),
                    }
                else:
                    seeds.setdefault(method, {})
                    seeds[method].setdefault(sc, {})
                    seeds[method][sc][str(n)] = [
                        _seed(master, mi, si, bi, rep) for rep in range(REPS_ESTIMATES)
                    ]
    return {
        "master_seed": master,
        "scenarios": scenario_keys,
        "budgets": [int(x) for x in BUDGETS],
        "methods": METHOD_ORDER,
        "reps": REPS_ESTIMATES,
        "sobol_scrambles": SOBOL_SCRAMBLES,
        "sobol_batches": SOBOL_BATCHES,
        "allocation_rule": "proportional_to_layer_weight_min1_remainder_by_fraction",
        "lane": "explicit_seed_list",
        "seeds": seeds,
    }


def load_or_create_config(out_dir: Path, scenario_keys: list[str]) -> dict:
    cfg_path = out_dir / "config.json"
    if cfg_path.is_file():
        return json.loads(cfg_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(scenario_keys)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg