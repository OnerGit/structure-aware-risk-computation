"""G3-A-P0-R1 — netlist hash reproduction (R1-04).

Recomputes component and aggregate SHA256 of the actual generated netlists
from the frozen inputs and the (unchanged) generators, and compares them to
the recorded values in results/g3_a_p0/netlist_hashes.json. Invoke:

    python -m adb_g3_a_p0.netlist_repro

Nothing is copied from the historical hash file into the fresh computation
(which is why the comparison is meaningful).
"""

from __future__ import annotations

import hashlib
import json
import pathlib

from .grover import GroverCircuit


def canon(net) -> str:
    """Canonical serialization used for hashing (QGate key tuples)."""
    return "\n".join(g.key().__repr__() for g in net)


def live_hashes() -> dict:
    gc = GroverCircuit("adb-base")
    parts = {"A": gc.netA, "A_dagger": gc.netAdag, "S_f": gc.netSf,
             "S_0": gc.netS0, "Q": gc.netQ}
    comp = {k: hashlib.sha256(canon(net).encode()).hexdigest()
            for k, net in parts.items()}
    agg = hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in comp.items()).encode()).hexdigest()
    return {"component_sha256": comp, "aggregate_netlist_sha256": agg,
            "gate_counts": {k: len(net) for k, net in parts.items()},
            "generator": "adb_g3_a_p0.loader/oracle/grover (unchanged) + "
                         "frozen adb-base terminal p"}


def reproduce() -> dict:
    live = live_hashes()
    rec = json.load(open("results/g3_a_p0/netlist_hashes.json"))
    comp_ok = live["component_sha256"] == rec["component_sha256"]
    agg_ok = live["aggregate_netlist_sha256"] == rec["aggregate_netlist_sha256"]
    # recorded counts were stored as strings; totals entry derived
    live_c = {k: int(v) for k, v in live["gate_counts"].items()}
    live_c["totals"] = sum(v for k, v in live_c.items() if k != "totals")
    rec_c = {k: int(v) for k, v in rec["gate_counts"].items()}
    rec_c["totals"] = sum(v for k, v in rec_c.items() if k != "totals")
    cnt_ok = live_c == rec_c
    return {"live": live, "recorded": rec,
            "component_match": comp_ok,
            "aggregate_match": agg_ok,
            "gate_counts_match": cnt_ok,
            "aggregate_sha256": live["aggregate_netlist_sha256"],
            "reproduction_pass": bool(comp_ok and agg_ok and cnt_ok)}


if __name__ == "__main__":
    r = reproduce()
    out = json.dumps(r, indent=2)
    pathlib.Path("evidence/G3_A_P0_R1_NETLIST_REPRODUCTION.txt").write_text(out + "\n")
    print(out)