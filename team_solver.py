#!/usr/bin/env python3
"""Official request/response adapter for the public seven-case interface.

This is a classical finite-state exact baseline. It uses the admitted official
qjlb model and exact solver from the supplied source root. It does not copy
fixed reference answers and it does not label a quantum execution.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path


def bind_official_source() -> Path | None:
    source = os.environ.get("QJLB_SOURCE_ROOT")
    if source:
        requested = Path(source).resolve()
        candidates = (requested, requested / "src")
        chosen = next((p for p in candidates if (p / "qjlb" / "model.py").is_file()), None)
        if chosen is None:
            raise RuntimeError(f"QJLB_SOURCE_ROOT has no src/qjlb/model.py: {requested.name}")
        sys.path.insert(0, str(chosen))
        import qjlb

        loaded = Path(qjlb.__file__).resolve()
        if chosen not in loaded.parents:
            raise RuntimeError("qjlb was imported from a different source root")
        return chosen
    try:
        import qjlb
    except ImportError as error:
        raise RuntimeError(
            "qjlb is unavailable; set QJLB_SOURCE_ROOT to the admitted official src directory"
        ) from error
    return None


def require_keys(request: dict) -> None:
    required = {
        "schema_version",
        "run_id",
        "problem_id",
        "track",
        "config_file",
        "seed",
        "samples",
        "targets",
    }
    missing = sorted(required - set(request))
    if missing:
        raise ValueError(f"request missing keys: {missing}")


def validate_request(request: dict, root: Path) -> None:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    require_keys(request)
    if request["schema_version"] != 1:
        raise ValueError("request schema_version must be 1")
    if not isinstance(request["run_id"], str) or not request["run_id"].strip():
        raise ValueError("run_id must be a nonempty string")
    if not isinstance(request["problem_id"], str) or not re.fullmatch(
        "[0-9a-f]{64}", request["problem_id"]
    ):
        raise ValueError("problem_id must be a lowercase SHA-256 hex string")
    if request["track"] not in ("adb_drb", "worldbank_elt"):
        raise ValueError("track must be adb_drb or worldbank_elt")
    if type(request["seed"]) is not int or request["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(request["samples"]) is not int or request["samples"] < 0:
        raise ValueError("samples must be a nonnegative integer")
    if not isinstance(request["targets"], list) or any(
        not isinstance(name, str) for name in request["targets"]
    ):
        raise ValueError("targets must be a list of strings")
    if len(request["targets"]) != 5 or len(set(request["targets"])) != 5:
        raise ValueError("targets must contain five unique names")
    config_name = request["config_file"]
    if not isinstance(config_name, str) or not config_name:
        raise ValueError("config_file must be a nonempty relative path")
    config = Path(config_name)
    if config.is_absolute() or ".." in config.parts:
        raise ValueError("config_file must stay inside the invocation directory")
    resolved = (root / config).resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError("config_file escapes the invocation directory")
    if not resolved.is_file():
        raise ValueError("config_file does not exist")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    bind_official_source()
    from qjlb.cli import load_model, methods_for, write_json
    from qjlb.validation import check_submission, make_submission, read_json

    work = Path.cwd().resolve()
    request = read_json(args.request)
    validate_request(request, work)
    model, preparation_seconds = load_model(work / request["config_file"])
    if model.fingerprint != request["problem_id"]:
        raise ValueError("request problem_id does not match the admitted model and data")
    if model.config["track"] != request["track"]:
        raise ValueError("request track does not match the admitted model")
    if list(model.metric_names) != request["targets"]:
        raise ValueError("request targets do not match the admitted metric names")

    result = methods_for(model)["exact"](model, request["samples"], request["seed"])
    elapsed_seconds = time.perf_counter() - started
    model_evaluations = result["resources"].get("model_evaluations")
    if type(model_evaluations) is not int or model_evaluations < 0:
        raise ValueError("exact solver did not report a nonnegative integer model_evaluations")

    submission = make_submission(
        model,
        result,
        "official_exact_classical_dp",
        elapsed_seconds,
        preparation_seconds,
    )
    submission["execution"] = "classical_cpu"
    submission["resources"]["model_evaluations"] = model_evaluations
    submission.update(
        run_id=request["run_id"],
        seed=request["seed"],
        track=request["track"],
        targets=list(model.metric_names),
    )
    check_submission(submission, model.fingerprint, model.metric_names)
    write_json(args.output, submission)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
