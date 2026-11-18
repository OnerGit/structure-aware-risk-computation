from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy
import qiskit
import scipy


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_sha256(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def environment(candidate_root: Path, official_source: str = "NOT_LOCATED") -> dict:
    return {
        "python": sys.version.split()[0],
        "python_full": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "qiskit": qiskit.__version__,
        "simulator": "qiskit.quantum_info.Statevector (ideal logical statevector; no QPU)",
        "compilation": "no transpile, no coupling map, no backend",
        "qpu": "NOT_RUN",
        "official_source": official_source,
        "dense_8192_unitary_built": False,
        "candidate_root": candidate_root.name,
    }
