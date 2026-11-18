from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

OFFICIAL_ARCHIVE_SHA256 = (
    "89d4139f809c316294b6115f115724b2af2304cba1992774e2d5086e49f7c79b"
)
OFFICIAL_ROOT_NAME = "adb-challenge-quantum-joint-loss-scenario-public-main"

CANONICAL_PATHS = (
    "README.md",
    "EVALUATION.md",
    "team_solver.py",
    "LICENSE",
    "pyproject.toml",
    "src/qjlb/__init__.py",
    "src/qjlb/model.py",
    "src/qjlb/bond.py",
    "src/qjlb/cli.py",
    "src/qjlb/worker.py",
    "src/qjlb/validation.py",
    "src/qjlb/evaluation.py",
    "src/qjlb/holdout.py",
    "data/adb_terms.json",
    "data/seismic.npz",
    "data/flood.npz",
    "data/audit.json",
    "data/LICENSE.txt",
    "data/source-manifest.json",
    "scenarios/adb-base.json",
    "scenarios/adb-common-shock.json",
    "scenarios/adb-independent.json",
    "scenarios/adb-rain-cover.json",
    "scenarios/adb-rain-stress.json",
    "scenarios/kaz-kgz-base.json",
    "scenarios/tjk-uzb-base.json",
    "SHA256SUMS",
)

DATA_SHA256 = {
    "data/adb_terms.json": (
        "50bdb628d0a67d14966d16bda35b4ab9d6b2fc473bdec42c9a664db10cc2b69b"
    ),
    "data/seismic.npz": (
        "ec8a93254dc4d9471792941022941d6ed5d76ab80e8950c21624393743a4bd25"
    ),
    "data/flood.npz": (
        "a843078285068838db0a7b83ba96aaecebceb4e69ba9774b99395a6e216bbaa1"
    ),
    "data/audit.json": (
        "11c3931d374b53a7091a88dcea5184660ffe13bf159146f5327bcaaee9690cb8"
    ),
}

SOURCE_FILE_SHA256 = {
    "README.md": "9d0a63b9cc14389c13bc3e608fb70ca17edcf6cc4926a1b472c6152a7131ac83",
    "EVALUATION.md": "80ebff4ee0f546f7625579930208ec2b11d817b7d2f83043f6abde51eedf1f6c",
    "team_solver.py": "8f70887577ecec73a4c659072ac6b9fc6aba82dabd1854b16380151b78a0837f",
    "src/qjlb/__init__.py": "f7d296be6c649c1540e436e1300ec52a981b917e8207a7550f18f6fa05296019",
    "src/qjlb/model.py": "bfae7be5c540c0a9c9cadcf6f8277b03430e3016d56ec0374be7eae858c99228",
    "src/qjlb/bond.py": "5b0f867816470fdac69188d09f1121f65dc68ba8d9240755e7fea64b6e43f9ea",
    "src/qjlb/cli.py": "1dbafa5815e61597e614366628db4e8910afb76bb20da67808a54dc085998508",
    "src/qjlb/worker.py": "d02b2d4406f2493fd9e1c30eb9d712df352b3e7cec83f3157533f312c40b19b9",
    "src/qjlb/validation.py": "b63e56fea8e78dd93dacf0718742a5a391738fac61fbc76fa0a1acaa8e26a8fb",
    "src/qjlb/evaluation.py": "f0248efa281b43aebc634ae8fc209506fc0ae9e1ec92966938486dfca664cbf4",
    "src/qjlb/holdout.py": "e9499212a36a81ef832a56930434fa667566f293e2467619d8e12ddee6fffde0",
    "data/LICENSE.txt": "98b8de4eaf4e5c24bc4a8586ef34e30370d8b081cadeed0ba608efbfaa60328d",
}

SCENARIO_SHA256 = {
    "adb-base": "91e4ca8e0ac73beb504b8a13bc80d5668c1f3005637d6e867bdc7bc31bfd1eb8",
    "adb-common-shock": "02eef9574f7193d66623dd8187d2c3224752dba0be8f12c83ef5c68eb704039a",
    "adb-independent": "7c00787e47e69a7dd62a175b08cf7005b8d8d7bd7460068ceeab366d0113dd50",
    "adb-rain-cover": "4e90443ebebb8aa7ee321f29ff7ab741fef810a3ddc69597769488387bcddaf1",
    "adb-rain-stress": "63126a131cbb9e250f3ade8fca9dd41af8930c4c64c82d9291d4f1fde608c735",
    "kaz-kgz-base": "2440dc1aed8757ad9168b690d24cb75c7a7be2d0230ba6fc7f132ef50944018b",
    "tjk-uzb-base": "27535b76fa74e0dbd80b45a64ce1cd8f376b12a4f6787ec2f0107dfe36b4cf31",
}

DRB_CASES = (
    "adb-base",
    "adb-common-shock",
    "adb-independent",
    "adb-rain-cover",
    "adb-rain-stress",
)
ELT_CASES = ("kaz-kgz-base", "tjk-uzb-base")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_missing(root: Path) -> list[str]:
    return [rel for rel in CANONICAL_PATHS if not (root / rel).is_file()]


def _manifest_audit(root: Path) -> dict:
    path = root / "SHA256SUMS"
    if not path.is_file():
        return {"status": "FAIL", "reason": "SHA256SUMS missing", "listed": 0, "matched": 0}
    rows = []
    for line in path.read_text().splitlines():
        digest, separator, rel = line.partition("  ")
        if separator and digest and rel:
            rows.append((digest.strip(), rel.strip()))
    matched = []
    missing = []
    mismatched = []
    for digest, rel in rows:
        target = root / rel
        if not target.is_file():
            missing.append(rel)
            continue
        actual = sha256_file(target)
        if actual == digest:
            matched.append(rel)
        else:
            mismatched.append(rel)
    if mismatched:
        status = "FAIL"
    elif missing == [".gitignore"]:
        status = "PARTIAL_36_OF_37_GITIGNORE_ABSENT"
    elif missing:
        status = "FAIL_MISSING_MANIFEST_FILES"
    else:
        status = "PASS"
    return {
        "status": status,
        "listed": len(rows),
        "matched": len(matched),
        "missing": missing,
        "mismatched": mismatched,
        "note": "The upstream archive omits .gitignore; do not claim 37/37.",
    }


def _symbol_audit(model_path: Path) -> dict:
    try:
        tree = ast.parse(model_path.read_text())
    except SyntaxError as error:
        return {"status": "FAIL", "error": f"syntax error: {error}"}
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    return {
        "status": "PASS" if {"ELTModel"} <= classes and {"elt_exact"} <= functions else "FAIL",
        "classes": sorted(classes),
        "functions": sorted(functions),
        "elt_module": "src/qjlb/model.py",
    }


def official_source_audit(
    source_root: Path | str | None,
    archive: Path | str | None = None,
    document: dict | None = None,
) -> dict:
    """Audit an explicit, already-extracted official source tree.

    There is intentionally no basename fallback. A missing or wrong canonical
    path fails closed; the caller may not silently substitute another source.
    """
    from .reference import load_reference

    document = document or load_reference()
    errors: list[str] = []
    if source_root is None:
        return {
            "status": "FAIL_CLOSED_SOURCE_ROOT_REQUIRED",
            "source_root_provided": False,
            "archive": {"status": "NOT_PROVIDED"},
            "minimum_missing": list(CANONICAL_PATHS),
            "minimum_missing_count": len(CANONICAL_PATHS),
            "canonical_path_count": len(CANONICAL_PATHS),
            "interface_validation": "NOT_PERFORMED",
            "note": "An explicit canonical official source root is required.",
        }
    root = Path(source_root)
    if not root.is_dir():
        return {
            "status": "FAIL_CLOSED_SOURCE_ROOT_NOT_FOUND",
            "source_root_provided": True,
            "archive": {"status": "NOT_VERIFIED"},
            "minimum_missing": list(CANONICAL_PATHS),
            "minimum_missing_count": len(CANONICAL_PATHS),
            "canonical_path_count": len(CANONICAL_PATHS),
            "interface_validation": "NOT_PERFORMED",
            "note": "The selected source root is not a directory.",
        }
    missing = _canonical_missing(root)
    if missing:
        errors.append(f"missing canonical paths: {missing}")

    archive_result = {"status": "NOT_PROVIDED", "sha256": None}
    if archive is None:
        errors.append("official archive not provided for identity verification")
        archive_result = {"status": "MISSING", "sha256": None}
    else:
        archive_path = Path(archive)
        if not archive_path.is_file():
            errors.append("official archive path is not a file")
            archive_result = {"status": "MISSING", "sha256": None}
        else:
            archive_sha = sha256_file(archive_path)
            archive_result = {
                "status": "PASS" if archive_sha == OFFICIAL_ARCHIVE_SHA256 else "FAIL",
                "sha256": archive_sha,
            }
            if archive_sha != OFFICIAL_ARCHIVE_SHA256:
                errors.append("official archive SHA256 mismatch")

    data_hashes = {}
    for rel, expected in DATA_SHA256.items():
        path = root / rel
        if not path.is_file():
            data_hashes[rel] = "NOT_LOCATED"
            errors.append(f"{rel}: missing")
            continue
        actual = sha256_file(path)
        data_hashes[rel] = actual
        if actual != expected:
            errors.append(f"{rel}: SHA256 mismatch")

    source_hashes = {}
    for rel, expected in SOURCE_FILE_SHA256.items():
        path = root / rel
        if not path.is_file():
            source_hashes[rel] = "NOT_LOCATED"
            errors.append(f"{rel}: missing")
            continue
        actual = sha256_file(path)
        source_hashes[rel] = actual
        if actual != expected:
            errors.append(f"{rel}: SHA256 mismatch")

    scenario_hashes = {}
    reference_rows = {
        row.get("scenario"): row for row in document.get("public_scenarios", [])
    }
    for name, expected in SCENARIO_SHA256.items():
        path = root / "scenarios" / f"{name}.json"
        if not path.is_file():
            scenario_hashes[name] = "NOT_LOCATED"
            errors.append(f"scenarios/{name}.json: missing")
            continue
        actual = sha256_file(path)
        scenario_hashes[name] = actual
        if actual != expected:
            errors.append(f"scenarios/{name}.json: SHA256 mismatch")
        ledger_hash = reference_rows.get(name, {}).get("config_sha256")
        if ledger_hash != expected:
            errors.append(f"{name}: frozen ledger config hash mismatch")

    manifest = _manifest_audit(root)
    if manifest["status"] not in ("PASS", "PARTIAL_36_OF_37_GITIGNORE_ABSENT"):
        errors.append(f"root SHA256SUMS: {manifest['status']}")
    symbols = _symbol_audit(root / "src/qjlb/model.py")
    if symbols["status"] != "PASS":
        errors.append("src/qjlb/model.py lacks ELTModel or elt_exact")
    if root.name != OFFICIAL_ROOT_NAME:
        errors.append(f"source root name is {root.name!r}, expected {OFFICIAL_ROOT_NAME!r}")

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "source_root_provided": True,
        "source_root_name": root.name,
        "archive": archive_result,
        "canonical_paths": list(CANONICAL_PATHS),
        "canonical_path_count": len(CANONICAL_PATHS),
        "minimum_missing": missing,
        "minimum_missing_count": len(missing),
        "data_hashes": data_hashes,
        "source_hashes": source_hashes,
        "scenario_hashes": scenario_hashes,
        "root_manifest": manifest,
        "elt_symbols": symbols,
        "interface_validation": "NOT_PERFORMED",
        "recorded_elt_path": "src/qjlb/model.py",
        "recorded_elt_symbols": ["ELTModel", "elt_exact"],
        "source_archive_sha256_expected": OFFICIAL_ARCHIVE_SHA256,
        "errors": errors,
        "note": "Exact canonical paths only; no rglob or basename fallback.",
    }
