from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT / "src"))

from adb_v2.reference import load_reference, validate_reference_ledger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()
    document = json.loads(Path(args.reference).read_text()) if args.reference else load_reference()
    result = validate_reference_ledger(document)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
