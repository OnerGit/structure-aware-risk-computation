# Frozen Scenario and Reference Metadata

This directory contains **metadata and frozen reference values**, not the
official source repository or World Bank NPZ data.

- `official_seven_cases_reference.json`: frozen seven-case schema and
  reference ledger used for identity checks.
- `official_seven_cases_metrics.csv`: flattened 35-row metric ledger.
- `p1b_annual_reference_amended.json`: frozen P1-B annual `adb-base` reference
  with corrected C-order rain indexing and unchanged probabilities.

The project executes the official request/response interface using an explicitly
supplied, hash-verified official source root. The official repository and NPZ
data are not redistributed here; actual execution evidence is in
`results/seven_cases/`.
