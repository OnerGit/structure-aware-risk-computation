# Third-Party Notices

No third-party source code, official benchmark repository, ADB PDF, dashboard
export or World Bank NPZ data file is vendored in this repository. The seven
public scenario configuration values reproduced under
`results/seven_cases/*/config.json` are execution metadata, not a source
distribution; their Apache-2.0 attribution and licence text are provided
below.

| Component | Use | Upstream licence |
|---|---|---|
| Python 3.12 | Interpreter | PSF License |
| NumPy | Numerical arrays | BSD-3-Clause |
| SciPy | Matrix exponential and numerical routines | BSD-3-Clause |
| Qiskit SDK 2.6.2 | Ideal circuit construction and statevector simulation | Apache-2.0 |
| pytest | Test runner (development only) | MIT |

Observed frozen environment: Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1,
Qiskit 2.6.2, pytest 8.4.2.

## ADB benchmark repository

The official benchmark repository is not included. Its root `LICENSE` is
Apache-2.0 according to the frozen source audit. The R1 gate re-verified the
official archive SHA256
`89d4139f809c316294b6115f115724b2af2304cba1992774e2d5086e49f7c79b`
and the Apache-2.0 root licence on 2026-11-18. The archive's own root
`SHA256SUMS` lists 37 entries; 36 matched and `.gitignore` is absent upstream,
so `36/37` is the correct statement. Any future redistribution of that source
requires preservation of its licence and notices, plus a new rights review.

### Public scenario configuration metadata

The seven public scenario configuration values reproduced in
`results/seven_cases/*/config.json` come from the same Apache-2.0 official
repository. They are retained only as execution metadata for the recorded
seven-case evidence, with no official source code, source archive, scenario
directory, challenge PDF or dashboard export included. A copy of the upstream
Apache-2.0 licence is provided in
`THIRD_PARTY_LICENSES/APACHE-2.0.txt`; upstream notices and attribution remain
with the official project.

## World Bank event-loss tables

The public repository's `data/LICENSE.txt` identifies the Central Asia seismic
and flood Event Loss Tables, CC BY 4.0:

- https://datacatalog.worldbank.org/search/dataset/0064272/central-asia-seismic-risk-event-loss-tables-elts
- https://datacatalog.worldbank.org/search/dataset/0064269/central-asia-flood-risk-event-loss-tables-elts
- https://creativecommons.org/licenses/by/4.0/

Declared transformations include country alignment, checked zero filling for
absent national rows, preservation of annual rates and mean/SD values, NPZ
conversion and no inflation adjustment. The source organisations do not
endorse this method. The NPZ files themselves are not redistributed here.
The R1 gate re-verified the frozen data hashes and the CC BY 4.0 notice on
2026-11-18 without copying or redistributing the data.

## AI assistance

OpenCode/DeepSeek and the ChatGPT web interface were used as AI assistance for
engineering, mathematical review, planning, independent verification, review
and editing. They are not vendored third-party code, formal team members,
legal authors, ADB reviewers or runtime dependencies. Their documented roles
and limits are in `AI_USE_DISCLOSURE.md`. No AI model weights, prompts,
credentials or private review archives are included.

## Citation

The ADB Challenge Brief is cited as the source of mathematical definitions but
is not redistributed. The World Bank data sources are credited above. No
copyrighted paper text, figure, table or image is reproduced beyond short
attribution. The retained project materials do not include official ADB PDFs,
dashboard exports or World Bank NPZ files.
