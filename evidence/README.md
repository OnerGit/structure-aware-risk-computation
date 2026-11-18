# Evidence Provenance

`p1b/` contains frozen P1-B source identity, a technical replay record,
the replay result JSON, selected P1-B verification output and original
pytest output. These files document provenance; they do not replace the
public technical-evidence summary.

`../results/official_source_admission.json` records the source archive,
canonical paths, data hashes, scenarios, manifest and `7/7` problem IDs.
`../results/seven_cases/` contains actual requests, results, comparisons,
validation records, logs and resource measurements. The input identity file records the external input hashes and entry counts.
No official source archive, World Bank NPZ file, private archive or full
home path is included.

Some frozen provenance files preserve original internal path labels (for
example `review-packages/...zip`) as part of their byte identity. Those labels
are retained for audit traceability only. No review-package contents,
credentials, official PDFs or NPZ data are included.
