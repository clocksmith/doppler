# Evidence storage

Committed evidence under `artifacts/` and `reports/` must be cited by code,
tests, config, docs, a report, or a workspace repository. Evidence that nothing
cites moves to the maintainers' evidence store, and this directory keeps its
digest index.

- [external-evidence-index.json](external-evidence-index.json) lists every moved
  file by repository-relative path, sha256 and byte count, plus the last commit
  that tracked it. Restoring a file means copying it back to the same path and
  checking its sha256 against the index, or checking it out from that commit.
- The store is not public. The index records no machine names or absolute paths.
- Raw evidence bundles (tarballs, logs, captures) go to the store with an index
  entry rather than into git, unless a test or receipt reads them from the
  repository.

On 2026-10-29, 39 unreferenced `artifacts/` folders (657 files, about 71 MB)
moved to the store. The other 48 folders that an earlier review had counted as
unreferenced are cited by receipts in `reports/` or by other workspace
repositories, so they stay.
