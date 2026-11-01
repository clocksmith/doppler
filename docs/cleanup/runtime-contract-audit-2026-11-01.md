# Runtime contract cleanup, 2026-11-01

Base: c23a855628e5567ad198abfb4c13d00593bc9ea1. This change repairs the
reproduced runtime and diagnostic contract violations; it is not GPU qualification.

## Ownership review

Configuration now depends on the existing `errors` owner for registered error
codes and cause-preserving envelopes. Configuration continues to own validation
policy; the dependency introduces no kernel selection or runtime fallback.

`tooling/browser-harness/benchmark-comparison.js` may read the canonical client
comparison fingerprint and observed backend identity. It does not execute client
policy. This named dependency exception avoids a second fingerprint serializer.
Remove the exception if those canonical evidence functions move to a lower shared
owner; do not broaden the exception to other tooling modules or client execution.

The architecture policy authorization digest is the SHA-256 of this document.
No line-count ceiling, dependency quarantine, or numerical acceptance threshold
is relaxed. The satisfied attention-executor soft-limit review is removed.

## Audit disposition

- Diagnostics retain the selected projection fusion and resolved recurrent precision.
- Kernel wrappers validate masks and pinned identity before selecting execution.
- The unimplemented ORT attention option rejects explicitly before allocation.
- Tokenizer and required dtype conversion failures remain failures.
- GPU and storage resources are cleaned up on partial acquisition and failure.
- Capture configuration is fully validated; debug state/counters have one owner.
- Benchmark comparison requires canonical identity and complete compatible run metadata.
- Wildcard/default runtime exports are checked against value declarations.
- CLI command output is an explicit exception to runtime debug logging, not a second logger.
- The kernel-path builder is a retained repository operator API with a documented contract
  and entrypoint test, not a new public package export.
- Existing shader/format filenames retain their identities; formatter-free development
  is explicit. No unrelated mass rename or formatting churn is introduced.
- Tracked historical receipts are retained. Ignore patterns govern new-file admission,
  not proof that existing evidence is unused.
- Existing bounded large-module/type debt remains inventoried. Planned extraction names
  are not claimed to exist; this patch does not mechanically split every large module.

The earlier forced-sequence capture and source-weight identity defects were already
fixed upstream and are not reimplemented here. Numerical gates, production deployment,
provider signup, monetary settlement and host configuration are outside this patch.
