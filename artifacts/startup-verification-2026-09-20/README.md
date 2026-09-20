# Reuse owned shard verification during model loading

Component: doppler.runtime-source.client, doppler.runtime-source.storage,
doppler.tests, doppler.docs, doppler.repository-tooling. Intent: preserved.
Boundary effects: the Capsule adapter consumes storage's canonical shard
normalization and verified-store ownership; no upward dependencies, new runtime
modules, model mathematics, or execution-policy changes.

## Hypothesis and bounded repair

The accepted runtime authenticates Capsule artifacts into private owned blocks,
then the pipeline's storage context reads those blocks again to verify the
manifest's shard hashes. The retained range-copy probe reproduces two complete
read passes for one shard. This hypothesis predicts both an extra model-sized
copy and another hash pass per session, even when sessions share backing.

The candidate reuses a snapshot only when the store was created by Doppler's
verified-store factory and every normalized manifest shard SHA-256 digest/size
agrees with its owned receipt. Store interfaces are frozen so those authenticated
readers cannot subsequently be replaced. A WeakSet identifies factory ownership;
it does not retain store lifetimes or provide a global content cache. Arbitrary
ports, copied interfaces, proxies, and BLAKE3 manifest digests retain byte
verification. Mismatches fail closed. The canonical normalization function remains
owned by storage rather than copied into the client adapter.

Incremental SHA-256, private block storage, host leases, cancellation ownership,
pooling, serialized compatibility execution, kernels, and numerical references
are unchanged. This does not implement chunked source acquisition or make the
initial hashing loop responsive to event-loop cancellation; those remain next.

## Installed physical comparison

The `doppler-perf` procedure required a fresh unchanged baseline before the
candidate. Both runs use the same Qwen3-4B Capsule, request, frozen 107-token
reference, unbounded retention, Linux/Chrome 146.0.7680.177, and physical AMD RDNA-3
adapter. Model runs execute sequentially, not concurrently. Local repository/UI
checks also ran during the investigation; this is local acceptance/tuning evidence,
not a claim-grade vendor benchmark or confidence interval.

| Measurement | Fresh baseline | Candidate |
| --- | ---: | ---: |
| First pipeline load (ms) | 129,727 | 4,070 |
| Second pipeline load (ms) | 129,941 | 3,785 |
| Complete lifecycle sequence (ms) | 486,942 | 239,480 |
| First-session returned bytes | 16,106,849,387 | 8,061,913,195 |
| Second-session returned bytes | 16,101,486,302 | 8,056,550,110 |
| Initial verified-store hashing (ms) | 120,267.1 | 121,251.6 |

Pipeline-load timers exclude earlier Capsule verification. The complete sequence
includes all repeated requests, failed/cancelled preparation, operation
cancellation, second-session opening, and cleanup; it is not single-request TTFT.
Both runs produce four exact token-reference matches. Both retain 5,698,022 metadata
cache bytes per open session and share 8,050,837,118 backing bytes. Second-session
source reads, store hashing, and new backing copies remain zero. Both closes
release backing leases/cache bytes; this is not immediate GC/driver reclamation.

The baseline archive SHA-256 is
`b2ba7dbf7a18df458a702881439d2f9195b8b2a6a864f5fed2fa51afba831f6c`.
The candidate is `/var/tmp/doppler-startup-reuse-20260920/doppler-gpu-0.6.2.tgz`,
SHA-256 `c3166b1a6a480f52dd0661b67b427f6918eb004b015da84d31bbc86ccb57f13c`.
Its installed-consumer smoke passes. Four existing shipped files change, with no
new entries; `package-audit.json` records the separate CI-toolchain size audit.
Historical archives, signatures, and references are not rewritten. No publication.

## Reproduction and regression checks

Use each retained config with its exact installed bundle/model roots and a fresh
output directory. The runner refuses to overwrite existing output.

```sh
node tools/check-installed-capabilities.js artifacts/startup-verification-2026-09-20/baseline-config.json
node tools/check-installed-capabilities.js artifacts/startup-verification-2026-09-20/candidate-config.json
node tools/check-installed-capabilities.js artifacts/startup-verification-2026-09-20/candidate-4gib-config.json
node tools/check-installed-capabilities.js artifacts/startup-verification-2026-09-20/candidate-embedding-reranking-config.json
node tools/check-installed-capabilities.js artifacts/startup-verification-2026-09-20/candidate-adapter-config.json
node tests/runtime/capsule-artifact-verification-reuse.test.js
node tests/runtime/capsule-artifact-source.test.js
node tests/runtime/capsule-artifact-retention.test.js
node tests/runtime/capsule-artifact-allocation.test.js
npm run check:green
```

The focused regressions cover one-copy reads, forged hash methods, copied/proxied
interfaces, frozen readers, manifest size/hash/path mismatch, normalization,
source/caller mutation, close, and valid/corrupt BLAKE3 manifests. Existing
allocation-failure and shared-backing race tests still pass. The demo contract
passes with mocked execution and the regenerated shell; it is not GPU parity.

## Completed acceptance

- `npm run check:green`: passed, 851 test files. The initial invocation found a
  stale generated dependency view; regenerating it and rerunning the full command
  passed. No runtime exception or test assertion was weakened.
- Installed package, CI-toolchain package boundaries, source types/style/architecture,
  component inventory, and mocked demo contract: passed. Log digests are in
  [checks.json](checks.json).
- [Original-settings candidate](candidate-physical-summary.json): four exact
  107-token matches against the frozen reference, compared with the
  [fresh baseline](baseline-physical-summary.json).
- [4 GiB control](candidate-4gib-physical-summary.json): four exact matches;
  complete sequence 237,207 ms. Returned bytes, backing, metadata retention, and
  second-session zero source/hash/backing allocation match the original-settings
  candidate. No cache evictions or repeated acquisition occurred. This cache limit
  still does not bound total model memory.
- [Embedding and reranking](candidate-embedding-reranking-physical-summary.json):
  four completed requests each, unchanged frozen references and tolerances.
  Maximum embedding absolute error is 0.000070625 (limit 0.02); reranking order is
  exact, maximum logit error 0.567641 (limit 1), probability error 0.012751
  (limit 0.05). These are tolerance-based comparisons, not bitwise parity claims.
- [Adapter](candidate-adapter-physical-summary.json): seven exact 107-token
  matches with the retained separate adapter-capable Capsule and zero-delta
  rank-one fixture under explicit 4 GiB retention. Activation, unload, failed
  preparation, cancellation, and closing the first session preserve later use.
  This does not qualify nonzero adapter mathematics.

All modes exercise cancellation, failed preparation, repeated requests, and
second-session execution after first-session close. Each summary pins the raw
receipt digest, tested archive, runner/fixture identity, and physical adapter.
The original-settings candidate's initial adapter probe was unavailable once,
then succeeded under the unchanged explicit retry policy; no CPU fallback ran.

The final original-settings GPU counters remain 4,949,428,824 buffer bytes across
486 objects without observed explicit destruction. No GPU-pool behavior changed;
this neither proves a leak nor accounts for physical driver residency. Pool
ownership/repeated-cycle accounting remains a separate next task.

The `doppler-perf` procedure kept this repair to one measured variable: reuse of
owned verification. Initial hash throughput, chunked acquisition/cancellable
hashing, GPU retention classification, and explicit operation contracts remain
future work; this record does not claim those are complete.
