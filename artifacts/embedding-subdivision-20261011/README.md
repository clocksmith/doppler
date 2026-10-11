# Embedding subdivision contract

This increment makes the maintained text-embedding implementation's existing
per-text semantics public. It does not qualify a new model, approve application
subdivision, or replace the standard 0.6.27 archive.

`resolveEmbeddingBatchSemantics(manifest)` validates the existing embedding
contract and describes single-text units, state reset per text, and input-order
output. It describes neither arbitrary injected programs nor generation or
reranking. Applications retain their existing signed model, execution, input,
comparison and authorization bindings; no parallel identity protocol was added.

The offline subdivision evaluator requires complete original positions, including
duplicate texts, and compares both whole and subdivided observations against the
same independent reference. It also compares them with each other using that
reference's unchanged vector tolerance and exact-token policy. Two identically
wrong results still fail the independent source check.

Validation:

- Four focused modules pass: `embedding-batch-semantics.test.js`,
  `embedding-reference.test.js`, `capsule-embedding.test.js`, and
  `embedding-state-reset.test.js`.
- The real public Capsule operation adapter passes whole versus separately
  submitted subsets with a synthetic pipeline, preserving duplicate texts and
  original order. This is a contract test, not physical model accuracy.
- Export, source type/style/architecture, config single-source, CATSCAN, API docs,
  package boundary and runtime-closure checks pass.
- The retained installed-package smoke passes public runtime/declaration imports,
  embedding admission and semantics, signed execution, cancellation and HTTP/direct
  parity with synthetic devices and programs. Its exact test-only archive identity
  is in `installed-receipt.json`; it remains under
  `/tmp/doppler-embedding-subdivision-installed-20261011/` locally.

The initial package check rejected stale generated closure and the newly exposed
declaration's file/byte inventory. The generated closure was synchronized and the
package budget increased only by the measured one declaration and 4,064 bytes.
`package-inventory.json` binds the resulting 1,898 entries and 13,119,792 bytes.
No model weights, new runtime module, release version, or standard archive changed.

## Physical qualification of the existing document embedding application

The existing `doppler-offline-document-search` application explicitly permits
subdivision between complete text inputs under its unchanged signed application,
model, execution, and reference bindings. A unit may contain complete texts in
their original positions; it must not split a text, alter prompt preparation,
change pooling, reuse another input's mutable state, or alter numerical policy.
This allowance covers its embedding operation only, not its reranker or chat.

The retained Qwen3-Embedding-0.6B F16 Capsule passed on AMD Radeon 8060S/RADV
and Chromium 145.0.7632.6 using the test-only installed archive above. All 18
frozen texts ran singly, repeatedly, as one complete batch, and as two nine-text
units in odd/even completion order through signed `executeOperation` calls.
Every token sequence matched the independent source exactly. Whole/reference and
split/reference maximum vector error was `0.00007295608520507812`, below the
unchanged embedding reference tolerance `0.02`. Split/whole vector error was zero.
The source reference is the retained PyTorch calculation pinned to Qwen source
revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`; its canonical digest remains
`sha256:be4ed910a95a6e7ed26cfe7ce39d539760187698d7f8703a106afbabcdc46772`.
This embedding policy does not replace the separate chat tolerance `0.001`.

`physical/attempts.json` preserves the initial setup failures: noncanonical ISO
time, missing v3 checkpoint persistence, and checkout-omitted weight shards.
The harness now persists verified checkpoints before execution, rejecting forks
and rollback. The isolated complete Capsule contains 42 size/digest-verified
artifacts (`physical/capsule-assets.json`); no model bytes were rebuilt.

The first passing run uses the application's existing retained-local decision;
its receipt explicitly records expired fresh eligibility. That mode rejects
delegated assignments. Only after current source and subdivision qualification
passed, the existing publisher appended eligibility event 2 with the maintained
release-signing API, preserving the complete history and unchanged Capsule,
application, weights, expiry duration and TargetPlan. `physical/release-renewal.json`
binds that local, unpublished renewal to the passing report.

`physical/managed.json.gz` retains the subsequent passing managed-eligibility run.
Its uncompressed SHA256 is
`c0a8cb541ee6bd65ec6c2d0d66fb544889a097e3546c44fd62d307de396a75ce`.
All singleton, whole and unit receipts, output vectors, source vectors, host
identities and cleanup results are retained. Reploid's real signed-job durability
acceptance remains a separate gate; this run alone does not qualify network
providers, fresh delivery, other devices, or independent adoption.

Component: `doppler.runtime-source.config`, `doppler.runtime-source.client`,
`doppler.tests`, `doppler.repository-tooling`.
Intent: preserved.
Acceptance evidence: focused tests, generated gates, retained installed receipt
and compressed smoke logs.
Boundary effects: additive root API and public declaration; no network or durable
job ownership enters Doppler.
