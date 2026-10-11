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

Physical whole/subdivided execution against an independent source reference for
one explicitly opted-in application remains unqualified. Prior generation or
synthetic embedding evidence cannot close that gate.

Component: `doppler.runtime-source.config`, `doppler.runtime-source.client`,
`doppler.tests`, `doppler.repository-tooling`.
Intent: preserved.
Acceptance evidence: focused tests, generated gates, retained installed receipt
and compressed smoke logs.
Boundary effects: additive root API and public declaration; no network or durable
job ownership enters Doppler.
