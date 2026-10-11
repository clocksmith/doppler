# Seeded sampling and cancellation during batch delivery

A physical fixed-logit control found a direct-generator cancellation defect:
`_runDecodeLoop()` checked the abort signal before submitting a recorded batch,
but kept delivering its buffered tokens after the consumer aborted. The retained
`cancellation-owner-settled.json.gz` records a second visible token after abort.
The repaired delivery boundary suppresses remaining output, records `aborted`,
and starts no fallback or speculative dispatch after observing cancellation.
The same delivery boundary also caught consumer exceptions as GPU computation
failures and initiated a single-token fallback. The retained before/after
sentinel test demonstrates that defect. Only the batch computation now enters
that fallback catch; callback, decoding and iterator errors propagate unchanged.
Already submitted GPU work may finish; cancellation does not roll back model
state or provide checkpoint recovery.

The final candidate (`settled-delivery-final.json.gz`) passes on Chrome 146.0.7680.177, AMD Radeon 8060S Graphics,
RADV 26.0.3, Vulkan 1.4.335. The receipt records the exact base commit, candidate
source digest, served module/shader digests, browser and CDP adapter/driver
identity. Only the model embedding/logit producers are replaced with fixed GPU
tensor fixtures. Sampling WGSL, uniform preparation, recorded command batches,
readback, buffer ownership, generator delivery and cancellation are real.

Five seeds (zero, integer, fractional, uint32 wrap, maximum finite Number) produce
identical independently expected tokens through whole batches, split batches and
same-index retry. Frozen caller seeds remain unchanged. After one delivered token,
abort suppresses the rest. A fresh attempt then produces all eight expected tokens.
Every checked boundary has zero active and deferred-cleanup bytes; the reusable
pool remains bounded at 18,432 bytes after the cancellation/fresh-attempt shape.
Consumer `onToken` and `onBatch` errors also preserve their original identity and
settle ownership without another decode. No uncaptured WebGPU errors occurred. These are sampler/lifecycle results, not
model accuracy, task-quality, release or cross-device qualification.

Deterministic regressions additionally exercise token and text delivery for
batched, scalar and speculative decoding, cancellation while computation is
pending, cancellation while the generator is suspended, callback-triggered abort,
and an aborted batch failure that must not start a fallback dispatch. Separate
sentinel tests cover `onToken`, `onBatch`, token decoding, and iterator `throw()`;
a genuine GPU computation failure still exercises the pre-existing fallback.

Earlier fixture import/context failures and a premature cleanup assertion are
retained separately. Buffer-pool retirement captures separate waves of retired
buffers; one queue-completion promise need not cover a later retirement wave.
The corrected harness awaits those actual completion waves with a bounded guard,
without sleeps, and requires zero pending ownership. The subsequent two-token
failure is the production cancellation defect, not a cleanup failure.

Reproduce against committed source:

```sh
node tests/inference/generator-abort-delivery.test.js
node tests/inference/generator-batch-delivery-errors.test.js
node tests/integration/seeded-batch-physical.test.js /tmp/seeded-batch.json
```

For an explicit uncommitted delivery candidate, the physical runner accepts
`--candidate-decode-runtime`; the receipt binds both original and candidate
hashes. All other runtime files come from an immutable `git archive HEAD` snapshot.
No standard archive, model, precision, numerical reference or tolerance changed.

Component: `doppler.runtime-source.inference.pipelines.text`, `doppler.tests`.
Intent: preserved. Boundary effects: generation output cancellation and scoped
physical evidence; sampling arithmetic and network coordination are unchanged.

Validation includes the affected deterministic tests and immutable HEAD plus
owned-repair architecture, declarations, style and runtime-closure checks. The
concurrent working-tree unit sweep passed 906 files; it is retained as a diagnostic,
not exact-source release acceptance. Other agents were editing unrelated modules
during that sweep. The added delivery-error sentinel test passed separately.
Shared-tree check failures while those modules were incomplete, the first
architecture snapshot's missing Git metadata, and the unsupported `typecheck`
script invocation are retained; corrected isolated checks passed.
