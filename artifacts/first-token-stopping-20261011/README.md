# First-token stopping metadata

The installed Gemma run on archive `47d7d18f` emits stopping token 106 as the
first token of `copy-code`, but records `stop-token-or-sequence`. That combined
reason is outside the public completion contract and makes scoped result
construction reject otherwise completed execution. Its empty answer still fails
the task-quality requirement; correcting metadata cannot make that answer useful.

The generator now classifies the same known stopping condition as `stop-token`
or `stop-sequence`, using the existing canonical token predicate. This matches
later-token behavior, including its treatment of EOS as a token stop. Cancellation
keeps precedence and records no stopping token. Generated tokens, numerical
operations, model files, thresholds and public schemas remain unchanged.

The three corrected initial-token paths are streaming generation, token-ID
generation and generation from an existing prefix. The behavioral test covers
EOS, declared stopping tokens, stopping text, simultaneous token/text stops,
callback cancellation and subsequent reuse. It fails before the repair and passes
afterward; compressed logs and the original physical observation are retained.
No extra decoding is allowed after any initial stop.

Passed checks:

```sh
node tests/inference/generate-token-ids-behavioral-parity.test.js
node tests/inference/generator-stop-token-parity.test.js
node tests/inference/generator-abort-delivery.test.js
node tests/inference/generate-token-ids-contract.test.js
node tests/integration/doppler-scoped-session.test.js
npm run typecheck:source
npm run source:style:check
npm run source:architecture:check
```

The stopped full repository-check invocation predates this correction and is
retained separately. The [final installed model replay](../w8a8-model-qualification-20261011/README.md#final-completion-metadata-control)
on archive `7c240dad` now verifies both variants: all task tokens and 200
teacher-forced observations remain unchanged, all twelve tasks expose valid
`stop-token` metadata, and both cancellation controls plus settled unload pass.
Task quality remains failed independently of completion classification.

Component: `doppler.runtime-source.inference.pipelines.text`, `doppler.tests`.
Intent: preserved. Boundary effects: valid completion classification reaches
the existing public scoped-session contract without a new schema or fallback.
