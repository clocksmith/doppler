# First-token cancellation disposition

The installed W8A8 model run exposed a cancellation during the first token's
callback that reported `stop-token-or-sequence` instead of `aborted`. The existing
behavioral parity harness reproduces it when that first token is EOS. Cancellation
must retain its own disposition even when the token also satisfies a stop rule.

`before.log.gz` preserves the failing invocation; `after.log.gz` records the same
regression passing after correction:

```sh
node tests/inference/generate-token-ids-behavioral-parity.test.js
```

The test exercises the real streaming and token-ID generation owners with stubbed
GPU logits. It covers ordinary and EOS first tokens, one callback, no further
decode, null stop-token identity on cancellation, released generation admission,
and a successful subsequent request. This is a synthetic control, not a new model
quality or physical-execution result. The same initial-token boundary in the
prefix-KV adapter receives the explicit abort precedence; it is not separately
physically qualified by this test.

Component: `doppler.runtime-source.inference.pipelines.text`.
Intent: preserved.
Boundary effects: cancellation remains distinct from natural stopping.
