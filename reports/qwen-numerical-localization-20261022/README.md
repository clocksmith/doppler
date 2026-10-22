# A winner flip that was state reuse, not numerical instability (2026-10-22)

While measuring the Q4_K LM-head change on AMD, every timestamp-profiled request on
`qwen-3-5-2b-q4k-ehaf16` (standard-BLAKE3 copy, Radeon 8060S, Chrome 146 Vulkan) diverged at
decode step 2 of "The capital of France is": unprofiled requests chose `.` (13), profiled ones
`\n` (198).

## Oracle

[f64-oracle.py](f64-oracle.py) runs the existing `tools/qwen35-f64-reference.py` equations in
F64 over the deployed Q4_K weights, after checking every shard against the independently retained
SHA-256 inventory. Prompt tokens `[760, 6511, 314, 9338, 369]` plus `11751` (" Paris") match
Doppler's 5-token prefill. The next token is `.` (13) at 20.931, then `,` (11) at 20.151
(margin 0.780); `\n` (198) is 14.826, 6.1 below the winner ([f64-oracle.json](f64-oracle.json)).
The production route agrees with the oracle; the profiled route is not a near tie.

## Localization ([flag-probe.js](flag-probe.js))

- `profile: true` alone causes the flip; `executionObserver` does not
  ([before-flag-variants.json](before-flag-variants.json)).
- Profiled requests dispatch every decode kernel about twice as often as unprofiled ones
  (LM head 62 vs 31, recurrent linear attention 1,134 vs 576 = 18 layers × 63 vs 32 forwards);
  prefill is not doubled.
- The runtime logged `Batch decode failed, falling back to single-token: Error: [CommandRecorder]
  Timestamp query capacity did not cover every compute pass.`
  ([before-fallback-warning.json](before-fallback-warning.json)).

`generateNTokensGPU` submits the batch (advancing KV and linear-attention recurrent state), then
the profiling coverage check throws before `currentSeqLen` advances. `decode-runtime.js` caught
the error and decoded the same position again single-step. Attention KV is rewritten in place,
but the recurrent state had already advanced, so the retry computed from corrupted state. Any
error thrown after a batch submission (coverage, invalid token, finiteness) took the same path.

## Repair

- `generateNTokensGPU` marks errors raised after any submission with `decodeStateAdvanced`.
- `generator/batch-retry-policy.js` refuses the in-place retry when that flag is set and the
  model has linear-attention or convolution state, and throws an explicit error with the cause.
  Attention-only models keep the existing single-token retry, including finiteness recovery.

After the repair the profiled request fails explicitly, and the next request on the same session
again produces `[11751, 13, 11751, 369, 279, 6511]` ([after-fix.json](after-fix.json)).
`tests/inference/batch-retry-policy.test.js` covers the decision; decode-related inference tests pass.

## Not done

- Profiled batched decode on recurrent models now fails instead of measuring: the timestamp query
  set (4,096 queries on this device) holds 2,048 passes, and profiling records one pass per
  dispatch (about 312 per token here). Resolving timestamps over several rounds would restore it;
  `gpu/command-recorder.js` is already over its 750-line review limit.
- The single-token profiled route also runs two forwards per step with coherent output; not
  explained here.
- Retained March LFM2.5 reports contain the same fallback warning; LFM2.5 has convolution state,
  so those runs may have been affected. Not re-examined.
- This case says nothing about the per-logit 0.001 failures on Qwen 3.5 0.8B, where every
  selected token matched and the October 11 audit attributed the error to the inherited hidden
  state, not the LM head (`reports/local/q4-accumulation-20261011/final-projection-audit.json`).

Component: `doppler.runtime-source.inference`, `doppler.tests`, `doppler` (evidence).
Intent: preserved.
Acceptance evidence: this directory, `tests/inference/batch-retry-policy.test.js`, decode-related `tests/inference/*.test.js`, `typecheck:source`, `source:style:check`.
Boundary effects: profiled batched decode on recurrent models now rejects instead of returning corrupted tokens.
