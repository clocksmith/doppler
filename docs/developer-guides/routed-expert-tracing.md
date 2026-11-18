# Request-scoped routed expert diagnostics

Internal diagnostic helpers live in `src/loader/experts/routed-expert-trace.js`.
They are not a public package export. A benchmark owning a loaded text pipeline
can bracket one request using its `pipeline.dopplerLoader`:

```js
beginRoutedExpertTrace(pipeline.dopplerLoader, {
  requestId: 'workload-1-attempt-1', maxRecords: 4096, maxIndices: 1048576,
});
try {
  // Fully consume/await the one request, including its cleanup on failure.
} finally {
  const trace = finishRoutedExpertTrace(pipeline.dopplerLoader, 'workload-1-attempt-1');
  // Save the trace beside the request receipt, including model/build identity.
}
```

Import `beginRoutedExpertTrace` and `finishRoutedExpertTrace` explicitly from that
internal module. Limits are mandatory, not runtime defaults. Begin only at an
idle request boundary, and finish after its outstanding work settles. An active
trace cannot be replaced; the matching request ID is required to finish it.
Unload cancels capture. Finishing while a diagnostic read is pending fails
rather than returning an incomplete snapshot as complete. Explicit cancellation
is available for an abandoned request; late reads cannot contaminate a new one.

Records contain the original layer index, prefill/decode phase, chunk-local
`tokenOffset`, token count, top-K and raw source expert indices in token-major
order. Duplicates are preserved, before compact slot remapping. A decode call's
offset restarts at zero; it is not an absolute sequence position. These records
cover Qwen execution with the routed loading policy enabled, including its tiered
resident banks. Ordinary fully resident execution with routed policy disabled
bypasses this capture; this is not universal MoE tracing.
Observed selections do not prove that their computation subsequently completed.

Paging reuses the readback already required for weight loading. Fully resident
banks normally need no readback; tracing adds one only while capture budget
remains. Therefore capture-enabled timings are diagnostics, not uninstrumented
performance evidence. Ordinary profiling does not enable this trace. With capture
disabled, resident execution adds no readback and paging uses its original read.

Bounds cover both retained record count and total expert-index count, including
in-flight reservations. An over-budget chunk is omitted whole; `omittedRecords`
and `omittedIndices` disclose truncation. A failed admitted read also increments
these fields. Never infer full prompt coverage from a truncated trace. Compare
per-layer expert sets between request receipts offline; count duplicate routes
separately when measuring frequency or estimating preload value.

Model-free checks: `node tests/loader/routed-expert-trace.test.js` and
`node tests/inference/routed-expert-trace-wiring.test.js`. The latter executes the
real JavaScript chunk wiring with CPU stubs for GPU operations. Physical GPU
qualification of recorded IDs remains separate; no performance improvement is
claimed by adding diagnostics.
