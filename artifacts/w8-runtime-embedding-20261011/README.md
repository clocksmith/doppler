# Existing signed embedding regression on the W8A8 runtime candidate

The retained Qwen3-Embedding-0.6B F16 Capsule passed with installed archive
`975adf34cd6985b9420bbbf8c4a8736e44a84fc916465fd90158bf73fddef8c5`.
This qualifies the unchanged embedding application on the same runtime candidate;
it does not claim Gemma embedding support or W8A8 model quality.

The full compressed report preserves its exact configuration, independent source,
all outputs, signed receipts, archive identity, driver, memory, and cleanup. It
reuses the existing source tolerance `0.02`; chat's separate `0.001` is unchanged.
All 18 token sequences matched, maximum vector error was `0.00007295608520507812`,
and repeated singleton, complete batch and independently ordered subdivision
results passed. After cancellation triggered by an actual GPU submission, the
full batch in reversed input order reproduced every original token/vector exactly.

The cancelled operation rejected with the supplied cancellation reason. The test
awaited operation settlement and queue completion, then separately checked active
allocation ownership against the prior resident baseline and required zero deferred
cleanup. Existing resident weights and the 16,777,216-byte active baseline remained
unchanged. This does not label resident allocations as leaked or claim zero memory
while a model remains loaded. The Capsule was closed and owned browser/server
resources were closed successfully.

Reproduction uses the installed bundle and complete retained artifacts named in
`qualification.json.gz`'s `config`. Refresh the explicit release-policy clock to the
real execution time and use a new output directory; never replay an expired
eligibility decision by copying its historical time.

```sh
node tools/qualify-embedding-browser.js /tmp/doppler-w8-embedding-regression.json
```

The command's input configuration is preserved in the full report. No signed
manifest, shader artifact, model, source reference, or release event was rewritten.
No standard package pin changed.
