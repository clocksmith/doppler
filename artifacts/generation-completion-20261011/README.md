# Generation output and completion boundary

The off-default, model-declared output decoder exposes final-answer text while retaining raw tokens and character usage. Missing or malformed final output fails with a bounded code; cancellation and token-budget truncation remain distinct. Scoped terminal events and generation results now carry the existing public completion schema, with null for older handles lacking authoritative stopping metadata.

The installed archive is diagnostic only. No accepted model enables the decoder, no reference or tolerance changed, and the standard archive remains unchanged. Reploid consumer handling is separately retained in its `artifacts/generation-completion-20261011/`.

Validation: the combined broad sweep passed 908 of 909 files and exposed an admission-order regression in usage reset. The correction moves usage reset to admitted generation and preserves active state when concurrent/unloaded requests reject. Seven affected files then passed, as did source declarations, style, architecture, generated package/runtime closure and final detached package tests. The broad failed run is retained; it is not described as a clean full sweep.

Installed checks cover all 32 public export declarations, existing signed Capsule/direct/HTTP/partition fixtures and three reused implementation fixtures for scoped completion, output decoding and resident limits. Those implementation fixtures use installed internal owners and GPU stubs; they are not new public APIs or physical-model evidence. `review.json` records exact source hashes and the final diagnostic archive.

The initial and closure-corrected package receipts are retained separately. The final archive includes the admission fix and generated package closure. A mistyped affected-test path invocation is retained; the corrected seven-file invocation passed. Full local evidence remains under `reports/local/generation-completion-20261011` and the exact retained consumer directory recorded by the receipt.
