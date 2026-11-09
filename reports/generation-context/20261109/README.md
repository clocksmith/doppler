# Explicit conversation context reuse

This is development evidence for retaining a stable prompt checkpoint across
whole-model requests. Applications select opaque contexts; Doppler owns KV,
hybrid recurrent/convolution memory, exact token-prefix validation and GPU
settlement. Weights remain shared. The checkpoint excludes generation-only chat
template suffixes so ordinary Qwen follow-ups can reuse it without changing the
template or treating visible answer text as identical to hidden thinking tokens.

`m3-physical.json` records a clean source candidate based on `e45aa918`, with
modified runtime hashes, Qwen 0.8B manifest identity, Node/Dawn/Metal adapter,
256-token KV capacity and nine checks. Cold and retained-prefix prefill were
compared with ordinary fresh prefill on every value of the observed vocabulary
row at absolute tolerance 0.001. These are cached-versus-fresh controls, not an
independent model reference or full-model qualification. Generated IDs match;
templated chat output matches. Longer-prefix input has 82 tokens and reuses 76.
Cancellation invalidates retention; edited history rebuilds; budget refusal and
eviction release retained bytes. Timings are individual development observations,
not a performance comparison. Batched decode does not expose every logit row.

Reproduce with local model bytes:

```sh
DOPPLER_CONTEXT_MODEL_DIR=/path/to/model DOPPLER_CONTEXT_RECEIPT=/tmp/context.json node tests/integration/generation-context-physical.test.js
```

Focused checks: scoped-session, model-session reset, model-host boundaries,
generation prompt reset and generate-token-IDs parity; source types, unused and
architecture checks; package import/type/runtime smoke, package source closure,
runtime closure and unchanged demo shell budgets.

The existing `pipeline-session-interleaving.test.js` fails three tests both on
the starting `e45aa918` and the working tree (construction/stream cleanup
failures). It is not reported as passing or attributed to this change.

Limits: ordinary contiguous GPU KV only; no hyper-connection, independent
convolution, partition, activation-recording or multimodal context support.
No decode batching across requests, GPU preemption, or persistent conversation
storage. Retained bytes exclude the working cache, weights and copy temporaries.
No 27B, AMD, browser or public-provider qualification is established here.
