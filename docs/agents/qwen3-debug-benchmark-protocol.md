# Qwen 3 Debug And Benchmark Protocol

Referenced by: `doppler-debug`, `doppler-bench`, `doppler-perf`

Use this protocol for Qwen 3, Qwen 3.5, Qwen3 Embedding, and Qwen3 Reranker
lanes. The goal is to keep artifact identity, semantic correctness, and
performance evidence tied to checked-in config instead of ad hoc runtime fixes.

## Scope

Small-model lanes currently under active support:

- `qwen-3-embedding-0-6b-q4k-ehf16-af32`
- `qwen-3-reranker-0-6b-f16-af32`
- `qwen-3-reranker-0-6b-q4k-ehf16-af32`
- `qwen-3-5-0-8b-q4k-ehaf16`
- `qwen-3-5-2b-q4k-ehaf16`

## Invariants

- Conversion config is the authored source of truth.
- Manifest identity is the runtime artifact identity.
- Do not patch runtime behavior to compensate for stale manifests.
- Do not promote a model from load success alone.
- Do not publish without explicit human confirmation after dry-run review.

## Correctness Gate

Before recording a benchmark as support evidence:

1. Run the registry-owned verify command for the catalog model ID.
2. Confirm manifest-owned runtime config supplies the workload payload.
3. Confirm semantic thresholds pass for the model mode.
4. Record the exact release-claim or verify report path.

Embedding lanes must pass the semantic fixture gate and report finite,
unit-norm embeddings with the expected dimension. Reranker lanes must pass the
semantic pair gate and preserve query/document scoring order. Text lanes must
produce coherent greedy output with the manifest-owned execution path.

## Benchmark Gate

Keep benchmark evidence mode-specific:

- Embedding: use `tools/compare-embeddings.js` and
  `benchmarks/vendors/embedding-compare.config.json`.
- Text generation: use `tools/compare-engines.js` and
  `benchmarks/vendors/compare-engines.config.json`.
- Reranking: use the catalog-owned rerank verification payload and record
  rerank latency from the runtime report until a rerank-specific vendor compare
  lane exists.

Every performance claim needs a saved JSON receipt, the command that produced
it, the artifact source (`local` or `quickstart-registry`), and explicit
claimability state. Local compare wins are not release claims until the hosted
artifact is published and re-measured from the hosted source.

## Q4K Failure Handling

When F16 passes and Q4K fails:

1. Keep F16 and Q4K as separate catalog/runtime claims.
2. Diff tokenizer IDs first, then pooled embeddings or logits.
3. Compare final norm, LM head, and scoring head materialization.
4. Check quantized dequant scales and row/group layout before changing model
   semantics.
5. Leave the Q4K lane unpromoted until semantic accuracy is fixed.

## Required Receipts

For each promoted Qwen lane, keep these surfaces aligned:

- `models/catalog.json`
- source conversion config under `src/config/conversion/qwen3/`
- local or hosted `manifest.json`
- release-claim or registry-verify report
- benchmark or compare JSON receipt
- generated support inventory and support matrix

If one surface is stale, refresh from the source of truth instead of editing a
generated artifact by hand.
