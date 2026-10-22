# Qwen3-Reranker-0.6B Q4K successor: routing and standard BLAKE3 (2026-10-21)

The checked-in reranker manifest (`03298342…`) routed the F16 FFN projections through the
one-output-per-thread `matmul_f16w_f32a` and prefill attention through `attention_small_f16kv`
(17 ms per layer at 613 tokens). Its shard hashes are historical nonstandard BLAKE3 values
([RDRR format](../../docs/rdrr-format.md#shard-hash-algorithms-and-legacy-blake3)), so standard
verification rejects all 14 shards.

The conversion configuration now selects `matmul_f16w_f32a_tiled` and `attention_head128_f16kv`,
both already registered and used by other models, under a new manifest variant
`…-mv-source-pinned-03-tiled-head128`. A successor artifact carries that graph and standard
BLAKE3. The predecessor manifest and its qualification records stay unchanged.

## Successor construction ([build-successor.js](build-successor.js))

1. Copied 14 shards and the tokenizer as separate files to local NVMe
   (`.cache/qwen-reranker-q4k-successor-20261021`).
2. All 15 files match the independent SHA-256 inventory in
   [model-derivation.json](../choice-scoring/model-derivation.json)
   ([verification](byte-inventory-verification.json)).
3. Doppler's corrected BLAKE3 and Python `blake3` 1.0.11 (official empty-input vector checked)
   agree on every shard ([migration record](blake3-migration.json)).
4. `shardSetHash` and `weightPackHash` were recomputed with the converter's own canonical
   method, after first reproducing the predecessor's values exactly from its legacy shard list.
5. `tools/refresh-converted-manifest.js` applied the conversion configuration: execution graph,
   current kernel digests, mechanism kernels, integrity extensions and identity.
   [successor-manifest.json](successor-manifest.json) is `451c6a87…`.

Weight and tokenizer bytes are unchanged; `metadata.integrityMigration` records the predecessor
identity and that qualification is still required.

## Behavior ([summary](behavior-summary.json))

On 90 NLBSE issue-triage requests (three `scoreChoices` calls each, Chrome 146 Vulkan,
Radeon 8060S, source compat API):

- Same selected route on 90/90; yes/no logits differ from the predecessor by at most 0.0086
  (median 0.0012). The frozen scoring tolerance is 0.05.
- Bit-identical to the routed experiment manifest, so the refresh's added fields
  (`queryKeyNormAxis`, `queryKeyNormType`, `supportsSequence`, implicit embedding vocabulary)
  do not change computation.
- Median 1,928.6 → 628.3 ms per issue, including the
  [register-tiled matmul fix](../f16w-tiled-shared-layout-20261021/README.md).

This is development evidence. Signed-opening qualification against the CPU reference on an
installed archive, browser qualification, and a Capsule remain to be done for the successor.
The external volume's copy (`rdrr/qwen-3-reranker-0-6b-q4k-ehf16-af32`, July manifest
`6b1eaca7…` routing prefill attention to `attention_streaming_f16kv`) is older than both
manifests and was not modified.

Component: `doppler.runtime-source.config`, `doppler` (evidence).
Intent: preserved.
Acceptance evidence: this directory; `npm run kernels:registry:check`, `onboarding:check`, `config:single-source:check`, `routing:audit:check`, `kernels:conversion-digests:check -- --source-only`.
Boundary effects: none; the predecessor manifest, fixtures and qualification records are unchanged.
