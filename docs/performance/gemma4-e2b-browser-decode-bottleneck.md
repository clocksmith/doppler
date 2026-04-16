# Gemma 4 E2B Browser Decode Bottleneck

**Status:** diagnosed and fixed (2026-04-15). Validation pending: re-run the
canonical compare on Apple M3 with the updated `gemma4-e2b-throughput` profile
to confirm the gap closes.

**Scope:** why `batched_gpu` decode of Gemma 4 E2B on Apple M3 browser was
~11 tok/s regardless of `batchSize` / `readbackInterval`, despite the
`rollingIds` fix unblocking the batched path.

## The evidence

Canonical receipt: [`compare-goal1.stdout`](../../compare-goal1.stdout)
(2026-04-15T20:48:31Z, captured with the tokenizerDelta P3 gating active).
An earlier point in the same investigation lives at
[`compare_20260415T170108.json`](../../benchmarks/vendors/results/compare_20260415T170108.json).

Hardware: Apple M3 / darwin arm64 / Chromium (`apple-m3` Metal 3).
Model: `gemma-4-e2b-it-q4k-ehf16-af32` (RDRR q4k/ehf16/af32).
Workload: `p064-d064-t0-k1`, 2 timed runs, warmup 1, greedy decode.

### Two points in batch-size space, ~identical throughput

| Decode profile | batchSize | readbackInterval | decodeMode | decode tok/s |
| --- | ---: | ---: | --- | ---: |
| `parity` (warm compute/parity) | 1 | 1 | `single_token` | 10.72 |
| `throughput` (warm compute/throughput) | 8 | 8 | `batched_gpu` | 11.13 |

The batched path is demonstrably live —
`batchedForwardCalls.median: 16` for 64 decode tokens × 2 runs at batch=8,
`unbatchedForwardCalls.median: 0`. Batching the forward passes is not the
decode unlock — the gain from batch=1 → batch=8 is ~4%.

### Per-phase breakdown at batch=8 / readback=8

From `sections.compute.throughput.doppler.result.metrics.gpu` in the fresh
receipt:

| Metric | Value |
| --- | ---: |
| `decodeRecordMs` (actual GPU command record) | **979.85 ms** |
| `decodeSubmitWaitMs` (queue submit stall) | 4756.60 ms |
| `decodeReadbackWaitMs` (readback stall) | 4768.35 ms |
| Total `decodeMs` | 5750.65 ms |

Interpretation:

- GPU actually doing compute: ~980 ms, i.e. **17%** of decode wall.
- Submit wait ≈ readback wait ≈ total decode time. Both counters observe
  the same GPU-completion fence; the two numbers are not two separate
  ~4.7-second stalls, they are one ~4.7-second stall observed twice.
- Per-batch wall: 5750 / 8 = **719 ms per 8-token batch**. Per-batch GPU
  record: 980 / 8 = **122 ms**. The **~600 ms / batch** (83%) that is *not*
  GPU compute is CPU↔GPU round-trip overhead per batch.

Per-token wall: ~90 ms. Per-token GPU work: ~15 ms. Non-compute overhead
per token: **~75 ms, ~83%**.

### Node surface looks the same shape

From the 2026-04-15 Node-surface bench (same model, batch=8/readback=8,
1 run): `decodeRecordMs ≈ 917 ms`, `decodeSubmitWaitMs ≈ 2893 ms`,
`decodeReadbackWaitMs ≈ 2721 ms`, total decode ≈ 3831 ms,
`decodeTokensPerSec ≈ 8.35`. GPU record is ~24% of decode; the rest is
submit/readback wait. Same fence, slightly different absolute numbers.

## Root cause — the embed_tokens fence

Within the `batched_gpu` decode path, every batch must:

1. Sample 8 tokens on the GPU (`recordArgmax` at
   `src/inference/pipelines/text/generator-steps.js:1478`, already wired).
2. Read the sampled token IDs back to the CPU
   (`readGpuTokenIdsForCpuEmbeddingGather` at
   `src/inference/pipelines/text/embed.js:144`) **because**
   `embed_tokens.weight` for Gemma 4 E2B is loaded as a CPU-resident
   range-backed source.
3. Run the CPU embedding gather row-by-row using `loadRange` against the
   OPFS-backed shard (8 sequential awaits per batch, ~24 KB per token at
   F16, hidden=1536).
4. Write the gathered embedding rows back to the GPU.
5. Submit the next batch's forward pass.

Steps 2-5 are the per-batch CPU↔GPU sync. Even though `recordArgmax` lives
entirely on the GPU, the *next* batch's input embedding lookup forces a hop
back to the CPU and back, because the embedding table is not on the GPU.
Batching just amortizes that fence over more tokens — it does not eliminate
it.

This explains:

- Batch=1 and batch=8 producing the same throughput: the per-batch
  round-trip cost is the bottleneck, so enlarging the batch reduces the
  *number* of round trips but not their wall.
- `decodeSubmitWaitMs ≈ decodeReadbackWaitMs`: both counters observe the
  same GPU-completion event from different sides.
- `decodeRecordMs` being only ~17% of decode wall.

### Why is `embed_tokens.weight` CPU-resident?

`embed_tokens.weight` is `[262144, 1536] F16` ≈ 768 MB
(`models/local/gemma-4-e2b-it-q4k-ehf16-af32/manifest.json:8991`). It hits
one of the three branches in `shouldUseRangeBackedEmbeddingSource()` at
`src/loader/embedding-loader.js:65`:

1. F16 source on a device without shader-f16.
2. The tensor location has a `sourceTransform` and `loadShardRange` is
   available, which defers full materialization.
3. `shouldStreamLargeWeight()` returns true because the estimated weight
   bytes exceed `maxBufferSize × safetyRatio`.

Diagnostic logging now distinguishes which branch fires for any given
embedding (added 2026-04-15). On Apple M3 with shader-f16 enabled, the
third branch is the most likely cause given the ~768 MB size and the
default 0.9 safety ratio.

## What the fix is *not*

**Not** tuning `batchSize`. The data already covers 1 and 8; neither helps.
Larger batches reduce round-trip count linearly but leave the per-round-trip
cost unchanged.

**Not** moving sampling to GPU. The on-GPU sampler
(`recordArgmax` / `recordGPUSample` in `src/gpu/kernels/sample.js`,
dispatched from `src/inference/pipelines/text/generator-steps.js` lines
~649 and 1478) **already ships** in both the single-token fused path and
the batched `generateNTokensGPU` path. Only 4 bytes of token IDs are read
back per batch — sampling is not the fence.

**Not** another round of the `generator-steps.js` `rollingIds` path. That
was a correctness fix that unblocked `batched_gpu` mode; the commit message
already said so.

**Not** switching to a different kernel path for attention/MLP. The compute
budget is already under 20% of decode wall. A 2× faster kernel would save
~490 ms out of 5750 ms — not enough to catch TJS's ~2160 ms decode wall for
the same 64 tokens.

**Not** the PLE per-layer-input fallback path. Gemma 4 E2B is *not* hitting
`generateNTokensGPUStepwiseRangeBackedPle` — the throughput profile has
`session.perLayerInputs.materialization: "gpu_split_tables"`, which routes
through `ensurePleGpuSplitTablesRuntime()` at
`src/inference/pipelines/text/per-layer-inputs.js:884` and populates
`embedTokensPerLayerSplit`. With that populated,
`hasRangeBackedPerLayerInputEmbeddings()` returns false and the stepwise
fallback is bypassed. The receipt confirms this: `decodeMode: "batched_gpu"`,
`batchedForwardCalls=16`, `unbatchedForwardCalls=0`.

## The fix that landed (2026-04-15)

A new runtime config field
`runtime.inference.largeWeights.gpuResidentOverrides` (string array | null)
lets a profile force specific weights to GPU residency, bypassing the
size-threshold check in `shouldStreamLargeWeight`. The
`gemma4-e2b-throughput` profile now sets:

```json
"largeWeights": {
  "gpuResidentOverrides": [
    "model.language_model.embed_tokens.weight"
  ]
}
```

This routes `embed_tokens.weight` through the GPU `recordGather` /
`runGather` path at `src/inference/pipelines/text/embed.js:309-336` instead
of the CPU `readGpuTokenIdsForCpuEmbeddingGather` branch. Token IDs stay on
the GPU; no per-batch CPU↔GPU sync is needed for the embedding lookup.

Schema and contract:

- `src/config/schema/doppler.schema.js` —
  `DEFAULT_LARGE_WEIGHT_CONFIG.gpuResidentOverrides: null`
- `src/config/schema/doppler.schema.d.ts` —
  `LargeWeightConfigSchema.gpuResidentOverrides: string[] | null`
- `src/loader/manifest-config.js` `shouldStreamLargeWeight()` checks the
  override list before the size estimate
- `src/loader/embedding-loader.js`
  `shouldUseRangeBackedEmbeddingSource()` — diagnostic logs for each
  residency branch
- `src/config/runtime/profiles/gemma4-e2b-throughput.json` — override
  populated
- Contract tests:
  - `tests/config/runtime-large-weights-defaults-contract.test.js`
    (default + override merge)
  - `tests/loader/should-stream-large-weight-overrides.test.js` (behavior)
  - `tests/config/gemma4-e2b-runtime-profiles-contract.test.js` (profile
    locks the override in)

Memory cost: ~768 MB extra GPU. Gemma 4 E2B currently uses ~5.15 GB on M3,
so the M3 budget fits.

## Validation

Re-run the canonical compare workload with the updated throughput profile.
The fence is closed if:

- Loader logs print
  `Embedding "model.language_model.embed_tokens.weight" forced GPU-resident via runtime.inference.largeWeights.gpuResidentOverrides.`
  once at load.
- `decodeRecordMs / decodeMs` ratio rises substantially (compute is no
  longer dominated by the round trip).
- `decodeSubmitWaitMs` and `decodeReadbackWaitMs` shrink toward
  `decodeRecordMs`.
- Decode tok/s rises toward the TJS baseline.

If the fence is *not* closed, check the loader logs to see which residency
branch actually fired. The new diagnostic logs in
`shouldUseRangeBackedEmbeddingSource` distinguish the F16-without-shader-f16
case, the sourceTransform case, and the `shouldStreamLargeWeight` case
explicitly. If branch 1 or 2 fires, the override does not apply and a
different fix is required.

## Claim implications

Until validation lands, the public claim should remain anchored to the
cold-load advantage; see
[`docs/model-support-matrix.md`](../model-support-matrix.md) and the README
summary for the current phrasing. Cold load is also noisy between runs
(2.5× in one receipt, 1.83× in another) and should be re-measured and
medianed across several runs before being cited.

After validation, the decode tok/s claim can move from "Doppler ~11 vs
TJS ~29" toward whatever the post-fix receipt shows.

## LiteRT/TFLite lane status (2026-04-15)

Goal 2 (the stronger "Doppler LiteRT beats both RDRR and TJS ONNX" claim)
is **parked until reference data exists**.

- The Gemma 4 E2B `.task` (`gemma-4-E2B-it-web.task`) PLE scale-companion
  layout is partially understood: the `_quantized_scale` bytes appear to be
  packed F32 row-scales (one F32 per row, 262144 rows per layer) stored
  inside a UINT8 tensor container. A local experiment that auto-detected
  this and bypassed the UINT8 affine-dequant gate *did* load the model
  end-to-end and ran prefill+decode to completion.
- **But real inference produced incoherent output** across two prompts:
  `"Hello"` → `"蔗izmiFCO🥥"`, `"The color of the sky is"` →
  `"Цена性别砾 लोहा"`. Both outputs bias toward high-vocabulary-ID
  characters, the signature of a dequant whose magnitude is wrong or
  whose sign/axis convention is mismatched.
- The pre-existing `resolveScale` absolute-vs-local row offset bug that
  fell out of that experiment is real and worth keeping. Commit
  `8f222a8` ships it in isolation.
- The PLE auto-detection itself is **not committed**. Landing more
  heuristics without a reference would violate Doppler's
  config-first/runtime-contract rules because it can silently produce
  plausible-but-wrong tokens — the worst possible regression surface.
- Performance was also unusable in that loads-but-wrong state:
  `decodeTokensPerSec: 0.04` on Node surface, ~25 seconds per token.

**What unblocks the lane:** either (a) ground-truth F16 intermediate
values from a reference LiteRT-LM runner for the same `.task` file and
the same prompt, diffed against Doppler's intermediates at the first
point of divergence, or (b) a published LiteRT-LM packing spec that
names the PLE scale-companion convention so Doppler's dequant rules can
be written from the spec instead of guessed.

Until one of those is in hand, the LiteRT lane stays experimental and
non-claimable.
