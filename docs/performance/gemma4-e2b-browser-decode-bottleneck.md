# Gemma 4 E2B Browser Decode Bottleneck

**Status:** 2026-04-16. The `gpuResidentOverrides` fix from 2026-04-15 is
shipped (commit `c3e137f`), fires correctly on Apple M3 browser, and
`embed_tokens.weight` is now GPU-resident — but the decode gap vs TJS
**did not close**. The earlier root-cause analysis was wrong. See
[Corrected diagnosis](#corrected-diagnosis-2026-04-16) below for the actual
bottleneck.

**Scope:** why `batched_gpu` decode of Gemma 4 E2B on Apple M3 browser is
~11 tok/s (vs TJS ~28.5 tok/s) regardless of `batchSize` /
`readbackInterval`, despite the `rollingIds` fix unblocking the batched path
and the `gpuResidentOverrides` fix moving `embed_tokens.weight` to GPU.

## The evidence

Canonical post-fix receipt: [`compare-goal1.stdout`](../../compare-goal1.stdout)
(2026-04-16T00:27:34Z, Apple M3, the manifest-owned throughput path with
`gpuResidentOverrides` active). The corresponding pre-fix receipt is
preserved in git history at commit
[`3e1a6f9`](https://github.com/clocksmith/doppler/commit/3e1a6f9fc0711a70a3b55856307e5033b850f34c)
(`git show 3e1a6f9:compare-goal1.stdout`). Both captured on the same
M3, same workload (`p064-d064-t0-k1`), tokenizerDelta P3 gating active.

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

**Not** a runtime-preset-owned PLE materialization choice. The shipped Gemma
4 E2B base fast path uses `session.perLayerInputs.materialization:
"gpu_split_tables"`, but that setting is owned by the conversion config and
manifest. Generic runtime profiles should not carry Gemma-specific
materialization overrides. The receipt confirms the important runtime
property: `decodeMode: "batched_gpu"`, `batchedForwardCalls=16`,
`unbatchedForwardCalls=0`.

## The fix that landed (2026-04-15)

A new inference config field
`largeWeights.gpuResidentOverrides` (string array | null) lets a converted
artifact force specific weights to GPU residency, bypassing the size-threshold
check in `shouldStreamLargeWeight`. Gemma 4 E2B conversion configs and
manifests now set:

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
- `src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json` —
  override populated for converted artifacts
- `models/local/gemma-4-e2b-it-q4k-ehf16-af32/manifest.json` — local
  manifest mirrors the converted artifact contract
- Contract tests:
  - `tests/config/runtime-large-weights-defaults-contract.test.js`
    (default + override merge)
  - `tests/loader/should-stream-large-weight-overrides.test.js` (behavior)
  - `tests/config/gemma4-e2b-runtime-profiles-contract.test.js`
    (conversion config and local manifest lock the override in)

Memory cost: ~768 MB extra GPU. Gemma 4 E2B currently uses ~5.15 GB on M3,
so the M3 budget fits.

## Corrected diagnosis (2026-04-16)

The 2026-04-15 root-cause analysis — that `decodeSubmitWaitMs` represents
per-batch CPU↔GPU round trips caused by CPU-resident `embed_tokens.weight`
— was **wrong**. Post-fix validation on the same M3 proves it.

### What the M3 validation actually showed

Running the canonical compare on the same hardware with
the manifest-owned throughput path loading correctly:

- Loader logs confirm the override fires:
  `[Loader] Embedding weight "model.language_model.embed_tokens.weight" forced GPU-resident via runtime.inference.largeWeights.gpuResidentOverrides.`
- Total GPU memory after load: **6.16 GB** (up from ~5.15 GB pre-fix — the
  +768 MB lands where expected: on the GPU side of `embed_tokens.weight`).
- `isWeightBuffer=true` for the embedding, so the CPU gather branch at
  `embed.js:144` (`readGpuTokenIdsForCpuEmbeddingGather`) no longer fires.

Yet the decode metrics barely moved:

| Metric (throughput, median) | Pre-fix | Post-fix | Δ |
| --- | ---: | ---: | ---: |
| decode tok/s | 11.13 | **11.56** | +3.9% |
| decodeRecordMs | 979.85 | 1030.70 | +5.2% |
| decodeSubmitWaitMs | 4756.60 | **4477.15** | -5.9% |
| decodeReadbackWaitMs | 4768.35 | 4499.95 | -5.6% |
| total decodeMs | 5750.65 | 5534.20 | -3.8% |
| TJS tok/s (same run) | 29.60 | 28.46 | — |

The same shape under parity (batch=1): **10.72 → 11.24** (+4.9%).

That is **within-noise on a single receipt** and nowhere near closing a
2.6× gap. Whatever the fence is, moving the main embedding table to the
GPU did not touch it.

### Why the old interpretation was wrong

`decodeSubmitWaitMs` does not measure "per-batch CPU↔GPU round-trip
overhead." It measures **wall time from `recorder.submit()` until the
submit-latency callback fires**, i.e. how long the GPU actually takes to
drain the submitted command list. `decodeRecordMs` is the CPU-side time
spent *building* the command encoder before the submit. The two together
with `decodeReadbackWaitMs` approximate total decode wall:

- `decodeRecordMs` ≈ CPU encoding time (small, ~17–18% of wall)
- `decodeSubmitWaitMs` ≈ GPU execution wall time (large, ~78% of wall)
- `decodeReadbackWaitMs` ≈ wall from submit to readback mapAsync resolving
  (reports ~the same number as submit wait on M3; they are observing the
  same GPU-completion fence from two sides, plus a small readback tail)

The correct interpretation of the per-batch breakdown is:

- Per batch of 8 tokens at batch=8/readback=8: ~360 ms wall,
  ~62 ms CPU encoding, ~300 ms GPU executing the recorded command list.
- Per token: ~45 ms of which ~37 ms is GPU kernel execution.
- TJS on the same hardware does ~35 ms per token end-to-end.

So Doppler's GPU kernel list is doing roughly **2.6× more GPU-second work
per token than TJS's ONNX runtime**. That is a kernel/plan issue, not a
CPU round-trip issue.

### What the override is still good for

Keep it. Without `gpuResidentOverrides`, Doppler's CPU gather branch in
`embed.js:144` fires for every decode step, and decode mode is forced
through `generateNTokensGPUStepwiseRangeBackedPle` whenever it runs out of
hot-vocabulary cache hits. The override is a *prerequisite* for any
GPU-resident batched decode path on models with ≥ ~400 MB embeddings, and
the browser behavior breaks in a different and worse way without it. What
the override is *not* is a TJS-parity unlock for Gemma 4 E2B on its own.

### Where the real decode time is going

The browser kernel-select log for this run shows:

```
matmul variant=f16w_f32a reason=path_override
matmul variant=gemv_subgroup_multicol reason=path_override
matmul variant=gemv_subgroup_vec4 reason=path_override
matmul variant=gemv_subgroup reason=gemv
attention variant=decode_online_f16kv reason=path_override:subgroup
```

Both the matmul and attention paths are running with **f32 activations**
(`f16w_f32a` and `_f16kv`). That doubles the activation memory bandwidth
relative to an `f16`-activation plan. `src/config/transforms/execution-graph-transforms.js:114`
is explicit about it: attention variants whose file name contains
`_f16kv` resolve to `precision.activationDtype = 'f32'`; only variants
whose file name contains pure `_f16` resolve to f16 activations. Gemma 4
E2B is locked to the `_f16kv` attention family by the current kernel-path
selection.

Pure-f16 attention WGSL kernels already exist in
`src/gpu/kernels/attention/` (`attention_decode_online_f16.wgsl`,
`attention_small_f16.wgsl`, `attention_streaming_f16.wgsl`) — they are
wired and used for Gemma 2 / Gemma 3 on the `_f16a` family of execution
graphs. This older note previously referenced the removed runtime
kernel-profile directory; new work must use execution-v1 graphs and inline
execution-v1-derived runtime overrides. Gemma 4 E2B has no equivalent
pure-f16 execution path yet.

That is the concrete path forward for closing the gap.

### Path forward

1. Add an execution-v1 graph or transform routing Gemma 4 E2B
   matmul/attention through the pure-f16 variants instead of
   `f16w_f32a` / `_f16kv`.
2. Validate numerically on the q4k-ehf16-af32 artifact first — activations
   becoming f16 has real magnitude implications for the per-layer
   embeddings (PLE) projection and the final logit softcap. Accept only if
   decode output stays coherent and logits match the f32 baseline within
   tolerance.
3. Re-run the canonical compare. The hypothesis is a roughly 1.5–2×
   speedup on decode, which would put Doppler within 30% of TJS on the
   same workload rather than 2.6× behind.
4. Only then move the public tok/s claim in
   [`docs/model-support-matrix.md`](../model-support-matrix.md) and the
   README.

Do *not* attribute the remaining gap back to embedding residency, CPU
round-trips, or the `gpuResidentOverrides` path — that lane is closed and
the receipts prove it.

## Claim implications

The public tok/s claim for Gemma 4 E2B must remain anchored to the
cold-load advantage; see
[`docs/model-support-matrix.md`](../model-support-matrix.md) and the README
summary for the current phrasing. Cold load is also noisy between runs
(2.5× in one receipt, 1.83× in another) and should be re-measured and
medianed across several runs before being cited.

A decode-speed claim is not available yet. The pre-fix claim "Doppler ~11
vs TJS ~29" remains the honest post-fix picture: 11.56 vs 28.46 on M3 with
the override active. Do not promote this.

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
