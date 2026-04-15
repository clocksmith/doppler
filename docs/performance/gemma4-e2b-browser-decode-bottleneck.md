# Gemma 4 E2B Browser Decode Bottleneck Diagnostic

**Status:** diagnosed (2026-04-15), not fixed. Next action is experimental, not code.

**Scope:** why `batched_gpu` decode of Gemma 4 E2B on Apple M3 browser is ~11.5
tok/s regardless of `batchSize` / `readbackInterval`, despite the
`rollingIds` fix unblocking the batched path.

## The evidence

Canonical receipt:
[`benchmarks/vendors/results/compare_20260415T170108.json`](../../benchmarks/vendors/results/compare_20260415T170108.json)

Hardware: Apple M3 / darwin arm64 / Chromium (`apple-m3` Metal 3).
Model: `gemma-4-e2b-it-q4k-ehf16-af32` (RDRR q4k/ehf16/af32).
Workload: `p064-d064-t0-k1`, 2 timed runs, warmup 1, greedy decode.

### Two points in batch-size space, ~identical throughput

| Decode profile | batchSize | readbackInterval | decodeMode | decode tok/s |
| --- | ---: | ---: | --- | ---: |
| `parity` (warm compute/parity) | 1 | 1 | `single_token` | 11.43 |
| `custom` (compute/throughput) | 8 | 8 | `batched_gpu` | 11.59 |

The batched path is demonstrably live — `batchedForwardCalls: 16` for 64
decode tokens × 2 runs at batch=8, `unbatchedForwardCalls: 0` — but the
throughput gain is ~1%. Batching the forward passes is not the decode unlock.

### Per-phase breakdown at batch=8 / readback=8

From `sections.compute.throughput.doppler.result.metrics.gpu`:

| Metric | Value |
| --- | ---: |
| `decodeRecordMs` (actual GPU command record) | **944.75 ms** |
| `decodeSubmitWaitMs` (queue submit stall) | 4559.05 ms |
| `decodeReadbackWaitMs` (readback stall) | 4576.15 ms |
| `decodeOrchestrationMs` (overlap credit) | −4556.10 ms |
| Total `decodeMs` | 5523.85 ms |

Interpretation:

- GPU actually doing compute: ~944 ms, i.e. **17%** of decode wall.
- Submit wait ≈ readback wait ≈ total decode time, and the orchestration
  metric is a large negative number. That's the shape you get when submit
  and readback are both waiting on the *same* GPU-completion fence and the
  counters double-attribute the wait; the two numbers are not two separate
  4.5-second stalls, they are one 4.5-second stall observed twice.
- Per-batch wall: 5524 / 16 = **345 ms per 8-token batch**. Per-batch GPU
  record: 944 / 16 = **59 ms**. The **286 ms / batch** (83%) that is *not*
  GPU compute is CPU-GPU round-trip overhead: submit, wait for completion,
  readback sampled tokens, prepare next batch, re-submit.

Per-token wall: 86.3 ms (from `decodeMsPerTokenP50`). Per-token GPU work:
~7.4 ms. Non-compute overhead per token: **~79 ms, ~91%**.

### Node surface looks the same

From the 2026-04-15 Node-surface bench (`loadMode=http`, same model,
batch=8/readback=8, 1 run):

| Metric | Value |
| --- | ---: |
| `decodeRecordMs` | 917 ms |
| `decodeSubmitWaitMs` | 2893 ms |
| `decodeReadbackWaitMs` | 2721 ms |
| Total decode | 3831 ms |
| `decodeTokensPerSec` | 8.35 |

Same shape: GPU record is ~24% of decode, the rest is submit/readback wait.
Node is in fact slower than browser (8.35 vs 11.59 tok/s), so the bottleneck
is not "the browser is adding overhead Node escapes."

## Hypothesis

The decode wall is dominated by a single CPU↔GPU serial round trip that
cannot be hidden by the current `readbackMode: "overlapped"` pipeline:

1. CPU records a command buffer for the next 8-token batch and submits.
2. CPU then has to wait for *this* batch to complete before it can stage
   the next one, because the sampled token from step N becomes the input
   for step N+1, and sampling currently happens CPU-side after readback.
3. The "overlap" that the ring buffers (`ringTokens: 2`, `ringStop: 1`,
   `ringStaging: 2`) were supposed to provide only masks a small part of
   that round trip; the dominant cost is still the readback→sample→submit
   chain per batch.

Equivalent framing: each batch's GPU work finishes in ~60 ms, but the next
batch can't start until the previous batch's output tokens are back on the
CPU and have been run through the sampler. That's the ~286 ms/batch of
non-GPU time we see.

This is consistent with:

- Batch=1 and batch=8 producing the same total throughput: the round-trip
  cost per batch is the bottleneck, so enlarging the batch reduces the
  number of round trips but not the wall.
- `decodeSubmitWaitMs ≈ decodeReadbackWaitMs`: both counters observe the
  same GPU-completion event from different sides.
- `decodeRecordMs` being small (944 ms of real GPU work in 5524 ms of
  decode): if compute were the bottleneck, this number would dominate.

## What the fix is *not*

**Not** tuning `batchSize`. The data already covers 1 and 8; neither helps.
Tuning up to 16 or 32 will reduce round-trip count linearly but leave the
per-round-trip cost unchanged, so the asymptote stays near the current
value. Stop expecting `--doppler-batch-size` alone to close the gap to TJS.

**Not** another round of the `generator-steps.js` rollingIds path. That
was a correctness fix that unblocked `batched_gpu` mode; it was not a
performance fix and the commit message already said so.

**Not** switching to a different kernel path for attention/MLP. The compute
budget is already under 20% of decode wall. A 2× faster kernel would save
~470 ms out of 5524 ms — not enough to catch TJS's ~2100 ms decode wall for
the same 64 tokens.

## What the next experiment is

In rough priority order:

1. **Move sampling to the GPU for batched decode.** If the sampler (argmax
   at greedy, top-k/top-p at temperature) runs on-device and feeds its
   output straight into the next batch's token embedding lookup, readback
   is no longer on the hot path and the `next batch needs previous batch's
   token` dependency moves from a CPU round trip to a GPU pipeline stage.
   `src/inference/pipelines/text/sampling.js` + the decode loop in
   `generator-steps.js` are the two places to touch.

2. **Verify the overlapped readback pipeline is actually overlapping.** The
   runtime profile says `readbackMode: "overlapped"` with ring sizes
   2/1/2 (see `src/config/runtime/profiles/gemma4-e2b-throughput.json`),
   but the submit/readback/record timings above look like a serial
   pipeline. Either the ring is too small to cover the round-trip cost, or
   the overlap path isn't wired in for Gemma 4. An A/B between
   `readbackMode: "overlapped"` and a sequential fallback, with the same
   metric capture, will say which.

3. **Check whether the CPU sampler is the residual.** If experiment 1 is
   infeasible, a cheaper probe is to measure CPU-side time between GPU
   readback of logits and CPU submission of the next token. If that gap is
   tens of milliseconds, the sampler itself (or its surrounding book-
   keeping: KV-cache update, PLE lookup, per-layer input preparation) is a
   target. The KV cache layout is already `contiguous` for Gemma 4 E2B
   (see memory note in `CLAUDE.md` / `MEMORY.md`) so the usual suspects
   don't apply, but per-layer input preparation is worth tracing.

4. **Don't touch the batch path until 1/2/3 have moved the needle.** The
   batched forward count is already maxed (16 / 16) and
   `unbatchedForwardCalls` is zero; there is no room for batching to help
   until the CPU round-trip cost is smaller than the per-batch GPU work.

## Claim implications

Until the round-trip cost comes down, the `Doppler RDRR` browser decode
lane on Apple M3 is ~2.6× behind Transformers.js ONNX/q4f16 for Gemma 4
E2B. The honest public claim is still the cold-load advantage; see
[`docs/model-support-matrix.md`](../model-support-matrix.md) and the
README summary for the current phrasing.
