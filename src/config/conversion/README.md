# Conversion Configs

Run conversion with the CLI `convert` command.
Canonical first-run examples live in [../../../docs/getting-started.md](../../../docs/getting-started.md).
Command contract details live in [../../../docs/api/tooling.md](../../../docs/api/tooling.md).

Notes:

- Canonical conversion-vs-runtime ownership matrix:
  [`docs/conversion-runtime-contract.md`](../../../docs/conversion-runtime-contract.md)
- `output.modelBaseId` is now authoritative; converter does not append implicit variant suffixes.
- All configs use `output.baseDir` (no implicit `--output-dir` requirement).
- Conversion configs are self-contained. Do not rely on external family selection or implicit family detection.
- Worker execution tuning belongs in `request.convertPayload.execution`
  (`workers`, `workerCountPolicy`, `rowChunkRows`, `rowChunkMinTensorBytes`,
  `maxInFlightJobs`, `useGpuCast`, `gpuCastMinTensorBytes`).
- To pin deterministic manifest timestamps, set `manifest.conversion.convertedAt` in converter config.
- Execution-v1 configs may set `execution.inlineKernelPath: false` when the
  manifest must own an explicit execution graph without lowering it into
  `runtime.inference.kernelPath`.
- Manifest session policy is authored as `inference.session`.

Current config intent:

- `src/config/conversion/gemma3/gemma-3-270m-it-f16-af32.json`
  - Output base: `models/local/gemma-3-270m-it-f16-af32`
  - Resolved modelId: `gemma-3-270m-it-f16-af32`
  - Compute: `f32`
  - Kernel path: `gemma3-f16-fused-f32a-online`

- `src/config/conversion/gemma3/gemma-3-270m-it-f16.json`
  - Output base: `models/local/gemma-3-270m-it-f16`
  - Resolved modelId: `gemma-3-270m-it-f16`
  - Compute: `f16`
  - Kernel path: `gemma3-f16-fused-f16a-online`

- `src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehaf16.json`
  - Output base: `models/local/gemma-3-270m-it-q4k-ehaf16`
  - Resolved modelId: `gemma-3-270m-it-q4k-ehaf16`
  - Weights: `q4k` (row layout), embeddings/lmHead: `f16`
  - Compute: `f16`
  - Kernel path: `gemma3-q4k-dequant-f16a-online`

- `src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json`
  - Output base: `models/local/gemma-3-270m-it-q4k-ehf16-af32`
  - Resolved modelId: `gemma-3-270m-it-q4k-ehf16-af32`
  - Weights: `q4k` (row layout), embeddings/lmHead: `f16`
  - Compute: `f32`
  - Kernel path: `gemma3-q4k-dequant-f32a-online`
  - Session: decode loop `batchSize=4`, `stopCheckMode=batch`, `readbackInterval=1`, `disableCommandBatching=false`

- `src/config/conversion/gemma3/gemma-3-1b-it-f16-af32.json`
  - Output base: `models/local/gemma-3-1b-it-f16-af32`
  - Resolved modelId: `gemma-3-1b-it-f16-af32`
  - Compute: `f32`
  - Kernel path: `gemma3-f16-fused-f32a-online`
  - Session: decode loop `batchSize=4`, `stopCheckMode=batch`, `readbackInterval=1`, `disableCommandBatching=false`

- `src/config/conversion/gemma3/gemma-3-1b-it-f16.json`
  - Output base: `models/local/gemma-3-1b-it-f16`
  - Resolved modelId: `gemma-3-1b-it-f16`
  - Compute: `f16`
  - Kernel path: `gemma3-f16-fused-f16a-online`

- `src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json`
  - Output base: `models/local/gemma-3-1b-it-q4k-ehf16-af32`
  - Resolved modelId: `gemma-3-1b-it-q4k-ehf16-af32`
  - Compute: `f16`
  - Kernel path: `gemma3-q4k-dequant-f16a-online`

- `src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json`
  - Output base: `models/local/gemma-3-1b-it-q4k-ehf16-af32`
  - Resolved modelId: `gemma-3-1b-it-q4k-ehf16-af32`
  - Weights: `q4k` (row layout), embeddings/lmHead: `f16`
  - Compute: `f32`
  - Kernel path: `gemma3-q4k-dequant-f32a-online`
  - Session: decode loop `batchSize=4`, `stopCheckMode=batch`, `readbackInterval=1`, `disableCommandBatching=false`

- `src/config/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json`
  - Output base: `models/local/translategemma-4b-it-q4k-ehf16-af32`
  - Resolved modelId: `translategemma-4b-it-q4k-ehf16-af32`
  - Output mode: `textOnly: true` (skip vision/projector tensors)
  - Weights: `q4k` (row layout), embeddings/lmHead: `f16`
  - Compute: `f16`
  - Kernel path: `gemma3-q4k-dequant-f16a-online`
  - Execution: explicit `session` + `execution.kernels`, `preLayer`, `decode`, `prefill`, `postLayer`, and `policies`
  - Session: decode loop `batchSize=4`, `stopCheckMode=batch`, `readbackInterval=1`, `disableCommandBatching=false`

- `src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json`
  - Output base: `models/local/translategemma-4b-1b-enes-q4k-ehf16-af32`
  - Resolved modelId: `translategemma-4b-1b-enes-q4k-ehf16-af32`
  - Weights: `q4k` (row layout), embeddings/lmHead: `f16`
  - Compute: `f32`
  - Kernel path: `gemma3-q4k-dequant-f32a-online`
  - Session: decode loop `batchSize=4`, `stopCheckMode=batch`, `readbackInterval=1`, `disableCommandBatching=false`

- `src/config/conversion/gemma4/gemma-4-moe-q4k-ehf16-af32.json`
  - Output base: `models/local/gemma-4-moe-q4k-ehf16-af32`
  - Resolved modelId: `gemma-4-moe-q4k-ehf16-af32`
  - Weights: `q4k` (row layout), embeddings/lmHead: `f16`
  - Compute: `f32`
  - Session: decode loop `batchSize=4`, `stopCheckMode=batch`, `readbackInterval=1`, `disableCommandBatching=false`

- `src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json`
  - Output base: `models/local/gemma-4-e2b-it-q4k-ehf16-af32`
  - Resolved modelId: `gemma-4-e2b-it-q4k-ehf16-af32`
  - Weights: `q4k` (row layout), embeddings/lmHead/vision/projector/audio: `f16`
  - Compute defaults: `activation=f32`, `math=f32`, `accum=f32`, `output=f32`
  - Execution-v1: explicit decode/prefill graph with pinned WGSL kernels
  - Session: decode loop `batchSize=8`, `stopCheckMode=batch`, `readbackInterval=8`, `disableCommandBatching=false`

- `src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32-int4ple.json`
  - Output base: `models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple`
  - Resolved modelId: `gemma-4-e2b-it-q4k-ehf16-af32-int4ple`
  - Weights: `q4k` (row layout), embeddings/lmHead/vision/projector/audio: `f16`
  - Per-layer embeddings: `int4_per_row` with runtime `range_backed` materialization
  - Compute defaults: `activation=f32`, `math=f32`, `accum=f32`, `output=f32`
  - Execution-v1: explicit decode/prefill graph with pinned WGSL kernels
  - Session: decode loop `batchSize=8`, `stopCheckMode=batch`, `readbackInterval=8`, `disableCommandBatching=false`

- `src/config/conversion/gpt-oss-20b-f16-xmxfp4.json`
  - Output base: `models/local/gpt-oss-20b-f16-xmxfp4`
  - Resolved modelId: `gpt-oss-20b-f16-xmxfp4`
  - Compute: `f16`
  - Session: decode loop `batchSize=4`, `stopCheckMode=batch`, `readbackInterval=1`, `disableCommandBatching=false`

- `src/config/conversion/janus/janus-pro-1b-text-q4k-ehaf16.json`
  - Output base: `models/local/janus-pro-1b-text-q4k-ehaf16`
  - Resolved modelId: `janus-pro-1b-text-q4k-ehaf16`
  - Output mode: `textOnly: true`
  - Weights: `q4k` (row layout), embeddings/lmHead: `f16`
  - Compute: `f16`
  - Session: decode loop `batchSize=4`, `stopCheckMode=batch`, `readbackInterval=1`, `disableCommandBatching=false`

- `src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json`
  - Output base: `models/local/qwen-3-5-0-8b-q4k-ehaf16`
  - Resolved modelId: `qwen-3-5-0-8b-q4k-ehaf16`
  - Output mode: `textOnly: false` (keep multimodal-compatible artifact layout; text path uses the language tower)
  - Manifest multimodal contract: explicit canonical `manifest.visionConfig`
  - Weights: `q4k` (row layout), embeddings/vision/projector: `f16`, tied lmHead: `q4k`
  - Compute: `f32`
  - Kernel path: `null` (no explicit manifest kernel-path contract)
  - Execution-v1: explicit execution graph with `inlineKernelPath: true`
  - Hybrid contract: fused-Q4 projections stay primary for the shared linear-attention path; optimized fused-Q4 `main_gemv` with `COLS_PER_WG=64` and `THREADS_PER_COL_GEMV=4` drives the quantized tied LM head
  - Session: decode loop `batchSize=4`, `stopCheckMode=batch`, `readbackInterval=32`, `disableCommandBatching=false`

- `src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json`
  - Output base: `models/local/qwen-3-5-2b-q4k-ehaf16`
  - Resolved modelId: `qwen-3-5-2b-q4k-ehaf16`
  - Output mode: `textOnly: true`
  - Weights: `q4k` (row layout), embeddings: `f16`, tied lmHead: `q4k`
  - Compute: `f32`
  - Kernel path: `null` (no explicit manifest kernel-path contract)
  - Execution-v1: explicit execution graph with `inlineKernelPath: true`
  - Hybrid contract: fixed fused-Q4 `main_gemv` is primary for transformer decode; optimized fused-Q4 `main_gemv` with `COLS_PER_WG=64` and `THREADS_PER_COL_GEMV=4` drives the quantized tied LM head, while stable fused-Q4 `main_multicol` remains declared as the fallback Q4 decode kernel
  - Prefill contract: `q4_widetile` projections and `attn_head256`
  - Session: decode loop `batchSize=12`, `stopCheckMode=batch`, `readbackInterval=32`, `disableCommandBatching=false`

- `src/config/conversion/sana/sana-sprint-0.6b-f16.json`
  - Output base: `models/local/sana-sprint-0.6b-f16`
  - Resolved modelId: `sana-sprint-0.6b-f16`
  - Weights/embeddings/lmHead: `f16`
  - Compute: `f16`
  - Intended source: `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`
  - Kept `f16` intentionally while Sana runtime work is still being validated; no reusable `q4k` diffusion config is checked in yet.

- `src/config/conversion/embeddinggemma/google-embeddinggemma-300m-bf16-af32.json`
  - Output base: `models/local/google-embeddinggemma-300m`
  - Resolved modelId: `google-embeddinggemma-300m`
  - Weights/Embeddings/lmHead: `bf16`
  - Compute: `f32`
  - Kernel path: `embeddinggemma-f16-f32a`

- `src/config/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json`
  - Output base: `models/local/lfm2.5-1.2b-instruct-q4k-ehf16-af32`
  - Resolved modelId: `lfm2.5-1.2b-instruct-q4k-ehf16-af32`
  - Weights: `q4k` (row layout), embeddings/lmHead: `f16`
  - Compute: `f32`
  - Kernel path: `lfm2-q4k-dequant-f32a-online` (explicit; LFM2 fast-prefill F32A path)
  - Session only: decode loop `batchSize=8`, `stopCheckMode=batch`, `readbackInterval=8`, `disableCommandBatching=false`
  - Does not emit execution schema because custom conv layer scheduling skips kernel-path auto-generation

- `src/config/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehaf16.json`
  - Output base: `models/local/lfm2.5-1.2b-instruct-q4k-ehaf16`
  - Resolved modelId: `lfm2.5-1.2b-instruct-q4k-ehaf16`
  - Weights: `q4k` (row layout), embeddings/lmHead: `f16`
  - Compute: `f16`
  - Kernel path: `gemma3-q4k-dequant-f16a-online`

LFM2.5 q4 kernel planning notes:

- Reuse candidates:
  - Q4K dequant primitives (`dequant_q4k*`) and q4k matmul backends (`matmul_q4k`, `dequant_matmul_f16w`) for linear weights.
  - Existing attention, RoPE, RMSNorm, residual, and sampler kernels for layers using `self_attn.*` tensors.
- New kernels required:
  - Conv operator kernels for `model.layers.*.conv.{in_proj,conv,out_proj}` blocks.
  - Execution-plan support that follows explicit hybrid `layer_types` schedules (conv vs full_attention), not only alternating/every_n attention patterns.
