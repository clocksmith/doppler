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
- Execution-v0 fields are supported under `inference.sessionDefaults` and `inference.execution`.
  `inference.execution` requires explicit `inference.sessionDefaults` and emits
  `manifest.inference.schema = "doppler.execution/v0"`.
- `inference.sessionDefaults` without `inference.execution` does not by itself
  emit execution-v0 schema; it persists manifest batching/session defaults only.
- If `inference.defaultKernelPath` is set and no explicit `inference.execution` is provided,
  converter auto-generates execution-v0 steps/session defaults from that kernel path.
  If `inference.sessionDefaults` is also provided, it overlays the generated
  execution-v0 session defaults before validation.
  Hybrid custom-layer models with explicit `layerPattern.layerTypes` containing `conv`
  skip this auto-generation and keep layer scheduling in manifest inference.

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

- `src/config/conversion/gemma3/gemma-3-1b-it-f16-af32.json`
  - Output base: `models/local/gemma-3-1b-it-f16-af32`
  - Resolved modelId: `gemma-3-1b-it-f16-af32`
  - Compute: `f32`
  - Kernel path: `gemma3-f16-fused-f32a-online`

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

- `src/config/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json`
  - Output base: `models/local/translategemma-4b-it-q4k-ehf16-af32`
  - Resolved modelId: `translategemma-4b-it-q4k-ehf16-af32`
  - Output mode: `textOnly: true` (skip vision/projector tensors)
  - Weights: `q4k` (row layout), embeddings/lmHead: `f16`
  - Compute: `f16`
  - Kernel path: `gemma3-q4k-dequant-f16a-online`
  - Execution-v0: explicit `sessionDefaults` + full `execution.steps` mirrored from `gemma3-q4k-dequant-f16a-online`

- `src/config/conversion/gpt-oss-20b-f16-xmxfp4.json`
  - Output base: `models/local/gpt-oss-20b-f16-xmxfp4`
  - Resolved modelId: `gpt-oss-20b-f16-xmxfp4`
  - Compute: `f16`

- `src/config/conversion/qwen3/qwen-3-5-0-8b-f16.json`
  - Output base: `models/local/qwen-3-5-0-8b-f16`
  - Resolved modelId: `qwen-3-5-0-8b-f16`
  - Output mode: `textOnly: true` (skip vision/projector tensors from Qwen3.5 multimodal checkpoints)
  - Weights/embeddings/lmHead: `f16`
  - Compute: `f16`
  - Kernel path: `null` (no explicit manifest kernel-path contract)
  - Session defaults only: decode loop `batchSize=4`, `stopCheckMode=batch`, `readbackInterval=1`, `disableCommandBatching=true`
  - Does not emit execution-v0 schema because no execution graph is authored/generated

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
  - Session defaults only: decode loop `batchSize=8`, `stopCheckMode=batch`, `readbackInterval=8`
  - Does not emit execution-v0 schema because custom conv layer scheduling skips kernel-path auto-generation

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
