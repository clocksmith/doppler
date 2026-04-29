# Conversion Configs

Run conversion with the CLI `convert` command.
Canonical first-run examples live in [../../../docs/getting-started.md](../../../docs/getting-started.md).
Command contract details live in [../../../docs/api/tooling.md](../../../docs/api/tooling.md).

Notes:

- Canonical conversion-vs-runtime ownership matrix:
  [`docs/conversion-runtime-contract.md`](../../../docs/conversion-runtime-contract.md)
- `output.modelBaseId` is authoritative for emitted model IDs: when set, it is used as the final resolved `modelId`.
- If `output.modelBaseId` is omitted, converter config resolution can still append the inferred `variantTag`
  to avoid collisions (same as legacy behavior for non-explicit IDs).
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

Checked-in config inventory:

| Config | Resolved Model ID | Notes |
| --- | --- | --- |
| `embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json` | `google-embeddinggemma-300m-q4k-ehf16-af32` | Q4K embedding model with explicit runtime/session policy |
| `gemma3/gemma-3-1b-it-f16-af32.json` | `gemma-3-1b-it-f16-af32` | F16 weights, F32 activation policy |
| `gemma3/gemma-3-1b-it-q4k-ehf16-af32.json` | `gemma-3-1b-it-q4k-ehf16-af32` | Q4K row layout with execution-v1 graph |
| `gemma3/gemma-3-270m-it-q4k-ehf16-af32.json` | `gemma-3-270m-it-q4k-ehf16-af32` | Q4K row layout with execution-v1 graph |
| `gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json` | `translategemma-4b-1b-enes-q4k-ehf16-af32` | TranslateGemma distill/variant artifact |
| `gemma3/translategemma-4b-it-q4k-ehf16-af32.json` | `translategemma-4b-it-q4k-ehf16-af32` | Text-only TranslateGemma artifact |
| `gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json` | `gemma-4-e2b-it-q4k-ehf16-af32` | Gemma 4 E2B Q4K, execution-v1 graph |
| `gemma4/gemma-4-e2b-it-q4k-ehf16-af32-int4ple.json` | `gemma-4-e2b-it-q4k-ehf16-af32-int4ple` | Gemma 4 E2B Q4K with INT4 PLE policy |
| `gemma4/gemma-4-moe-q4k-ehf16-af32.json` | `gemma-4-moe-q4k-ehf16-af32` | Gemma 4 MoE Q4K |
| `gpt-oss-20b-f16-xmxfp4.json` | `gpt-oss-20b-f16-xmxfp4` | GPT-OSS F16/XMXFP4 config |
| `janus/janus-pro-1b-text-q4k-ehaf16.json` | `janus-pro-1b-text-q4k-ehaf16` | Janus text-only Q4K artifact |
| `lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json` | `lfm2.5-1.2b-instruct-q4k-ehf16-af32` | LFM2 hybrid conv/attention scheduling |
| `qwen3/qwen-3-5-0-8b-q4k-ehaf16.json` | `qwen-3-5-0-8b-q4k-ehaf16` | Qwen hybrid linear/full attention graph |
| `qwen3/qwen-3-5-2b-q4k-ehaf16.json` | `qwen-3-5-2b-q4k-ehaf16` | Qwen hybrid linear/full attention graph |
| `qwen3/qwen-3-6-27b-q4k-ehaf16.json` | `qwen-3-6-27b-q4k-ehaf16` | Qwen 3.6 27B f32 activation lane |
| `qwen3/qwen-3-6-27b-q4k-eaf16.json` | `qwen-3-6-27b-q4k-eaf16` | Qwen 3.6 27B f16 sibling over the existing Q4K weight pack |

LFM2.5 q4 kernel planning notes:

- Reuse candidates:
  - Q4K dequant primitives (`dequant_q4k*`) and q4k matmul backends (`matmul_q4k`, `dequant_matmul_f16w`) for linear weights.
  - Existing attention, RoPE, RMSNorm, residual, and sampler kernels for layers using `self_attn.*` tensors.
- New kernels required:
  - Conv operator kernels for `model.layers.*.conv.{in_proj,conv,out_proj}` blocks.
  - Execution-plan support that follows explicit hybrid `layer_types` schedules (conv vs full_attention), not only alternating/every_n attention patterns.
