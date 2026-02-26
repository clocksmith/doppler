# Conversion Configs

Use these files with:

```bash
node tools/doppler-cli.js convert <inputPath> --config <config.json>
```

Notes:

- `output.modelBaseId` is now authoritative; converter does not append implicit variant suffixes.
- All configs use `output.baseDir` (no implicit `--output-dir` requirement).
- Gemma configs pin `presets.model = gemma3` for deterministic preset selection.
- CLI worker execution can be tuned via `--workers` and `--worker-policy` (command payload execution policy, not manifest policy).
- To pin deterministic manifest timestamps, set `manifest.conversion.convertedAt` in converter config.
- Execution-v0 fields are now supported under `inference.sessionDefaults` and `inference.execution`.
  Use this to emit explicit runtime session defaults and execution policy/steps into manifest inference.
- If `inference.defaultKernelPath` is set and no explicit `inference.execution` is provided,
  converter auto-generates execution-v0 steps/session defaults from that kernel path.

Current config intent:

- `tools/configs/conversion/gemma3/gemma-3-270m-it-f16-f32a.json`
  - Output base: `models/curated/gemma-3-270m-it-f16-f32a`
  - Resolved modelId: `gemma-3-270m-it-f16-f32a`
  - Compute: `f32`
  - Kernel path: `gemma3-f16-fused-f32a-online`

- `tools/configs/conversion/gemma3/gemma-3-270m-it-f16-f16a.json`
  - Output base: `models/local/gemma-3-270m-it-f16-f16a`
  - Resolved modelId: `gemma-3-270m-it-f16-f16a`
  - Compute: `f16`
  - Kernel path: `gemma3-f16-fused-f16a-online`

- `tools/configs/conversion/gemma3/gemma-3-270m-it-wq4k-ef16.json`
  - Output base: `models/local/gemma-3-270m-it-wq4k-ef16`
  - Resolved modelId: `gemma-3-270m-it-wq4k-ef16`
  - Weights: `q4k` (row layout), embeddings/lmHead: `f16`
  - Compute: `f32`
  - Kernel path: `gemma3-q4k-dequant-f32a-online`

- `tools/configs/conversion/gemma3/gemma-3-1b-it-wf16-ef16-hf16-f32.json`
  - Output base: `models/local/gemma-3-1b-it-wf16-ef16-hf16-f32`
  - Resolved modelId: `gemma-3-1b-it-wf16-ef16-hf16`
  - Compute: `f32`
  - Kernel path: `gemma3-f16-fused-f32a-online`

- `tools/configs/conversion/gemma3/gemma-3-1b-it-wf16-ef16-hf16-f16.json`
  - Output base: `models/local/gemma-3-1b-it-wf16-ef16-hf16-f16`
  - Resolved modelId: `gemma-3-1b-it-wf16-ef16-hf16`
  - Compute: `f16`
  - Kernel path: `gemma3-f16-fused-f16a-online`

- `tools/configs/conversion/gemma3/gemma-3-1b-it-wq4k-ef16-hf16-f16.json`
  - Output base: `models/local/gemma-3-1b-it-wq4k-ef16-hf16-f16`
  - Resolved modelId: `gemma-3-1b-it-wq4k-ef16-hf16`
  - Compute: `f16`
  - Kernel path: `gemma3-q4k-dequant-f16a-online`

- `tools/configs/conversion/gemma3/translategemma-4b-it-wq4.json`
  - Output base: `models/local/translategemma-4b-it-wq4`
  - Resolved modelId: `translategemma-4b-it-wq4`
  - Preset: `translategemma`
  - Output mode: `textOnly: true` (skip vision/projector tensors)
  - Weights: `q4k` (row layout), embeddings/lmHead: `f16`
  - Compute: `f16`
  - Kernel path: `gemma3-q4k-dequant-f16a-online`
  - Execution-v0: explicit `sessionDefaults` + full `execution.steps` mirrored from `gemma3-q4k-dequant-f16a-online`

- `tools/configs/conversion/gpt-oss-20b.json`
  - Output base: `models/local/gpt-oss-20b`
  - Resolved modelId: `gpt-oss-20b`
  - Preset: `gpt_oss`
  - Compute: `f16`

- `tools/configs/conversion/embeddinggemma/embeddinggemma-300m-wbf16.json`
  - Output base: `models/curated/google-embeddinggemma-300m`
  - Resolved modelId: `google-embeddinggemma-300m`
  - Preset: `embeddinggemma`
  - Weights/Embeddings/lmHead: `bf16`
  - Compute: `f32`
  - Kernel path: `embeddinggemma-f16-f32a`
