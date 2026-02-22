# Conversion Configs

Use these files with:

```bash
node tools/doppler-cli.js convert <inputPath> --config <config.json>
```

Notes:

- All configs set `output.modelId` explicitly to the final emitted ID.
- All configs use `output.baseDir` (no implicit `--output-dir` requirement).
- Gemma configs pin `presets.model = gemma3` for deterministic preset selection.
- `*-f16-f32a-*` configs are the stability-first path (F16 weights + F32 activations).
- `*-wf16` configs are F16-activation experimental paths.
- `*-wbf16` configs preserve BF16 weight storage with F32 compute.

Current config intent:

- `configs/conversion/gemma3/gemma-3-270m-it-f16-f32a.json`
  - Output: `models/curated/gemma-3-270m-it-f16-f32a-wf16`
  - Compute: `f32`
  - Kernel path: `gemma3-f16-fused-f32a-online`

- `configs/conversion/gemma3/gemma-3-270m-it-wf16.json`
  - Output: `models/local/gemma-3-270m-it-wf16`
  - Compute: `f16`
  - Kernel path: `gemma3-f16-fused-f16a-online`

- `configs/conversion/gemma3/gemma-3-1b-it-f16-f32a.json`
  - Output: `models/local/gemma-3-1b-it-f16-f32a-wf16`
  - Compute: `f32`
  - Kernel path: `gemma3-f16-fused-f32a-online`

- `configs/conversion/gemma3/gemma-3-1b-it-wf16.json`
  - Output: `models/local/gemma-3-1b-it-wf16`
  - Compute: `f16`
  - Kernel path: `gemma3-f16-fused-f16a-online`

- `configs/conversion/gpt-oss-20b.json`
  - Output: `models/local/gpt-oss-20b-wf16`
  - Preset: `gpt_oss`
  - Compute: `f16`

- `configs/conversion/embeddinggemma/embeddinggemma-300m-wbf16.json`
  - Output: `models/curated/google-embeddinggemma-300m-wbf16`
  - Preset: `embeddinggemma`
  - Weights/Embeddings/lmHead: `bf16`
  - Compute: `f32`
  - Kernel path: `embeddinggemma-f16-f32a`
