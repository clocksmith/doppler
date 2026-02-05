# Conversion Configs

Use these files with:

```bash
node tools/doppler-cli.js convert <inputPath> --config <config.json>
```

Notes:

- `output.modelBaseId` is now authoritative; converter does not append implicit variant suffixes.
- All configs use `output.baseDir` (no implicit `--output-dir` requirement).
- Gemma configs pin `presets.model = gemma3` for deterministic preset selection.

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

- `tools/configs/conversion/gemma3/gemma-3-1b-it-f16-f32a.json`
  - Output base: `models/local/gemma-3-1b-it-f16-f32a`
  - Resolved modelId: `gemma-3-1b-it-f16-f32a`
  - Compute: `f32`
  - Kernel path: `gemma3-f16-fused-f32a-online`

- `tools/configs/conversion/gemma3/gemma-3-1b-it-f16-f16a.json`
  - Output base: `models/local/gemma-3-1b-it-f16-f16a`
  - Resolved modelId: `gemma-3-1b-it-f16-f16a`
  - Compute: `f16`
  - Kernel path: `gemma3-f16-fused-f16a-online`

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
