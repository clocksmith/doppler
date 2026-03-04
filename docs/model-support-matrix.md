# Model Support Matrix

Auto-generated from preset registry (`src/config/loader.js`), conversion configs (`tools/configs/conversion/**`), and catalog (`models/catalog.json`).
`models/catalog.json` lifecycle metadata is the canonical source for hosted/demo/tested status.
Run `npm run support:matrix:sync` after adding/changing presets, conversion configs, or catalog entries.

Updated at: 2026-03-04

| Preset | Runtime modelType | Runtime | Conversion configs | Catalog models | Hosted (HF) | Demo | Tested | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| functiongemma | transformer | active | 1 (tools/configs/conversion/functiongemma/functiongemma-template-wf16-ef16-hf16-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| embeddinggemma | embedding | active | 2 (tools/configs/conversion/embeddinggemma/embeddinggemma-300m-wbf16.json, tools/configs/conversion/embeddinggemma/embeddinggemma-300m-wq4k-ef16.json) | 1 (google-embeddinggemma-300m-wq4k-ef16) | yes | curated | verified (2026-03-04) | ready | - |
| modernbert | embedding | active | 1 (tools/configs/conversion/modernbert/modernbert-template-wf16-ef16-hf16-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| diffusion | diffusion | active | 1 (tools/configs/conversion/diffusion/diffusion-template-wf16-ef16-hf16-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| gemma2 | transformer | active | 1 (tools/configs/conversion/gemma2/gemma2-template-wf16-ef16-hf16-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| translategemma | transformer | active | 1 (tools/configs/conversion/gemma3/translategemma-4b-it-wq4k-ef16-hf16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| gemma3 | transformer | active | 8 (tools/configs/conversion/gemma3/gemma-3-1b-it-wf16-ef16-hf16-f16.json, tools/configs/conversion/gemma3/gemma-3-1b-it-wf16-ef16-hf16-f32.json, tools/configs/conversion/gemma3/gemma-3-1b-it-wq4k-ef16-hf16-f32.json, +5 more) | 1 (gemma-3-270m-it-wq4k-ef16-hf16) | yes | curated | verified (2026-03-04) | ready | - |
| llama3 | transformer | active | 1 (tools/configs/conversion/llama3/llama3-template-wf16-ef16-hf16-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| lfm2 | transformer | active | 2 (tools/configs/conversion/lfm2/lfm2.5-1.2b-instruct-wq4k-ef16-hf16-f16.json, tools/configs/conversion/lfm2/lfm2.5-1.2b-instruct-wq4k-ef16-hf16-f32.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| qwen3 | transformer | active | 1 (tools/configs/conversion/qwen3/qwen-3-5-0-8b-wf16-ef16-hf16-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| kimi_k2 | transformer | active | 1 (tools/configs/conversion/kimi-k2/kimi-k2-template-wf16-ef16-hf16-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| gpt_oss | gpt-oss | active | 1 (tools/configs/conversion/gpt-oss-20b-wf16-ef16-hf16-xmxfp4.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| deepseek | deepseek | active | 1 (tools/configs/conversion/deepseek/deepseek-template-wf16-ef16-hf16-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| mixtral | mixtral | active | 1 (tools/configs/conversion/mixtral/mixtral-template-wf16-ef16-hf16-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| mamba | mamba | blocked | 1 (tools/configs/conversion/mamba/mamba-template-wf16-ef16-hf16-f16.json) | 0 | no | none | unknown | blocked-runtime | fail-closed runtime path; not in local catalog; not verified in catalog lifecycle |
| transformer | transformer | active | 1 (tools/configs/conversion/transformer/transformer-template-wf16-ef16-hf16-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |

## Summary

- Presets tracked: 16
- Presets with conversion configs: 16
- Presets present in catalog: 2
- Ready presets (active runtime + conversion + catalog): 2
- Presets with HF-hosted catalog entries: 2
- Presets with verified catalog lifecycle: 2
- Blocked runtime presets: 1
- Catalog entries: 2

