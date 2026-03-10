# Model Support Matrix

Auto-generated from preset registry (`src/config/loader.js`), conversion configs (`tools/configs/conversion/**`), and catalog (`models/catalog.json`).
`models/catalog.json` lifecycle metadata is the canonical source for hosted/demo/tested status.
Run `npm run support:matrix:sync` after adding/changing presets, conversion configs, or catalog entries.

Updated at: 2026-03-08

## Current Inference Status

This section answers "which models work now?" from `models/catalog.json` lifecycle metadata plus the quickstart registry.

### 1. Verified

| Model ID | Preset | Modes | Last verified | Surface | Notes |
| --- | --- | --- | --- | --- | --- |
| gemma-3-270m-it-q4k-ehf16-af32 | gemma3 | run | 2026-03-04 | auto | - |
| google-embeddinggemma-300m-q4k-ehf16-af32 | embeddinggemma | embedding | 2026-03-04 | auto | - |
| translategemma-4b-it-q4k-ehf16-af32 | translategemma | run | 2026-03-06 | browser | - |

### 2. Loads But Unverified

None right now.

### 3. Known Failing

| Model ID | Preset | Modes | Last checked | Surface | Notes |
| --- | --- | --- | --- | --- | --- |
| qwen-3-5-0-8b-q4k-ehaf16 | qwen3 | run | 2026-03-06 | browser | Loads and runs but produces incoherent output. Linear attention kernel correctness not yet verified against HF reference. |
| qwen-3-5-2b-q4k-ehaf16 | qwen3 | run | 2026-03-06 | browser | Loads and runs but produces incoherent output. Same root cause as 0.8B variant — linear attention kernel correctness not yet verified. |

### 4. Quickstart-Supported Only

None right now.

### 5. Everything Else

| Entry | Type | Status | Notes |
| --- | --- | --- | --- |
| deepseek | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| diffusion | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| functiongemma | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| gemma2 | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| gpt_oss | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| janus_text | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| kimi_k2 | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| lfm2 | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| llama3 | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| mamba | preset family | blocked-runtime | runtime path is fail-closed |
| mixtral | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| modernbert | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| transformer | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |

## Preset Coverage Matrix

| Preset | Runtime modelType | Runtime | Conversion configs | Catalog models | Hosted (HF) | Demo | Tested | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| functiongemma | transformer | active | 1 (tools/configs/conversion/functiongemma/functiongemma-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| embeddinggemma | embedding | active | 2 (tools/configs/conversion/embeddinggemma/google-embeddinggemma-300m-bf16-af32.json, tools/configs/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json) | 1 (google-embeddinggemma-300m-q4k-ehf16-af32) | yes | curated | verified (2026-03-04) | verified | - |
| janus_text | transformer | active | 1 (tools/configs/conversion/janus/janus-pro-1b-text-q4k-ehaf16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| modernbert | embedding | active | 1 (tools/configs/conversion/modernbert/modernbert-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| diffusion | diffusion | active | 2 (tools/configs/conversion/diffusion/diffusion-template-f16.json, tools/configs/conversion/sana/sana-sprint-0.6b-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| gemma2 | transformer | active | 1 (tools/configs/conversion/gemma2/gemma2-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| translategemma | transformer | active | 1 (tools/configs/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json) | 1 (translategemma-4b-it-q4k-ehf16-af32) | yes | none | verified (2026-03-06) | verified | - |
| gemma3 | transformer | active | 9 (tools/configs/conversion/gemma3/gemma-3-1b-it-f16-af32.json, tools/configs/conversion/gemma3/gemma-3-1b-it-f16.json, tools/configs/conversion/gemma3/gemma-3-1b-it-q4k-ehaf16.json, +6 more) | 1 (gemma-3-270m-it-q4k-ehf16-af32) | yes | curated | verified (2026-03-04) | verified | - |
| llama3 | transformer | active | 1 (tools/configs/conversion/llama3/llama3-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| lfm2 | transformer | active | 2 (tools/configs/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehaf16.json, tools/configs/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| qwen3 | transformer | active | 3 (tools/configs/conversion/qwen3/qwen-3-5-0-8b-f16.json, tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json, tools/configs/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json) | 2 (qwen-3-5-0-8b-q4k-ehaf16, qwen-3-5-2b-q4k-ehaf16) | yes | none | failed | verification-failed | - |
| kimi_k2 | transformer | active | 1 (tools/configs/conversion/kimi-k2/kimi-k2-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| gpt_oss | gpt-oss | active | 1 (tools/configs/conversion/gpt-oss-20b-f16-xmxfp4.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| deepseek | deepseek | active | 1 (tools/configs/conversion/deepseek/deepseek-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| mixtral | mixtral | active | 1 (tools/configs/conversion/mixtral/mixtral-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| mamba | mamba | blocked | 1 (tools/configs/conversion/mamba/mamba-template-f16.json) | 0 | no | none | unknown | blocked-runtime | fail-closed runtime path; not in local catalog; not verified in catalog lifecycle |
| transformer | transformer | active | 1 (tools/configs/conversion/transformer/transformer-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |

## Summary

- Presets tracked: 17
- Presets with conversion configs: 17
- Presets present in catalog: 4
- Verified presets (active runtime + conversion + catalog + passing verification): 3
- Cataloged presets pending verification: 0
- Presets with HF-hosted catalog entries: 4
- Presets with verified catalog lifecycle: 3
- Presets with failed catalog verification: 1
- Blocked runtime presets: 1
- Catalog entries: 5

