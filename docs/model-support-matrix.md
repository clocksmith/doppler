# Model Support Matrix

Auto-generated from preset registry (`src/config/loader.js`), conversion configs (`tools/configs/conversion/**`), and the repo catalog mirror (`models/catalog.json`).
Hosted/demo/tested lifecycle metadata is canonically owned by the external support registry and mirrored into `models/catalog.json` for repo checks.
Run `npm run support:matrix:sync` after syncing the repo catalog mirror or changing presets/conversion configs.

Updated at: 2026-03-16

## Current Inference Status

This section answers "which models work now?" from `models/catalog.json` lifecycle metadata plus the quickstart registry.

### 1. Verified

| Model ID | Preset | Modes | Last verified | Surface | Notes |
| --- | --- | --- | --- | --- | --- |
| gemma-3-270m-it-q4k-ehf16-af32 | gemma3 | run | 2026-03-12 | - | Deterministic greedy (temperature=0, topK=1) browser/WebGPU on Apple M3. Produces coherent output on appropriate prompts. The lm_head_prefill phase-drop bug was the one confirmed runtime defect (now fixed and regression-covered). Remaining factual-question inaccuracy on hard prompts is a 270M model capacity limitation, not a code bug — confirmed by 2x2 comparison showing F16 270M is also wrong. |
| google-embeddinggemma-300m-q4k-ehf16-af32 | embeddinggemma | embedding | 2026-03-13 | - | queryPreAttnScalar fixed from 16 (sqrt(headDim)) to 256 (headDim). 768-dim embeddings, all finite, semantic retrieval test passed. Republished to HF 2026-03-14. |
| gemma-3-1b-it-q4k-ehf16-af32 | gemma3 | run | 2026-03-12 | - | Deterministic greedy (temperature=0, topK=1) browser/WebGPU on Apple M3. Correctly produces A. Blue on factual sky-color prompt. The lm_head_prefill phase-drop bug was the one confirmed runtime defect (now fixed and regression-covered). Q4K runtime path is functionally correct. |
| gemma-3-1b-it-f16-af32 | gemma3 | run | 2026-03-10 | - | Prompt: 'The color of the sky is' → ' a constant, but its appearance changes with the seasons.\n\nThe sky is blue during the summer, when the sun is high in the sky.\nThe sky is red during the autumn, when the leaves change color.\nThe sky is gray during the winter, when the sun is low in the sky.\nThe' (64 tokens, temperature=0, topK=1). Fluent and stable; factual quality mixed but coherent. All contracts pass. |
| translategemma-4b-1b-enes-q4k-ehf16-af32 | gemma3 | run | 2026-03-15 | - | Node/WebGPU AMD RDNA-3. Deterministic greedy (temp=0, topK=1). Prompt: EN->ES 'The weather is nice today.' -> 'El tiempo está agradable hoy.' (7 tokens including <end_of_turn>). Correct translation, coherent output. All execution-v0 contract checks pass. |
| qwen-3-5-0-8b-q4k-ehaf16 | qwen3_5 | run | 2026-03-14 | - | Node/WebGPU AMD RDNA-3. Deterministic greedy (temp=0, topK=1). Prompt: "The color of the sky is" -> coherent factual response about blue sky / atmospheric scattering. ropeTheta fixed from 1M to 10M, manifest RoPE fields (mropeInterleaved, mropeSection, partialRotaryFactor) patched. |
| qwen-3-5-2b-q4k-ehaf16 | qwen3_5 | run | 2026-03-14 | - | Node/WebGPU AMD RDNA-3. Deterministic greedy (temp=0, topK=1). Prompt: "The color of the sky is" -> coherent factual response about blue sky / atmospheric scattering. ropeTheta fixed from 1M to 10M, manifest RoPE fields (mropeInterleaved, mropeSection, partialRotaryFactor) patched. |
| lfm2-5-1-2b-instruct-q4k-ehf16-af32 | lfm2 | run | 2026-03-16 | - | Coherence-verified 2026-03-16 on Node/WebGPU AMD RDNA-3 after applying ChatML template fix. Manifest was missing chatTemplate.type/enabled — model was receiving raw unformatted text. With chatml applied: deterministic greedy (temp=0, topK=1). Prompt: 'What color is the sky on a clear day?' (ChatML) -> 'The color of the sky on a clear day is white. This is because the atmosphere is not obsceded by any particles or objects that are in between.' (32 tokens). Output is coherent and responsive to prompt. No execution-v0 schema in manifest (known LFM2 conversion limitation; N/A for this model). Contract checks 2/2 pass. |

### 2. Loads But Unverified

None right now.

### 3. Known Failing

| Model ID | Preset | Modes | Last checked | Surface | Notes |
| --- | --- | --- | --- | --- | --- |
| translategemma-4b-it-q4k-ehf16-af32 | translategemma | run | 2026-03-15 | - | Reconverted 2026-03-14 with f32 compute. Re-verified 2026-03-15 on Node/WebGPU AMD RDNA-3. Output incoherent: garbled multilingual gibberish ('has سوریistisson...IsIsIs'). Execution-v0 contract passes. Gemma 3 4B architecture has never been verified in Doppler (only 270m and 1B verified). Root cause unknown: suspect 4B architecture-level issue (RoPE linear scaling factor=8, or scale-dependent bug). Old gemma-3-4b-it conversion also untestable (stale kernel digests). |

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
| llama3 | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| mamba | preset family | blocked-runtime | runtime path is fail-closed |
| mixtral | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| modernbert | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| qwen3 | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| transformer | preset family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |

## Preset Coverage Matrix

| Preset | Runtime modelType | Runtime | Conversion configs | Catalog models | Hosted (HF) | Demo | Tested | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| functiongemma | transformer | active | 1 (tools/configs/conversion/functiongemma/functiongemma-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| embeddinggemma | embedding | active | 2 (tools/configs/conversion/embeddinggemma/google-embeddinggemma-300m-bf16-af32.json, tools/configs/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json) | 1 (google-embeddinggemma-300m-q4k-ehf16-af32) | yes | none | verified (2026-03-13) | verified | catalog verification applies only to cataloged models (1/2 conversion configs cataloged) |
| janus_text | transformer | active | 1 (tools/configs/conversion/janus/janus-pro-1b-text-q4k-ehaf16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| modernbert | embedding | active | 1 (tools/configs/conversion/modernbert/modernbert-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| diffusion | diffusion | active | 2 (tools/configs/conversion/diffusion/diffusion-template-f16.json, tools/configs/conversion/sana/sana-sprint-0.6b-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| gemma2 | transformer | active | 1 (tools/configs/conversion/gemma2/gemma2-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| translategemma | transformer | active | 2 (tools/configs/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json, tools/configs/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json) | 1 (translategemma-4b-it-q4k-ehf16-af32) | yes | none | failed | verification-failed | catalog verification applies only to cataloged models (1/2 conversion configs cataloged) |
| gemma3 | transformer | active | 9 (tools/configs/conversion/gemma3/gemma-3-1b-it-f16-af32.json, tools/configs/conversion/gemma3/gemma-3-1b-it-f16.json, tools/configs/conversion/gemma3/gemma-3-1b-it-q4k-ehaf16.json, +6 more) | 4 (gemma-3-1b-it-f16-af32, gemma-3-1b-it-q4k-ehf16-af32, gemma-3-270m-it-q4k-ehf16-af32, +1 more) | yes | none | verified (2026-03-15) | verified | catalog verification applies only to cataloged models (4/9 conversion configs cataloged) |
| llama3 | transformer | active | 1 (tools/configs/conversion/llama3/llama3-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| lfm2 | transformer | active | 2 (tools/configs/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehaf16.json, tools/configs/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json) | 1 (lfm2-5-1-2b-instruct-q4k-ehf16-af32) | yes | none | verified (2026-03-16) | verified | catalog verification applies only to cataloged models (1/2 conversion configs cataloged) |
| qwen3_5 | transformer | active | 4 (tools/configs/conversion/qwen3/qwen-3-5-0-8b-f16.json, tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16-af32.json, tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json, +1 more) | 2 (qwen-3-5-0-8b-q4k-ehaf16, qwen-3-5-2b-q4k-ehaf16) | yes | none | verified (2026-03-14) | verified | catalog verification applies only to cataloged models (2/4 conversion configs cataloged) |
| qwen3 | transformer | active | 1 (tools/configs/conversion/qwen3/qwen3-template-q4k.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| kimi_k2 | transformer | active | 1 (tools/configs/conversion/kimi-k2/kimi-k2-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| gpt_oss | gpt-oss | active | 1 (tools/configs/conversion/gpt-oss-20b-f16-xmxfp4.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| deepseek | deepseek | active | 1 (tools/configs/conversion/deepseek/deepseek-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| mixtral | mixtral | active | 1 (tools/configs/conversion/mixtral/mixtral-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| mamba | mamba | blocked | 1 (tools/configs/conversion/mamba/mamba-template-f16.json) | 0 | no | none | unknown | blocked-runtime | fail-closed runtime path; not in local catalog; not verified in catalog lifecycle |
| transformer | transformer | active | 1 (tools/configs/conversion/transformer/transformer-template-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |

## Summary

- Presets tracked: 18
- Presets with conversion configs: 18
- Presets present in catalog: 5
- Verified presets (active runtime + conversion + catalog + passing verification): 4
- Cataloged presets pending verification: 0
- Presets with HF-hosted catalog entries: 5
- Presets with verified catalog lifecycle: 4
- Presets with failed catalog verification: 1
- Blocked runtime presets: 1
- Catalog entries: 9

