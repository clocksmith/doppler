# Model Support Matrix

Auto-generated from conversion configs (`src/config/conversion/**`) and `models/catalog.json`.
Run `npm run support:matrix:sync` after editing `models/catalog.json` or changing conversion configs.

Updated at: 2026-03-19

## Current Inference Status

This section answers "which models work now?" from `models/catalog.json` lifecycle metadata plus the quickstart registry.

### 1. Verified

| Model ID | Family | Modes | Last verified | Surface | Notes |
| --- | --- | --- | --- | --- | --- |
| gemma-3-270m-it-q4k-ehf16-af32 | gemma3 | run | 2026-03-12 | browser | Deterministic greedy (temperature=0, topK=1) browser/WebGPU on Apple M3. Produces coherent output on appropriate prompts. The lm_head_prefill phase-drop bug was the one confirmed runtime defect (now fixed and regression-covered). Remaining factual-question inaccuracy on hard prompts is a 270M model capacity limitation, not a code bug — confirmed by 2x2 comparison showing F16 270M is also wrong. |
| google-embeddinggemma-300m-q4k-ehf16-af32 | embeddinggemma | embedding | 2026-03-13 | auto | queryPreAttnScalar fixed from 16 (sqrt(headDim)) to 256 (headDim). 768-dim embeddings, all finite, semantic retrieval test passed. Republished to HF 2026-03-14. |
| gemma-3-1b-it-q4k-ehf16-af32 | gemma3 | run | 2026-03-12 | browser | Deterministic greedy (temperature=0, topK=1) browser/WebGPU on Apple M3. Correctly produces A. Blue on factual sky-color prompt. The lm_head_prefill phase-drop bug was the one confirmed runtime defect (now fixed and regression-covered). Q4K runtime path is functionally correct. |
| gemma-3-1b-it-f16-af32 | gemma3 | run | 2026-03-10 | browser | Prompt: 'The color of the sky is' → ' a constant, but its appearance changes with the seasons.\n\nThe sky is blue during the summer, when the sun is high in the sky.\nThe sky is red during the autumn, when the leaves change color.\nThe sky is gray during the winter, when the sun is low in the sky.\nThe' (64 tokens, temperature=0, topK=1). Fluent and stable; factual quality mixed but coherent. All contracts pass. |
| translategemma-4b-it-q4k-ehf16-af32 | translategemma | run, translate | 2026-03-17 | browser | Browser/WebGPU verified on 2026-03-17 with deterministic greedy decoding. The working Q4K path is gemma3-q4k-dequant-f32w-f32a-online (F32 matmul weights, F32 activations, F16 KV). Output: 'Bonjour, monde.' for the default en->fr Hello world prompt. The prior failure was isolated to the older q4k_dequant F16-weight path; F16 artifact and HF reference were both coherent during the root-cause diff. |
| translategemma-4b-1b-enes-q4k-ehf16-af32 | gemma3 | run, translate | 2026-03-15 | node | Node/WebGPU AMD RDNA-3. Deterministic greedy (temp=0, topK=1). Prompt: EN->ES 'The weather is nice today.' -> 'El tiempo está agradable hoy.' (7 tokens including <end_of_turn>). Correct translation, coherent output. All execution-v0 contract checks pass. |
| qwen-3-5-0-8b-q4k-ehaf16 | qwen3 | run | 2026-03-14 | node | Node/WebGPU AMD RDNA-3. Deterministic greedy (temp=0, topK=1). Prompt: "The color of the sky is" -> coherent factual response about blue sky / atmospheric scattering. ropeTheta fixed from 1M to 10M, manifest RoPE fields (mropeInterleaved, mropeSection, partialRotaryFactor) patched. |
| qwen-3-5-2b-q4k-ehaf16 | qwen3 | run | 2026-03-14 | node | Node/WebGPU AMD RDNA-3. Deterministic greedy (temp=0, topK=1). Prompt: "The color of the sky is" -> coherent factual response about blue sky / atmospheric scattering. ropeTheta fixed from 1M to 10M, manifest RoPE fields (mropeInterleaved, mropeSection, partialRotaryFactor) patched. |
| lfm2-5-1-2b-instruct-q4k-ehf16-af32 | lfm2 | run | 2026-03-16 | node | Coherence-verified 2026-03-16 on Node/WebGPU AMD RDNA-3 after applying ChatML template fix. Manifest was missing chatTemplate.type/enabled — model was receiving raw unformatted text. With chatml applied: deterministic greedy (temp=0, topK=1). Prompt: 'What color is the sky on a clear day?' (ChatML) -> 'The color of the sky on a clear day is white. This is because the atmosphere is not obsceded by any particles or objects that are in between.' (32 tokens). Output is coherent and responsive to prompt. No execution-v0 schema in manifest (known LFM2 conversion limitation; N/A for this model). Contract checks 2/2 pass. |

### 2. Loads But Unverified

None right now.

### 3. Known Failing

None right now.

### 4. Quickstart-Supported Only

None right now.

### 5. Everything Else

| Entry | Type | Status | Notes |
| --- | --- | --- | --- |
| gemma4 | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| gpt_oss | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| janus_text | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| sana | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |

## Family Coverage Matrix

| Family | Runtime modelType | Runtime | Conversion configs | Catalog models | Hosted (HF) | Demo | Tested | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| embeddinggemma | embedding | active | 1 (src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json) | 1 (google-embeddinggemma-300m-q4k-ehf16-af32) | yes | none | verified (2026-03-13) | verified | - |
| gemma3 | transformer | active | 3 (src/config/conversion/gemma3/gemma-3-1b-it-f16-af32.json, src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json, src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json) | 4 (gemma-3-1b-it-f16-af32, gemma-3-1b-it-q4k-ehf16-af32, gemma-3-270m-it-q4k-ehf16-af32, +1 more) | yes | none | verified (2026-03-15) | verified | - |
| translategemma | transformer | active | 2 (src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json, src/config/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json) | 1 (translategemma-4b-it-q4k-ehf16-af32) | yes | none | verified (2026-03-17) | verified | catalog verification applies only to cataloged models (1/2 conversion configs cataloged) |
| gemma4 | transformer | active | 1 (src/config/conversion/gemma4/gemma-4-moe-q4k-ehf16-af32.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| qwen3 | transformer | active | 2 (src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json, src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json) | 2 (qwen-3-5-0-8b-q4k-ehaf16, qwen-3-5-2b-q4k-ehaf16) | yes | none | verified (2026-03-14) | verified | - |
| lfm2 | transformer | active | 1 (src/config/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json) | 1 (lfm2-5-1-2b-instruct-q4k-ehf16-af32) | yes | none | verified (2026-03-16) | verified | - |
| gpt_oss | transformer | active | 1 (src/config/conversion/gpt-oss-20b-f16-xmxfp4.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| janus_text | transformer | active | 1 (src/config/conversion/janus/janus-pro-1b-text-q4k-ehaf16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| sana | diffusion | active | 1 (src/config/conversion/sana/sana-sprint-0.6b-f16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |

## Summary

- Families tracked: 9
- Families with conversion configs: 9
- Families present in catalog: 5
- Verified families (active runtime + conversion + catalog + passing verification): 5
- Cataloged families pending verification: 0
- Families with HF-hosted catalog entries: 5
- Families with verified catalog lifecycle: 5
- Families with failed catalog verification: 0
- Blocked runtime families: 0
- Catalog entries: 9

