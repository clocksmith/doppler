# Model Support Matrix

Auto-generated from conversion configs (`src/config/conversion/**`) and `models/catalog.json`.
Run `npm run support:matrix:sync` after editing `models/catalog.json` or changing conversion configs.

Updated at: 2026-03-31

## Current Inference Status

This section answers "which models work now?" from `models/catalog.json` lifecycle metadata plus the quickstart registry.

### 1. Verified

| Model ID | Family | Modes | Last verified | Surface | Notes |
| --- | --- | --- | --- | --- | --- |
| gemma-3-270m-it-q4k-ehf16-af32 | gemma3 | text, vision | 2026-03-20 | browser, node | Browser/WebGPU on Apple M3 and Node/WebGPU on AMD RDNA-3 re-verified through 2026-03-20 with deterministic greedy decoding. Produces coherent sky-color answers on simple prompts. The lm_head_prefill phase-drop bug was the one confirmed runtime defect (now fixed and regression-covered). Remaining factual drift on harder prompts is a model-capacity limitation, not a runtime bug. |
| google-embeddinggemma-300m-q4k-ehf16-af32 | embeddinggemma | embedding | 2026-03-21 | node | Sentence-transformers embedding postprocessor is included in the artifact. Host verify on 2026-03-21 produced finite 768-dim unit-norm embeddings and matched local SafeTensors source parity on the 20 retrieval / 14 pair semantic suite (0.95 retrieval, 0.6429 pair). Republished to HF 2026-03-21. |
| gemma-3-1b-it-q4k-ehf16-af32 | gemma3 | text, vision | 2026-03-20 | browser, node | Browser/WebGPU on Apple M3 and Node/WebGPU on AMD RDNA-3 re-verified through 2026-03-20 with deterministic greedy decoding. Produces correct blue sky-color answers on simple prompts. The lm_head_prefill phase-drop bug was the one confirmed runtime defect (now fixed and regression-covered). Q4K runtime path is functionally correct. |
| translategemma-4b-it-q4k-ehf16-af32 | translategemma | translate | 2026-03-20 | browser, node | Browser/WebGPU on Apple M3 and Node/WebGPU on AMD RDNA-3 verified through 2026-03-20 with structured TranslateGemma requests. en->fr 'Hello world.' -> 'Bonjour le monde.' The working Q4K path is gemma3-q4k-dequant-f32w-f32a-online; the prior failure remained isolated to the older q4k_dequant F16-weight path. |
| qwen-3-5-0-8b-q4k-ehaf16 | qwen3 | text, vision | 2026-03-31 | browser, node | Browser/WebGPU headless Chromium on Apple M3 verified 2026-03-31: GEMV decode path (remapQ4KDecodeToGemv) at 33 tok/s, prefill remapped to dense f16. capability-transforms rule applied: useHead256PrefillAttention + remapQ4KPrefillToDense + remapQ4KDecodeToGemv. maxSeqLen capped to 8192. execution.inlineKernelPath=true. |
| qwen-3-5-2b-q4k-ehaf16 | qwen3 | text, vision | 2026-03-31 | browser, node | Browser/WebGPU headless Chromium on Apple M3 verified 2026-03-31: GEMV decode path (remapQ4KDecodeToGemv) at 17 tok/s, prefill remapped to dense f16. capability-transforms rule applied: useHead256PrefillAttention + remapQ4KPrefillToDense + remapQ4KDecodeToGemv. maxSeqLen capped to 8192. execution.inlineKernelPath=true. |
| lfm2-5-1-2b-instruct-q4k-ehf16-af32 | lfm2 | text | 2026-03-20 | node | Node/WebGPU on AMD RDNA-3 re-verified on 2026-03-20 after the ChatML fix and refreshed manifest sync. One-word sky-color prompts are stable ('Blue'), but short open-ended factual prompts remain prompt-sensitive and weaker than Gemma/Qwen. Runtime path is coherent; the refreshed manifest now includes tokenizer BOS/EOS/PAD IDs and decodeLoop=null. |

### 2. Loads But Unverified

None right now.

### 3. Known Failing

None right now.

### 4. Quickstart-Supported Only

None right now.

### 5. Everything Else

| Entry | Type | Status | Notes |
| --- | --- | --- | --- |
| sana-sprint-0-6b-wf16-ef16-hf16-f16 | catalog model | experimental | Cataloged model without a verified or failing inference lifecycle result. |
| gemma4 | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| gpt_oss | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| janus_text | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |

## Family Coverage Matrix

| Family | Runtime modelType | Runtime | Conversion configs | Catalog models | Hosted (HF) | Demo | Tested | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| embeddinggemma | embedding | active | 1 (src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json) | 1 (google-embeddinggemma-300m-q4k-ehf16-af32) | yes | none | verified (2026-03-21) | verified | - |
| gemma3 | transformer | active | 3 (src/config/conversion/gemma3/gemma-3-1b-it-f16-af32.json, src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json, src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json) | 2 (gemma-3-1b-it-q4k-ehf16-af32, gemma-3-270m-it-q4k-ehf16-af32) | yes | none | verified (2026-03-20) | verified | catalog verification applies only to cataloged models (2/3 conversion configs cataloged) |
| translategemma | transformer | active | 2 (src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json, src/config/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json) | 1 (translategemma-4b-it-q4k-ehf16-af32) | yes | none | verified (2026-03-20) | verified | catalog verification applies only to cataloged models (1/2 conversion configs cataloged) |
| gemma4 | transformer | active | 1 (src/config/conversion/gemma4/gemma-4-moe-q4k-ehf16-af32.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| qwen3 | transformer | active | 2 (src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json, src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json) | 2 (qwen-3-5-0-8b-q4k-ehaf16, qwen-3-5-2b-q4k-ehaf16) | yes | none | verified (2026-03-31) | verified | - |
| lfm2 | transformer | active | 1 (src/config/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json) | 1 (lfm2-5-1-2b-instruct-q4k-ehf16-af32) | yes | none | verified (2026-03-20) | verified | - |
| gpt_oss | transformer | active | 1 (src/config/conversion/gpt-oss-20b-f16-xmxfp4.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| janus_text | transformer | active | 1 (src/config/conversion/janus/janus-pro-1b-text-q4k-ehaf16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| sana | diffusion | active | 1 (src/config/conversion/sana/sana-sprint-0.6b-f16.json) | 1 (sana-sprint-0-6b-wf16-ef16-hf16-f16) | no | none | unknown | verification-pending | not verified in catalog lifecycle |

## Summary

- Families tracked: 9
- Families with conversion configs: 9
- Families present in catalog: 6
- Verified families (active runtime + conversion + catalog + passing verification): 5
- Cataloged families pending verification: 1
- Families with HF-hosted catalog entries: 5
- Families with verified catalog lifecycle: 5
- Families with failed catalog verification: 0
- Blocked runtime families: 0
- Catalog entries: 8

