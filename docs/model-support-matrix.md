# Model Support Matrix

Auto-generated from conversion configs (`src/config/conversion/**`) and `models/catalog.json`.
Run `npm run support:matrix:sync` after editing `models/catalog.json` or changing conversion configs.

Updated at: 2026-04-18

## Current Inference Status

This section answers "which models work now?" from `models/catalog.json` lifecycle metadata plus the quickstart registry.

### 1. Verified

| Model ID | Family | Modes | Last verified | Surface | Notes |
| --- | --- | --- | --- | --- | --- |
| gemma-3-270m-it-q4k-ehf16-af32 | gemma3 | text, vision | 2026-03-20 | browser, node | Browser/WebGPU on Apple M3 and Node/WebGPU on AMD RDNA-3 re-verified through 2026-03-20 with deterministic greedy decoding. Produces coherent sky-color answers on simple prompts. The lm_head_prefill phase-drop bug was the one confirmed runtime defect (now fixed and regression-covered). Remaining factual drift on harder prompts is a model-capacity limitation, not a runtime bug. |
| google-embeddinggemma-300m-q4k-ehf16-af32 | embeddinggemma | embedding | 2026-04-04 | node | Sentence-transformers embedding postprocessor is included in the artifact. Node/WebGPU on AMD RDNA-3 re-verified on 2026-04-04 after refreshing the manifest-owned execution-v1 session baseline and embedding postprocessor stamp. Produced finite 768-dim unit-norm embeddings and matched the local 20 retrieval / 14 pair semantic suite thresholds (0.95 retrieval, 0.6429 pair). Republished to HF 2026-04-04. |
| gemma-3-1b-it-q4k-ehf16-af32 | gemma3 | text, vision | 2026-04-13 | node | Node/WebGPU on Apple M3 re-verified on 2026-04-13 after syncing the refreshed manifest-owned session baseline and republishing the hosted artifact. Produces coherent sky-color output on the execution-inline Q4K path, and the hosted manifest now matches the refreshed local artifact. Earlier browser verification remains from the 2026-03-20 review cycle. |
| gemma-4-e2b-it-q4k-ehf16-af32 | gemma4 | text, vision | 2026-04-18 | node | Apple M3 / Node WebGPU darwin re-verified 2026-04-18 against the current local manifest; execution contract pass. Receipt: reports/gemma-4-e2b-it-q4k-ehf16-af32/2026-04-18T19-17-45.997Z.json. Previous AMD RDNA-3 re-verify 2026-04-09 after repinning Gemma 4 E2B text decode to attention_decode_online_f16kv and splitting prefill attention between sliding-window and full-attention layers. Apple M3 browser compare receipt: benchmarks/vendors/results/compare_20260415T170108.json (Doppler RDRR wins cold load ~3.4x, Transformers.js ONNX/q4f16 wins warm decode ~2.6x; product-engine comparison across different artifact formats, not a format-identical kernel benchmark). Manifest-only republish to HF 2026-04-18 at revision c690ac0274891afbddb8afb776d000e4c0051b7b. |
| gemma-4-e2b-it-q4k-ehf16-af32-int4ple | gemma4 | text, vision | 2026-04-16 | node | Node/WebGPU on AMD RDNA-3 verified on 2026-04-16 from the repo-local artifact after adding INT4 PLE conversion and range_backed per-layer input materialization. Deterministic greedy text smokes: 'Write one short sentence about the moon.' -> 'The moon orbits the Earth.'; 'Answer in one short phrase: What do bees make?' -> 'Honey'; sky-color prompt -> 'Blue'. Direct LiteRT .task source still loads but remains numerically wrong. |
| translategemma-4b-it-q4k-ehf16-af32 | translategemma | translate | 2026-03-20 | browser, node | Browser/WebGPU on Apple M3 and Node/WebGPU on AMD RDNA-3 verified through 2026-03-20 with structured TranslateGemma requests. en->fr 'Hello world.' -> 'Bonjour le monde.' The working Q4K path is gemma3-q4k-dequant-f32w-f32a-online; the prior failure remained isolated to the older q4k_dequant F16-weight path. |
| qwen-3-5-0-8b-q4k-ehaf16 | qwen3 | text, vision | 2026-04-18 | node | Re-verified 2026-04-18 on AMD Strix Halo / Radeon 8060S (RDNA-3) + RADV Mesa 26.0.3, Node/WebGPU after kernel-ref swap on the 6 full-attention layers (indices 3/7/11/15/19/23): attn_stream -> attn_head256, and q4_prefill multicol_shared -> q4_widetile (register-tiled). Warm-cache 3-run means showed prefill@80 +57.5% (sigma=3.8), prefill@15 +16.6%, decode flat (sigma=1.3); doppler verify match with the expected Qwen <think> wrapper; execution contract gate pass. Receipts live on the Strix Halo box (this darwin workspace only carries Apr 15 pre-swap receipts under reports/qwen-3-5-0-8b-q4k-ehaf16/). Previous 2026-04-17 run confirmed the Apr-17 profiles/qwen-3-5-0-8b-throughput retune (batchSize=4 readbackInterval=2 batch stop-check); 2026-04-13 Apple M3 browser verification still applies to that surface. Manifest-only republish to HF 2026-04-18 at revision 1c509e37e4430d7ffd094ce5dfa623792603670e. |
| qwen-3-5-2b-q4k-ehaf16 | qwen3 | text, vision | 2026-04-18 | browser, node | Apple M3 / Node WebGPU darwin re-verified 2026-04-18 against the current local manifest; execution contract pass. Receipt: reports/qwen-3-5-2b-q4k-ehaf16/2026-04-18T19-18-27.036Z.json. Node/WebGPU verification uses execution-v1 with fused-Q4 decode/prefill projections as the primary hybrid Qwen path and execution.inlineKernelPath=true. The earlier GEMV/head256 note was rolled back after correctness regressions on the linear-attention path. Manifest-only republish to HF 2026-04-18 at revision 3f54bc54c19181bd737d1773dcd70fb783297e73. |

### 2. Loads But Unverified

None right now.

### 3. Known Failing

None right now.

### 4. Quickstart-Supported Only

None right now.

### 5. Everything Else

| Entry | Type | Status | Notes |
| --- | --- | --- | --- |
| lfm2-5-1-2b-instruct-q4k-ehf16-af32 | catalog model | experimental | Cataloged model without a verified or failing inference lifecycle result. |
| sana-sprint-0-6b-wf16-ef16-hf16-f16 | catalog model | experimental | Cataloged model without a verified or failing inference lifecycle result. |
| gpt_oss | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| janus_text | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |

## Family Coverage Matrix

| Family | Runtime modelType | Runtime | Conversion configs | Catalog models | Hosted (HF) | Demo | Tested | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| embeddinggemma | embedding | active | 1 (src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json) | 1 (google-embeddinggemma-300m-q4k-ehf16-af32) | yes | none | verified (2026-04-04) | verified | - |
| gemma3 | transformer | active | 3 (src/config/conversion/gemma3/gemma-3-1b-it-f16-af32.json, src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json, src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json) | 2 (gemma-3-1b-it-q4k-ehf16-af32, gemma-3-270m-it-q4k-ehf16-af32) | yes | none | verified (2026-04-13) | verified | catalog verification applies only to cataloged models (2/3 conversion configs cataloged) |
| translategemma | transformer | active | 2 (src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json, src/config/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json) | 1 (translategemma-4b-it-q4k-ehf16-af32) | yes | none | verified (2026-03-20) | verified | catalog verification applies only to cataloged models (1/2 conversion configs cataloged) |
| gemma4 | transformer | active | 3 (src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32-int4ple.json, src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json, src/config/conversion/gemma4/gemma-4-moe-q4k-ehf16-af32.json) | 2 (gemma-4-e2b-it-q4k-ehf16-af32, gemma-4-e2b-it-q4k-ehf16-af32-int4ple) | yes | none | verified (2026-04-18) | verified | catalog verification applies only to cataloged models (2/3 conversion configs cataloged) |
| qwen3 | transformer | active | 2 (src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json, src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json) | 2 (qwen-3-5-0-8b-q4k-ehaf16, qwen-3-5-2b-q4k-ehaf16) | yes | none | verified (2026-04-18) | verified | - |
| lfm2 | transformer | active | 1 (src/config/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json) | 1 (lfm2-5-1-2b-instruct-q4k-ehf16-af32) | no | none | unknown | verification-pending | not verified in catalog lifecycle |
| gpt_oss | transformer | active | 1 (src/config/conversion/gpt-oss-20b-f16-xmxfp4.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| janus_text | transformer | active | 1 (src/config/conversion/janus/janus-pro-1b-text-q4k-ehaf16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| sana | diffusion | active | 1 (src/config/conversion/sana/sana-sprint-0.6b-f16.json) | 1 (sana-sprint-0-6b-wf16-ef16-hf16-f16) | no | none | unknown | verification-pending | not verified in catalog lifecycle |

## Summary

- Families tracked: 9
- Families with conversion configs: 9
- Families present in catalog: 7
- Verified families (active runtime + conversion + catalog + passing verification): 5
- Cataloged families pending verification: 2
- Families with HF-hosted catalog entries: 5
- Families with verified catalog lifecycle: 5
- Families with failed catalog verification: 0
- Blocked runtime families: 0
- Catalog entries: 10

