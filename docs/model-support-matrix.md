# Model Support Matrix

Auto-generated from conversion configs (`src/config/conversion/**`) and `models/catalog.json`.
Run `npm run support:matrix:sync` after editing `models/catalog.json` or changing conversion configs.

Updated at: 2026-04-20

## Current Inference Status

This section answers "which models work now?" from `models/catalog.json` lifecycle metadata plus the quickstart registry.

### 1. Verified

| Model ID | Family | Modes | Last verified | Surface | Notes |
| --- | --- | --- | --- | --- | --- |
| gemma-3-270m-it-q4k-ehf16-af32 | gemma3 | text, vision | 2026-03-20 | browser, node | Browser/WebGPU on Apple M3 and Node/WebGPU on AMD RDNA-3 re-verified through 2026-03-20 with deterministic greedy decoding. Produces coherent sky-color answers on simple prompts. The lm_head_prefill phase-drop bug was the one confirmed runtime defect (now fixed and regression-covered). Remaining factual drift on harder prompts is a model-capacity limitation, not a runtime bug. |
| google-embeddinggemma-300m-q4k-ehf16-af32 | embeddinggemma | embedding | 2026-04-04 | node | Sentence-transformers embedding postprocessor is included in the artifact. Node/WebGPU on AMD RDNA-3 re-verified on 2026-04-04 after refreshing the manifest-owned execution-v1 session baseline and embedding postprocessor stamp. Produced finite 768-dim unit-norm embeddings and matched the local 20 retrieval / 14 pair semantic suite thresholds (0.95 retrieval, 0.6429 pair). Republished to HF 2026-04-04. |
| gemma-3-1b-it-q4k-ehf16-af32 | gemma3 | text, vision | 2026-04-13 | node | Node/WebGPU on Apple M3 re-verified on 2026-04-13 after syncing the refreshed manifest-owned session baseline and republishing the hosted artifact. Produces coherent sky-color output on the execution-inline Q4K path, and the hosted manifest now matches the refreshed local artifact. Earlier browser verification remains from the 2026-03-20 review cycle. |
| gemma-4-e2b-it-q4k-ehf16-af32 | gemma4 | text, vision | 2026-04-21 | browser, node | Browser/WebGPU on AMD Strix Halo / Radeon 8060S re-verified 2026-04-21 with compare-owned Gemma 4 prompt rendering. Compare receipt benchmarks/vendors/results/compare_20260421T001902.json shows valid non-empty output and Doppler decode throughput ahead of the paired Transformers.js ONNX/q4f16 runner on warm OPFS (15.49 vs 10.58 tok/s), but exact generated text still mismatches TJS, so this is performance evidence rather than correctness-parity evidence. Apple M3 / Node WebGPU darwin execution contract pass remains from reports/gemma-4-e2b-it-q4k-ehf16-af32/2026-04-18T19-17-45.997Z.json. Full artifact republished to HF 2026-04-21 at revision 2070777c77047d54e6eae105f6dcb1891cf6f21a. |
| gemma-4-e2b-it-q4k-ehf16-af32-int4ple | gemma4 | text, vision | 2026-04-20 | browser, node | Browser/WebGPU on AMD Strix Halo / Radeon 8060S re-verified 2026-04-20 with release decode cadence. Compare receipts benchmarks/vendors/results/compare_20260420T162233.json (t=0 greedy) and benchmarks/vendors/results/compare_20260420T163016.json (t=1 topK50) show Doppler decode throughput ahead of the paired Transformers.js ONNX/q4f16 runner on warm OPFS. Node/WebGPU on AMD RDNA-3 was previously verified from the repo-local INT4 PLE artifact with deterministic greedy text smokes. Direct LiteRT .task source still loads but remains numerically wrong. |
| translategemma-4b-it-q4k-ehf16-af32 | translategemma | translate | 2026-03-20 | browser, node | Browser/WebGPU on Apple M3 and Node/WebGPU on AMD RDNA-3 verified through 2026-03-20 with structured TranslateGemma requests. en->fr 'Hello world.' -> 'Bonjour le monde.' The working Q4K path is gemma3-q4k-dequant-f32w-f32a-online; the prior failure remained isolated to the older q4k_dequant F16-weight path. |
| qwen-3-5-0-8b-q4k-ehaf16 | qwen3 | text, vision | 2026-04-21 | browser, node | Browser/WebGPU on AMD Strix Halo / Radeon 8060S re-verified 2026-04-21 from the repaired Clocksmith/rdrr hosted manifest. Compare receipt benchmarks/vendors/results/compare_20260421T002103.json shows exact output match against the paired Transformers.js runner, 64/64 prompt tokens, Doppler decode 61.76 vs TJS 36.14 tok/s, and TTFT 541.0 ms vs 7019.3 ms. Earlier 2026-04-18 Node/WebGPU verification after the Qwen prefill kernel-ref swap remains valid; execution contract gate pass. Full artifact republished to HF 2026-04-21 at revision cf02075803e0a8de7ab26bd76888f55ded35ac5f. |
| qwen-3-5-2b-q4k-ehaf16 | qwen3 | text, vision | 2026-04-21 | browser, node | Browser/WebGPU on AMD Strix Halo / Radeon 8060S re-verified 2026-04-21 from the repaired Clocksmith/rdrr hosted manifest; execution contract pass with manifest-owned decodeBatchSize=12. Current execution-v1 uses the fixed fused-Q4 main_gemv transformer decode path, keeps stable fused-Q4 main_multicol declared as fallback, upgrades the tied LM head to Q4 via optimized lm_head_q4 GEMV 64x4, upgrades prefill projections to q4_widetile, and upgrades prefill attention to attn_head256. Compare receipt benchmarks/vendors/results/compare_20260421T002238.json shows exact output match against the paired Transformers.js runner, 64/64 prompt tokens, Doppler decode 50.00 vs TJS 32.66 tok/s, and TTFT 602.9 ms vs 7297.8 ms; run with --allow-non-comparable-lane because the lane remains capability_only. Full artifact republished to HF 2026-04-21 at revision a8c45dd885a789042d3b82c95b471d66ca8d5152. |

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
| gpt_oss | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| janus_text | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |

## Family Coverage Matrix

| Family | Runtime modelType | Runtime | Conversion configs | Catalog models | Hosted (HF) | Demo | Tested | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| embeddinggemma | embedding | active | 1 (src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json) | 1 (google-embeddinggemma-300m-q4k-ehf16-af32) | yes | none | verified (2026-04-04) | verified | - |
| gemma3 | transformer | active | 3 (src/config/conversion/gemma3/gemma-3-1b-it-f16-af32.json, src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json, src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json) | 2 (gemma-3-1b-it-q4k-ehf16-af32, gemma-3-270m-it-q4k-ehf16-af32) | yes | none | verified (2026-04-13) | verified | catalog verification applies only to cataloged models (2/3 conversion configs cataloged) |
| translategemma | transformer | active | 2 (src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json, src/config/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json) | 1 (translategemma-4b-it-q4k-ehf16-af32) | yes | none | verified (2026-03-20) | verified | catalog verification applies only to cataloged models (1/2 conversion configs cataloged) |
| gemma4 | transformer | active | 3 (src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32-int4ple.json, src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json, src/config/conversion/gemma4/gemma-4-moe-q4k-ehf16-af32.json) | 2 (gemma-4-e2b-it-q4k-ehf16-af32, gemma-4-e2b-it-q4k-ehf16-af32-int4ple) | yes | none | verified (2026-04-21) | verified | catalog verification applies only to cataloged models (2/3 conversion configs cataloged) |
| qwen3 | transformer | active | 2 (src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json, src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json) | 2 (qwen-3-5-0-8b-q4k-ehaf16, qwen-3-5-2b-q4k-ehaf16) | yes | none | verified (2026-04-21) | verified | - |
| lfm2 | transformer | active | 1 (src/config/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json) | 1 (lfm2-5-1-2b-instruct-q4k-ehf16-af32) | no | none | unknown | verification-pending | not verified in catalog lifecycle |
| gpt_oss | transformer | active | 1 (src/config/conversion/gpt-oss-20b-f16-xmxfp4.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| janus_text | transformer | active | 1 (src/config/conversion/janus/janus-pro-1b-text-q4k-ehaf16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |

## Summary

- Families tracked: 8
- Families with conversion configs: 8
- Families present in catalog: 6
- Verified families (active runtime + conversion + catalog + passing verification): 5
- Cataloged families pending verification: 2
- Families with HF-hosted catalog entries: 5
- Families with verified catalog lifecycle: 5
- Families with failed catalog verification: 0
- Blocked runtime families: 0
- Catalog entries: 10
