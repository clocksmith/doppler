# Model Support Matrix

Auto-generated from conversion configs (`src/config/conversion/**`), `models/catalog.json`, and `models/gemma4-targets.json`.
Run `npm run support:matrix:sync` after editing `models/catalog.json`, `models/gemma4-targets.json`, or changing conversion configs.

Updated at: 2026-06-15

## Current Inference Status

This section answers "which models work now?" from `models/catalog.json` lifecycle metadata plus the quickstart registry.

### 1. Verified

| Model ID | Family | Modes | Last verified | Surface | Notes |
| --- | --- | --- | --- | --- | --- |
| gemma-3-270m-it-q4k-ehf16-af32 | gemma3 | text, vision | 2026-03-20 | browser, node | Browser/WebGPU on Apple M3 and Node/WebGPU on AMD RDNA-3 re-verified through 2026-03-20 with deterministic greedy decoding. Produces coherent sky-color answers on simple prompts. The lm_head_prefill phase-drop bug was the one confirmed runtime defect (now fixed and regression-covered). Remaining factual drift on harder prompts is a model-capacity limitation, not a runtime bug. |
| google-embeddinggemma-300m-q4k-ehf16-af32 | embeddinggemma | embedding | 2026-04-04 | node | Sentence-transformers embedding postprocessor is included in the artifact. Node/WebGPU on AMD RDNA-3 re-verified on 2026-04-04 after refreshing the manifest-owned execution-v1 session baseline and embedding postprocessor stamp. Produced finite 768-dim unit-norm embeddings and matched the local 20 retrieval / 14 pair semantic suite thresholds (0.95 retrieval, 0.6429 pair). Republished to HF 2026-04-04. |
| gemma-3-1b-it-q4k-ehf16-af32 | gemma3 | text, vision | 2026-04-13 | node | Node/WebGPU on Apple M3 re-verified on 2026-04-13 after syncing the refreshed manifest-owned session baseline and republishing the hosted artifact. Produces coherent sky-color output on the execution-inline Q4K path, and the hosted manifest now matches the refreshed local artifact. Earlier browser verification remains from the 2026-03-20 review cycle. |
| gemma-4-e2b-it-q4k-ehf16-af32 | gemma4 | text, vision | 2026-05-07 | node | Node/WebGPU execution-contract evidence is present in reports/gemma-4-e2b-it-q4k-ehf16-af32/2026-05-07T14-35-06.195Z.json. Browser compare receipt benchmarks/vendors/results/compare_20260421T001902.json shows valid non-empty output and warm OPFS performance evidence, but exact generated text still mismatches TJS, so browser remains performance evidence rather than verified support until a browser runtime pass receipt is listed. Full artifact republished to HF 2026-04-21 at revision 2070777c77047d54e6eae105f6dcb1891cf6f21a. |
| gemma-4-e2b-it-q4k-ehf16-af32-int4ple | gemma4 | text, vision | 2026-05-07 | node | Node/WebGPU execution-contract evidence is present in reports/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/2026-05-07T17-03-36.276Z.json. Browser compare receipts benchmarks/vendors/results/compare_20260420T162233.json (t=0 greedy) and benchmarks/vendors/results/compare_20260420T163016.json (t=1 topK50) remain performance evidence only until a browser runtime pass receipt is listed. Direct LiteRT .task source still loads but remains numerically wrong. |
| gemma-4-e2b-it-q4k-ehf16-af16-int4ple | gemma4 | text, vision | 2026-05-07 | node | Manifest-only f16 activation sibling over the verified Gemma 4 E2B INT4 PLE weight pack. Node/WebGPU execution-contract evidence is present in reports/gemma-4-e2b-it-q4k-ehf16-af16-int4ple/2026-05-07T17-14-52.263Z.json. Browser remains unverified until a browser runtime pass receipt is listed. The demo preflights execution-v1 capability rules and uses the AF32 primary when this lane is rejected, including the Apple Metal fused-q4k/f16 NaN guard. |
| translategemma-4b-it-q4k-ehf16-af32 | translategemma | translate | 2026-03-20 | browser, node | Browser/WebGPU on Apple M3 and Node/WebGPU on AMD RDNA-3 verified through 2026-03-20 with structured TranslateGemma requests. en->fr 'Hello world.' -> 'Bonjour le monde.' The working Q4K path is gemma3-q4k-dequant-f32w-f32a-online; the prior failure remained isolated to the older q4k_dequant F16-weight path. |
| gemma-4-12b-it-text-w4a16-ct-ehf16-af16 | gemma4 | text | 2026-06-08 | node | Node/WebGPU verified from /home/x/models/rdrr/gemma-4-12b-it-text-w4a16-ct-ehf16-af16 with deterministic greedy output. Report reports/gemma-4-12b-it-text-w4a16-ct-ehf16-af16/2026-06-08T22-33-51.728Z.json produced coherent text, executionContractOk=true, manifest-owned decodeBatchSize=8/readbackInterval=8, and batched decode only with 2 batched forward calls and 0 unbatched calls. |
| gemma-4-31b-it-text-q4k-ehf16-af32 | gemma4 | text | 2026-04-29 | browser | Complete Q4K weight pack is the hosted primary for the Gemma 4 31B f16 demo lane. Browser/WebGPU debug receipt reports/gemma-4-31b-it-text-q4k-ehf16-af32/2026-04-28T22-40-53.165Z.json has executionContractOk=true; a committed Node receipt is still missing. |
| gemma-4-31b-it-text-q4k-ehf16-af16 | gemma4 | text | 2026-04-29 | browser | Browser/WebGPU report reports/gemma-4-31b-it-text-q4k-ehf16-af16/2026-04-29T17-25-41.735Z.json and program-bundle references both decode the sky prompt to blue with f16 compute. A committed Node receipt is still missing. |
| qwen-3-5-0-8b-q4k-ehaf16 | qwen3 | text | 2026-06-15 | browser, node | Node/WebGPU on AMD Strix Halo / Radeon 8060S re-verified 2026-06-15 after making Qwen ChatML stop tokens explicit in the manifest and preserving batch-path stopTokenId receipts. Local receipt reports/qwen-3-5-0-8b-q4k-ehaf16/2026-06-15T00-06-08.527Z.json produced "The sky is blue.", stopReason=stop-token, stopTokenId=248046, executionContractOk=true. Existing comparable claim still rests on the 2026-04-21 vendor compare receipt benchmarks/vendors/results/compare_20260421T002103.json: exact output match against Transformers.js, 64/64 prompt tokens, Doppler decode 61.76 vs TJS 36.14 tok/s, TTFT 541.0 ms vs 7019.3 ms. Hosted Clocksmith/rdrr revision 95a01447eecbf13fc5964986f507b08ded0cd40f still exposes scalar eos_token_id=248044 rather than [248046, 248044], so HF/quickstart availability is disabled until republished. The local artifact includes qwen3vl vision weights, but release-facing modes stay text-only until Qwen vision has an end-to-end receipt. |
| qwen-3-5-2b-q4k-ehaf16 | qwen3 | text | 2026-06-15 | browser, node | Node/WebGPU on AMD Strix Halo / Radeon 8060S re-verified 2026-06-15 after making Qwen ChatML stop tokens explicit in the manifest. Local receipt reports/qwen-3-5-2b-q4k-ehaf16/2026-06-15T00-07-18.449Z.json produced a coherent sky-color answer, executionContractOk=true, required inference fields pass, and decode 34.48 tok/s on a 24-token capped run. Current execution-v1 uses the fixed fused-Q4 main_gemv transformer decode path, keeps stable fused-Q4 main_multicol declared as fallback, upgrades the tied LM head to Q4 via optimized lm_head_q4 GEMV 64x4, upgrades prefill projections to q4_widetile, and upgrades prefill attention to attn_head256. The 2026-04-21 compare receipt benchmarks/vendors/results/compare_20260421T002238.json showed exact output match against Transformers.js with --allow-non-comparable-lane; the release lane remains capability_only until a correctness-clean comparable fixture is committed. Hosted Clocksmith/rdrr revision a8c45dd885a789042d3b82c95b471d66ca8d5152 still exposes scalar eos_token_id=248044 rather than [248046, 248044], so HF/quickstart availability is disabled until republished. |
| qwen-3-6-27b-q4k-ehaf16 | qwen3 | text | 2026-04-28 | browser | Local browser/WebGPU smoke report reports/qwen-3-6-27b-q4k-ehaf16/2026-04-28T01-19-10.624Z.json captured a 4-token deterministic reference transcript and executionContractOk=true. Hosted artifact is enabled for web demo download; quickstart remains disabled. |
| qwen-3-6-27b-q4k-eaf16 | qwen3 | text | 2026-04-29 | browser, node | Browser/WebGPU report reports/qwen-3-6-27b-q4k-eaf16/2026-04-29T17-28-17.095Z.json and the Node program-bundle reference produce coherent sky prompt output with f16 compute. |

### 2. Loads But Unverified

None right now.

### 3. Known Failing

None right now.

### 4. Quickstart-Supported Only

None right now.

### 5. Everything Else

| Entry | Type | Status | Notes |
| --- | --- | --- | --- |
| gemma-4-12b-it-text-q4k-ehf16-af32 | catalog model | experimental | Cataloged model without a verified or failing inference lifecycle result. |
| gemma-4-12b-it-text-q4k-ehf16-af16 | catalog model | experimental | Cataloged model without a verified or failing inference lifecycle result. |
| diffusiongemma-26b-a4b-it-q4k-ehf16-af16 | catalog model | experimental | Cataloged model without a verified or failing inference lifecycle result. |
| lfm2-5-1-2b-instruct-q4k-ehf16-af32 | catalog model | experimental | Cataloged model without a verified or failing inference lifecycle result. |
| gpt_oss | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| janus_text | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |

## Gemma 4 Target Coverage

Generated from `models/gemma4-targets.json`. This section tracks the latest official Gemma 4 target set separately from the catalog, so unsupported or unverified targets stay visible.

| Target | Doppler status | Browser | Electron | Node | Serve | Official MTP | Doppler MTP | Runtime receipts | Benchmark receipts | Serve receipts | Preflight receipts | Current lanes | Source packages | Missing | Blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 E2B | partially_verified | unverified | unverified | verified | verified | yes | not_implemented | gemma-4-e2b-it-q4k-ehf16-af32 (node, pass)<br>gemma-4-e2b-it-q4k-ehf16-af32-int4ple (node, pass)<br>gemma-4-e2b-it-q4k-ehf16-af16-int4ple (node, pass) | gemma-4-e2b-it-q4k-ehf16-af32 (browser, performance_evidence)<br>gemma-4-e2b-it-q4k-ehf16-af32-int4ple (browser, performance_evidence)<br>gemma-4-e2b-it-q4k-ehf16-af16-int4ple (node, diagnostic) | gemma-4-e2b-it-q4k-ehf16-af32-int4ple (serve, pass)<br>gemma-4-e2b-it-q4k-ehf16-af32-int4ple (serve, diagnostic) | gemma-4-e2b-it-q4k-ehf16-af32 (node, pass)<br>gemma-4-e2b-it-q4k-ehf16-af32-int4ple (node, pass)<br>gemma-4-e2b-it-q4k-ehf16-af16-int4ple (node, pass) | gemma-4-e2b-it-q4k-ehf16-af32 (verified)<br>gemma-4-e2b-it-q4k-ehf16-af32-int4ple (verified)<br>gemma-4-e2b-it-q4k-ehf16-af16-int4ple (verified-local) | none | browser runtime pass receipt<br>hosted quickstart artifact refresh<br>electron receipt<br>mtp lane<br>full multimodal browser receipt | browser-runtime-pass-receipt-missing (browser, unverified)<br>hosted-quickstart-artifact-stale (serve, diagnostic)<br>electron-receipt-missing (electron, unverified)<br>mtp-lane-not-implemented (mtp, not_implemented)<br>multimodal-browser-receipt-incomplete (browser, incomplete) |
| Gemma 4 E4B | gap | unsupported | unsupported | unsupported | unsupported | yes | not_implemented | none | none | none | none | none | litert/gemma-4-e4b-it (blocked) | conversion config<br>catalog model<br>RDRR artifact<br>runtime receipt<br>benchmark receipt<br>mtp lane | browser-runtime-unsupported (browser, unsupported)<br>electron-runtime-unsupported (electron, unsupported)<br>node-runtime-unsupported (node, unsupported)<br>serve-runtime-unsupported (serve, unsupported)<br>conversion-config-missing (model, missing)<br>catalog-model-missing (model, missing)<br>rdrr-artifact-missing (model, missing)<br>runtime-receipt-missing (model, missing)<br>benchmark-receipt-missing (benchmark, missing)<br>mtp-lane-not-implemented (mtp, not_implemented) |
| Gemma 4 12B Unified | partially_verified | unverified | unverified | verified | unsupported | yes | not_implemented | gemma-4-12b-it-text-w4a16-ct-ehf16-af16 (node, pass) | gemma-4-12b-it-text-w4a16-ct-ehf16-af16 (browser, diagnostic) | none | none | gemma-4-12b-it-text-w4a16-ct-ehf16-af16 (verified)<br>gemma-4-12b-it-text-q4k-ehf16-af32 (experimental)<br>gemma-4-12b-it-text-q4k-ehf16-af16 (experimental) | none | browser receipt<br>electron receipt<br>doppler-serve quickstart lane<br>q4k correctness receipt<br>q4k benchmark receipt<br>mtp lane | browser-receipt-missing (browser, unverified)<br>electron-receipt-missing (electron, unverified)<br>serve-quickstart-lane-missing (serve, unsupported)<br>q4k-correctness-receipt-missing (model, missing)<br>q4k-benchmark-receipt-missing (benchmark, missing)<br>mtp-lane-not-implemented (mtp, not_implemented) |
| Gemma 4 31B | partially_verified | verified | unverified | unverified | unsupported | yes | not_implemented | gemma-4-31b-it-text-q4k-ehf16-af16 (browser, pass)<br>gemma-4-31b-it-text-q4k-ehf16-af32 (browser, pass) | gemma-4-31b-it-text-q4k-ehf16-af32 (browser, diagnostic) | none | none | gemma-4-31b-it-text-q4k-ehf16-af32 (verified)<br>gemma-4-31b-it-text-q4k-ehf16-af16 (verified) | none | electron receipt<br>doppler-serve quickstart lane<br>node receipt<br>mtp lane<br>current benchmark receipt | electron-receipt-missing (electron, unverified)<br>serve-quickstart-lane-missing (serve, unsupported)<br>node-receipt-missing (node, unverified)<br>mtp-lane-not-implemented (mtp, not_implemented)<br>current-benchmark-receipt-missing (benchmark, missing) |
| Gemma 4 26B A4B | gap | unsupported | unsupported | unsupported | unsupported | yes | not_implemented | none | none | none | none | none | none | source package profile<br>conversion config<br>catalog model<br>MoE runtime receipt<br>benchmark receipt<br>mtp lane | browser-runtime-unsupported (browser, unsupported)<br>electron-runtime-unsupported (electron, unsupported)<br>node-runtime-unsupported (node, unsupported)<br>serve-runtime-unsupported (serve, unsupported)<br>source-package-profile-missing (model, missing)<br>conversion-config-missing (model, missing)<br>catalog-model-missing (model, missing)<br>official-moe-topology-unsupported (model, unsupported)<br>moe-runtime-receipt-missing (model, missing)<br>benchmark-receipt-missing (benchmark, missing)<br>mtp-lane-not-implemented (mtp, not_implemented) |

## Family Coverage Matrix

| Family | Runtime modelType | Runtime | Conversion configs | Catalog models | Hosted (HF) | Demo | Tested | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| embeddinggemma | embedding | active | 1 (src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json) | 1 (google-embeddinggemma-300m-q4k-ehf16-af32) | yes | none | verified (2026-04-04) | verified | - |
| gemma3 | transformer | active | 3 (src/config/conversion/gemma3/gemma-3-1b-it-f16-af32.json, src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json, src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json) | 2 (gemma-3-1b-it-q4k-ehf16-af32, gemma-3-270m-it-q4k-ehf16-af32) | yes | none | verified (2026-04-13) | verified | catalog verification applies only to cataloged models (2/3 conversion configs cataloged) |
| translategemma | transformer | active | 2 (src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json, src/config/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json) | 1 (translategemma-4b-it-q4k-ehf16-af32) | yes | none | verified (2026-03-20) | verified | catalog verification applies only to cataloged models (1/2 conversion configs cataloged) |
| gemma4 | transformer | active | 10 (src/config/conversion/gemma4/gemma-4-12b-it-text-q4k-ehf16-af16.json, src/config/conversion/gemma4/gemma-4-12b-it-text-q4k-ehf16-af32.json, src/config/conversion/gemma4/gemma-4-12b-it-text-q4k-ehf16-hq4k-af16.json, +7 more) | 8 (gemma-4-12b-it-text-q4k-ehf16-af16, gemma-4-12b-it-text-q4k-ehf16-af32, gemma-4-12b-it-text-w4a16-ct-ehf16-af16, +5 more) | yes | none | partially verified (6/8) | verified | catalog verification applies only to cataloged models (8/10 conversion configs cataloged); partial verification (6/8 catalog models verified) |
| qwen3 | transformer | active | 4 (src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json, src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json, src/config/conversion/qwen3/qwen-3-6-27b-q4k-eaf16.json, +1 more) | 4 (qwen-3-5-0-8b-q4k-ehaf16, qwen-3-5-2b-q4k-ehaf16, qwen-3-6-27b-q4k-eaf16, +1 more) | yes | none | verified (2026-06-15) | verified | - |
| lfm2 | transformer | active | 1 (src/config/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json) | 1 (lfm2-5-1-2b-instruct-q4k-ehf16-af32) | no | none | unknown | verification-pending | not verified in catalog lifecycle |
| diffusiongemma | transformer | active | 1 (src/config/conversion/diffusiongemma/diffusiongemma-26b-a4b-it-q4k-ehf16-af16.json) | 1 (diffusiongemma-26b-a4b-it-q4k-ehf16-af16) | no | none | unknown | verification-pending | not verified in catalog lifecycle |
| gpt_oss | transformer | active | 1 (src/config/conversion/gpt-oss-20b-f16-xmxfp4.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| janus_text | transformer | active | 1 (src/config/conversion/janus/janus-pro-1b-text-q4k-ehaf16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |

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
- Catalog entries: 18
