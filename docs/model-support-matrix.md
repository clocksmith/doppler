# Model Support Matrix

Auto-generated from conversion configs (`src/config/conversion/**`), `models/catalog.json`, and `models/gemma4-targets.json`.
Run `npm run support:matrix:sync` after editing `models/catalog.json`, `models/gemma4-targets.json`, or changing conversion configs.

Updated at: 2026-07-11

## Current Inference Status

This section answers "which models work now?" from `models/catalog.json` lifecycle metadata plus the quickstart registry.

### 1. Verified

| Model ID | Family | Modes | Last verified | Surface | Notes |
| --- | --- | --- | --- | --- | --- |
| gemma-3-270m-it-f16-af32 | gemma3 | text, vision | 2026-06-29 | node | Node/WebGPU verification is package-visible at reports/release-claims/gemma-3-270m-it-f16-af32/2026-06-29T22-02-14.821Z.json. The report produced the coherent answer "The sky is blue.", stopReason=stop-token, executionContractOk=true, and decodeTokensPerSec > 0. |
| gemma-3-270m-it-q4k-ehf16-af32 | gemma3 | text, vision | 2026-06-24 | browser, node | Browser/WebGPU quickstart smoke is package-visible at reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json. The report produced a coherent sky-color answer, stopReason=stop-token, executionContractOk=true, and decodeTokensPerSec > 0. |
| google-embeddinggemma-300m-q4k-ehf16-af32 | embeddinggemma | embedding | 2026-06-24 | node | Node/WebGPU embedding verification is package-visible at reports/release-claims/google-embeddinggemma-300m-q4k-ehf16-af32/2026-06-24T01-42-03.814Z.json. The report produced finite 768-dimensional unit-norm embeddings and passed the semantic retrieval/pair suite thresholds. Registry verification uses profiles/vector-stability so the manifest-owned f32 KV-cache contract is preserved. |
| gemma-3-1b-it-q4k-ehf16-af32 | gemma3 | text, vision | 2026-06-24 | node | Node/WebGPU quickstart smoke is package-visible at reports/release-claims/gemma-3-1b-it-q4k-ehf16-af32/2026-06-24T01-23-10.494Z.json. The report produced coherent text, executionContractOk=true, and decodeTokensPerSec > 0. |
| gemma-4-e2b-it-q4k-ehf16-af32 | gemma4 | text, vision | 2026-04-18 | node | Package-visible runtime evidence is staged at reports/release-claims/gemma-4-e2b-it-q4k-ehf16-af32/2026-04-18T19-17-45.997Z.json. Browser and Electron remain unverified in the Gemma 4 target matrix until same-surface runtime receipts are promoted. |
| gemma-4-e2b-it-q4k-ehf16-af32-int4ple | gemma4 | text, vision | 2026-04-22 | node | Committed program-bundle reference reports/program-bundles/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/2026-04-22T18-09-10.471Z.reference.json decodes the sky prompt to blue with executionContractOk=true. Browser and Electron remain unverified in the Gemma 4 target matrix until same-surface runtime receipts are promoted. |
| translategemma-4b-1b-enes-q4k-ehf16-af32 | translategemma | translate | 2026-07-11 | browser, node | The exact Q4K artifact passed shard integrity, deterministic EN/ES smoke outputs, 8/8 repeated-token checks, and a 128-row WMT13 evaluation at 31.9149 BLEU / 58.2124 chrF. Baseline and optimized Doppler outputs match 128/128. The pinned hosted revision also passed browser/WebGPU verification and a 2+3-run optimized benchmark with exact expected output. Experimental visibility preserves the measured 2.2748 BLEU / 1.7264 chrF Q4K gap to the source BF16 checkpoint and does not imply Apple Metal evidence. |
| gemma-4-e2b-it-q4k-ehf16-af16-int4ple | gemma4 | text, vision | 2026-05-07 | node | F16 activation sibling over the verified Gemma 4 E2B INT4 PLE weight pack. Package-visible runtime evidence is staged at reports/release-claims/gemma-4-e2b-it-q4k-ehf16-af16-int4ple/2026-05-07T20-15-34.710Z.json; Apple Metal capability resolution still fails closed to the AF32 primary when this lane is incompatible. |
| translategemma-4b-it-q4k-ehf16-af32 | translategemma | translate | 2026-03-20 | browser, node | Browser/WebGPU on Apple M3 and Node/WebGPU on AMD RDNA-3 verified through 2026-03-20 with structured TranslateGemma requests. en->fr 'Hello world.' -> 'Bonjour le monde.' The working Q4K path is gemma3-q4k-dequant-f32w-f32a-online; the prior failure remained isolated to the older q4k_dequant F16-weight path. |
| gemma-4-12b-it-text-q4k-ehf16-af32 | gemma4 | text | 2026-06-29 | node | Node/WebGPU verification is package-visible at reports/release-claims/gemma-4-12b-it-text-q4k-ehf16-af32/2026-06-29T22-13-57.102Z.json. The safe-single report produced a coherent sky-color answer, stopReason=max-tokens, executionContractOk=true, and decodeTokensPerSec > 0. |
| gemma-4-12b-it-text-q4k-ehf16-af16 | gemma4 | text | 2026-06-29 | node | Node/WebGPU verification is package-visible at reports/release-claims/gemma-4-12b-it-text-q4k-ehf16-af16/2026-06-29T22-14-43.963Z.json. The f16 activation safe-single report produced a coherent sky-color answer, stopReason=max-tokens, executionContractOk=true, and decodeTokensPerSec > 0. |
| gemma-4-12b-it-text-w4a16-ct-ehf16-af16 | gemma4 | text | 2026-06-29 | node | Node/WebGPU verification is package-visible at reports/release-claims/gemma-4-12b-it-text-w4a16-ct-ehf16-af16/2026-06-29T22-04-29.156Z.json. The report produced a coherent sky-color answer, stopReason=max-tokens, executionContractOk=true, and decodeTokensPerSec > 0. |
| gemma-4-31b-it-text-q4k-ehf16-af32 | gemma4 | text | 2026-06-29 | node | Node/WebGPU verification is package-visible at reports/release-claims/gemma-4-31b-it-text-q4k-ehf16-af32/2026-06-29T22-07-45.149Z.json. The safe single-token report produced the coherent completion "blue", stopReason=stop-token, executionContractOk=true, and decodeTokensPerSec > 0. |
| gemma-4-31b-it-text-q4k-ehf16-af16 | gemma4 | text | 2026-06-29 | node | Node/WebGPU verification is package-visible at reports/release-claims/gemma-4-31b-it-text-q4k-ehf16-af16/2026-06-29T22-09-42.445Z.json. The f16 activation report produced the coherent completion "blue", stopReason=stop-token, executionContractOk=true, and decodeTokensPerSec > 0. |
| qwen-3-embedding-0-6b-q4k-ehf16-af32 | qwen3 | embedding | 2026-07-05 | browser, node | Fresh hosted registry verify receipts pass on node (reports/release-claims/qwen-3-embedding-0-6b-q4k-ehf16-af32/2026-07-05T16-24-38.048Z.node.json) and browser (reports/release-claims/qwen-3-embedding-0-6b-q4k-ehf16-af32/2026-07-05T16-26-35.664Z.browser.json). Semantic retrieval top-1 accuracy 0.90, semantic pair accuracy 0.7143, finite ratio 1.0, 1024-dimensional unit-norm embeddings. Fresh speed evidence: node embeddingMs 144.14, browser embeddingMs 184.70, and local comparable benchmark benchmarks/vendors/results/embedding_compare_qwen-3-embedding-0-6b-q4k-ehf16-af32_20260705T152659.json (Doppler medianEmbeddingMs 67.60; Transformers.js q4f16 medianEmbeddingMs 155.68; correctnessOk true). |
| qwen-3-reranker-0-6b-q4k-ehf16-af32 | qwen3 | rerank | 2026-07-05 | browser, node | Fresh hosted registry verify receipts pass on node (reports/release-claims/qwen-3-reranker-0-6b-q4k-ehf16-af32/2026-07-05T16-28-17.169Z.node.json) and browser (reports/release-claims/qwen-3-reranker-0-6b-q4k-ehf16-af32/2026-07-05T16-30-58.312Z.browser.json). Semantic pair accuracy 1.0 (5/5) on both receipts, WebGPU query topDocumentIndex 0, rerankMs 1665.34 on node and 1797.80 in browser for the three-document smoke request. Speed evidence: reports/release-claims/qwen-3-reranker-0-6b-q4k-ehf16-af32/2026-07-04T18-43-20.530Z.bench.json (medianRerankMs 1546.27 for the same request). External reranker receipts compare against onnx-community/Qwen3-Reranker-0.6B-ONNX using the q4 ONNX artifact and must not be summarized as a Doppler speed win unless the saved receipt says Doppler leads. |
| qwen-3-reranker-0-6b-f16-af32 | qwen3 | rerank | 2026-07-04 | node | Release evidence: reports/release-claims/qwen-3-reranker-0-6b-f16-af32/2026-07-04T02-00-00.000Z.json. Semantic pair accuracy 1.0 (5/5), WebGPU query topDocumentIndex 0, rerankMs 4057.56 for the three-document smoke request. No separate benchmark receipt is promoted for this f16 reranker lane. |
| qwen-3-5-0-8b-q4k-ehaf16 | qwen3 | text | 2026-06-15 | browser, node | Browser/Node release evidence is package-visible at reports/release-claims/qwen-3-5-0-8b-q4k-ehaf16/2026-05-10T02-22-04.891Z.json. The hosted Clocksmith/rdrr artifact is approved for quickstart resolution with explicit stop-token metadata. |
| qwen-3-5-2b-q4k-ehaf16 | qwen3 | text | 2026-06-15 | browser, node | Browser/Node release evidence is package-visible at reports/release-claims/qwen-3-5-2b-q4k-ehaf16/2026-05-03T02-33-21.397Z.json. The execution-v1 lane keeps the fixed fused-Q4 decode path and q4 LM-head/prefill kernels; the hosted Clocksmith/rdrr artifact is approved for quickstart resolution with explicit stop-token metadata. |
| qwen-3-6-27b-q4k-ehaf16 | qwen3 | text | 2026-04-28 | browser | Committed program-bundle reference reports/program-bundles/qwen-3-6-27b-q4k-ehaf16/2026-04-28T01-19-10.497Z.reference.json captured a deterministic reference transcript and executionContractOk=true. Hosted artifact is enabled for web demo download; quickstart remains disabled. |
| qwen-3-6-27b-q4k-eaf16 | qwen3 | text | 2026-04-29 | browser, node | Committed program-bundle reference reports/program-bundles/qwen-3-6-27b-q4k-eaf16/capture.node.reference.json produces coherent sky prompt output with f16 compute. |
| amplify-120m-f16-af32 | amplify | embedding | 2026-07-19 | node | Node/WebGPU sequence parity and synthetic q_proj LoRA lifecycle passed at docs/status/amplify-120m-sequence-webgpu-lora-qualification-2026-07-19.json. This is not a downstream biological-task quality claim. |
| esm2-t12-35m-ur50d-f32-af32 | esm | embedding | 2026-07-19 | node | Node/WebGPU sequence parity and synthetic q_proj LoRA lifecycle passed at docs/status/esm2-t12-35m-sequence-webgpu-lora-qualification-2026-07-19.json. The initial lane exposes embeddings, not masked-token logits or downstream biological quality. |
| esmc-300m-f32-af32 | esmc | embedding | 2026-07-19 | node | Node/WebGPU sequence parity and synthetic q_proj LoRA lifecycle passed at docs/status/esmc-300m-sequence-webgpu-lora-qualification-2026-07-19.json. The initial lane exposes embeddings, not masked-token logits or downstream biological quality. |
| nucleotide-transformer-v2-50m-f32-af32 | nucleotide-transformer | embedding | 2026-07-19 | node | Node/WebGPU sequence parity and synthetic q_proj LoRA lifecycle passed at docs/status/nucleotide-transformer-v2-50m-sequence-webgpu-lora-qualification-2026-07-19.json. The initial lane exposes embeddings, not masked-token logits or downstream biological quality; upstream use is CC-BY-NC-SA-4.0. |

### 2. Loads But Unverified

None right now.

### 3. Known Failing

| Model ID | Family | Modes | Last checked | Surface | Notes |
| --- | --- | --- | --- | --- | --- |
| minicpm4-0-5b-f16-af32 | minicpm | text | 2026-07-04 | node | Failure evidence: reports/release-claims/minicpm4-0-5b-f16-af32/2026-07-04T01-07-00.000Z.failed.json. The prompt "Q: What color is the sky on a clear day? A:" produced "This problem question: When是我国 an an", so the artifact remains local-only until tokenizer/template/config correctness is fixed. |

### 4. Quickstart-Supported Only

None right now.

### 5. Everything Else

| Entry | Type | Status | Notes |
| --- | --- | --- | --- |
| diffusiongemma-26b-a4b-it-q4k-ehf16-af16 | catalog model | experimental | Cataloged model without a verified or failing inference lifecycle result. |
| lfm2-5-1-2b-instruct-q4k-ehf16-af32 | catalog model | experimental | Cataloged model without a verified or failing inference lifecycle result. |
| gpt_oss | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |
| janus_text | model family | conversion-ready | conversion configs exist, but there is no cataloged model entry yet |

## Gemma 4 Target Coverage

Generated from `models/gemma4-targets.json`. This section tracks the latest official Gemma 4 target set separately from the catalog, so unsupported or unverified targets stay visible.

| Target | Doppler status | Browser | Electron | Node | Serve | Official MTP | Doppler MTP | Runtime receipts | Benchmark receipts | Serve receipts | Preflight receipts | Current lanes | Source packages | Missing | Blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 E2B | partially_verified | unverified | unverified | verified | verified | yes | not_implemented | gemma-4-e2b-it-q4k-ehf16-af32 (node, pass)<br>gemma-4-e2b-it-q4k-ehf16-af32-int4ple (node, pass)<br>gemma-4-e2b-it-q4k-ehf16-af16-int4ple (node, pass) | none | gemma-4-e2b-it-q4k-ehf16-af32-int4ple (serve, pass)<br>gemma-4-e2b-it-q4k-ehf16-af32-int4ple (serve, diagnostic) | gemma-4-e2b-it-q4k-ehf16-af32 (node, pass)<br>gemma-4-e2b-it-q4k-ehf16-af32-int4ple (node, pass)<br>gemma-4-e2b-it-q4k-ehf16-af16-int4ple (node, pass) | gemma-4-e2b-it-q4k-ehf16-af32 (verified)<br>gemma-4-e2b-it-q4k-ehf16-af32-int4ple (verified)<br>gemma-4-e2b-it-q4k-ehf16-af16-int4ple (verified-local) | none | browser runtime pass receipt<br>hosted quickstart artifact refresh<br>electron receipt<br>benchmark receipt<br>mtp lane<br>full multimodal browser receipt | browser-runtime-pass-receipt-missing (browser, unverified)<br>benchmark-receipt-missing (benchmark, missing)<br>electron-receipt-missing (electron, unverified)<br>mtp-lane-not-implemented (mtp, not_implemented)<br>multimodal-browser-receipt-incomplete (browser, incomplete) |
| Gemma 4 E4B | gap | unsupported | unsupported | unsupported | unsupported | yes | not_implemented | none | none | none | none | none | litert/gemma-4-e4b-it (blocked) | conversion config<br>catalog model<br>RDRR artifact<br>runtime receipt<br>benchmark receipt<br>mtp lane | browser-runtime-unsupported (browser, unsupported)<br>electron-runtime-unsupported (electron, unsupported)<br>node-runtime-unsupported (node, unsupported)<br>serve-runtime-unsupported (serve, unsupported)<br>conversion-config-missing (model, missing)<br>catalog-model-missing (model, missing)<br>rdrr-artifact-missing (model, missing)<br>runtime-receipt-missing (model, missing)<br>benchmark-receipt-missing (benchmark, missing)<br>mtp-lane-not-implemented (mtp, not_implemented) |
| Gemma 4 12B Unified | partially_verified | unsupported | unsupported | verified | unsupported | yes | not_implemented | gemma-4-12b-it-text-q4k-ehf16-af32 (node, pass)<br>gemma-4-12b-it-text-q4k-ehf16-af16 (node, pass)<br>gemma-4-12b-it-text-w4a16-ct-ehf16-af16 (node, pass) | none | none | none | gemma-4-12b-it-text-q4k-ehf16-af32 (verified)<br>gemma-4-12b-it-text-q4k-ehf16-af16 (verified)<br>gemma-4-12b-it-text-w4a16-ct-ehf16-af16 (verified) | none | browser receipt<br>electron receipt<br>doppler-serve quickstart lane<br>q4k benchmark receipt<br>mtp lane | browser-runtime-unsupported (browser, unsupported)<br>electron-runtime-unsupported (electron, unsupported)<br>serve-quickstart-lane-missing (serve, unsupported)<br>q4k-benchmark-receipt-missing (benchmark, missing)<br>mtp-lane-not-implemented (mtp, not_implemented) |
| Gemma 4 31B | partially_verified | unverified | unverified | verified | unsupported | yes | not_implemented | gemma-4-31b-it-text-q4k-ehf16-af32 (node, pass)<br>gemma-4-31b-it-text-q4k-ehf16-af16 (node, pass) | none | none | none | gemma-4-31b-it-text-q4k-ehf16-af32 (verified)<br>gemma-4-31b-it-text-q4k-ehf16-af16 (verified) | none | electron receipt<br>doppler-serve quickstart lane<br>mtp lane<br>current benchmark receipt | browser-release-claim-missing (browser, unverified)<br>electron-receipt-missing (electron, unverified)<br>serve-quickstart-lane-missing (serve, unsupported)<br>mtp-lane-not-implemented (mtp, not_implemented)<br>current-benchmark-receipt-missing (benchmark, missing) |
| Gemma 4 26B A4B | gap | unsupported | unsupported | unsupported | unsupported | yes | not_implemented | none | none | none | none | none | none | source package profile<br>conversion config<br>catalog model<br>MoE runtime receipt<br>benchmark receipt<br>mtp lane | browser-runtime-unsupported (browser, unsupported)<br>electron-runtime-unsupported (electron, unsupported)<br>node-runtime-unsupported (node, unsupported)<br>serve-runtime-unsupported (serve, unsupported)<br>source-package-profile-missing (model, missing)<br>conversion-config-missing (model, missing)<br>catalog-model-missing (model, missing)<br>official-moe-topology-unsupported (model, unsupported)<br>moe-runtime-receipt-missing (model, missing)<br>benchmark-receipt-missing (benchmark, missing)<br>mtp-lane-not-implemented (mtp, not_implemented) |

## Family Coverage Matrix

| Family | Runtime modelType | Runtime | Conversion configs | Catalog models | Hosted (HF) | Demo | Tested | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| embeddinggemma | embedding | active | 1 (src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json) | 1 (google-embeddinggemma-300m-q4k-ehf16-af32) | yes | none | verified (2026-06-24) | verified | - |
| gemma3 | transformer | active | 5 (src/config/conversion/gemma3/gemma-3-1b-it-f16-af32.json, src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json, src/config/conversion/gemma3/gemma-3-270m-it-f16-af32.json, +2 more) | 3 (gemma-3-1b-it-q4k-ehf16-af32, gemma-3-270m-it-f16-af32, gemma-3-270m-it-q4k-ehf16-af32) | yes | none | verified (2026-06-29) | verified | catalog verification applies only to cataloged models (3/5 conversion configs cataloged) |
| translategemma | transformer | active | 2 (src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json, src/config/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json) | 2 (translategemma-4b-1b-enes-q4k-ehf16-af32, translategemma-4b-it-q4k-ehf16-af32) | yes | none | verified (2026-07-11) | verified | - |
| gemma4 | transformer | active | 10 (src/config/conversion/gemma4/gemma-4-12b-it-text-q4k-ehf16-af16.json, src/config/conversion/gemma4/gemma-4-12b-it-text-q4k-ehf16-af32.json, src/config/conversion/gemma4/gemma-4-12b-it-text-q4k-ehf16-hq4k-af16.json, +7 more) | 8 (gemma-4-12b-it-text-q4k-ehf16-af16, gemma-4-12b-it-text-q4k-ehf16-af32, gemma-4-12b-it-text-w4a16-ct-ehf16-af16, +5 more) | yes | none | verified (2026-06-29) | verified | catalog verification applies only to cataloged models (8/10 conversion configs cataloged) |
| qwen3 | transformer | active | 9 (src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json, src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json, src/config/conversion/qwen3/qwen-3-5-9b-f16-af32.json, +6 more) | 7 (qwen-3-5-0-8b-q4k-ehaf16, qwen-3-5-2b-q4k-ehaf16, qwen-3-6-27b-q4k-eaf16, +4 more) | yes | none | verified (2026-07-05) | verified | catalog verification applies only to cataloged models (7/9 conversion configs cataloged) |
| lfm2 | transformer | active | 1 (src/config/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json) | 1 (lfm2-5-1-2b-instruct-q4k-ehf16-af32) | no | none | unknown | verification-pending | not verified in catalog lifecycle |
| amplify | transformer | active | 1 (src/config/conversion/amplify/amplify-120m-f16-af32.json) | 1 (amplify-120m-f16-af32) | yes | none | verified (2026-07-19) | verified | - |
| diffusiongemma | transformer | active | 1 (src/config/conversion/diffusiongemma/diffusiongemma-26b-a4b-it-q4k-ehf16-af16.json) | 1 (diffusiongemma-26b-a4b-it-q4k-ehf16-af16) | no | none | unknown | verification-pending | not verified in catalog lifecycle |
| esm | transformer | active | 1 (src/config/conversion/esm/esm2-t12-35m-ur50d-f32-af32.json) | 1 (esm2-t12-35m-ur50d-f32-af32) | yes | none | verified (2026-07-19) | verified | - |
| esmc | transformer | active | 1 (src/config/conversion/esmc/esmc-300m-f32-af32.json) | 1 (esmc-300m-f32-af32) | yes | none | verified (2026-07-19) | verified | - |
| gpt_oss | transformer | active | 1 (src/config/conversion/gpt-oss-20b-f16-xmxfp4.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| janus_text | transformer | active | 1 (src/config/conversion/janus/janus-pro-1b-text-q4k-ehaf16.json) | 0 | no | none | unknown | conversion-ready | not in local catalog; not verified in catalog lifecycle |
| minicpm | transformer | active | 1 (src/config/conversion/minicpm/minicpm4-0-5b-f16-af32.json) | 1 (minicpm4-0-5b-f16-af32) | no | none | failed | verification-failed | - |
| nucleotide-transformer | transformer | active | 1 (src/config/conversion/nucleotide-transformer/nucleotide-transformer-v2-50m-f32-af32.json) | 1 (nucleotide-transformer-v2-50m-f32-af32) | yes | none | verified (2026-07-19) | verified | - |

## Summary

- Families tracked: 14
- Families with conversion configs: 14
- Families present in catalog: 12
- Verified families (active runtime + conversion + catalog + passing verification): 9
- Cataloged families pending verification: 2
- Families with HF-hosted catalog entries: 9
- Families with verified catalog lifecycle: 9
- Families with failed catalog verification: 1
- Blocked runtime families: 0
- Catalog entries: 28
