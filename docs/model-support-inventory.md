# Model Support Inventory

Generated from model catalog, conversion configs, support rollout policy, compare profiles, release matrix, claim lanes, and release-claim receipts.

Updated at: 2026-07-05

Policy: smallest artifact size first. Size tiers use catalog artifact bytes, not parameter class. A preferred architecture is selected only when runtime verification, compare JSON, and summary SVG evidence exist for that lane.

## Gate Order

1. conversion-config: Checked-in conversion config exists.
2. manifest-weights: Manifest and weight identity are complete, or the manifest explicitly references a hosted weight pack.
3. runtime-verify: A deterministic runtime verification receipt is committed.
4. hf-publish: Public support has a Hugging Face repo, revision, and artifact path.
5. compare-profile: A cross-engine compare profile exists.
6. benchmark-receipts: Compare JSON and summary SVG receipts exist for the lane.
7. preferred-architecture: Preferred architecture is selected only from benchmark receipts.

## Summary

- Catalog models: 23
- Source checkpoints: 16
- Conversion-only configs: 7
- HF-published catalog models: 13
- Runtime-verified catalog models: 20
- Benchmark-selected source architectures: 1
- Sources pending benchmark-selected architecture: 15

## Rollout Queue

| Tier | Source checkpoint | Smallest artifact | Status | Preferred architecture | Evidence | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| small | google/gemma-3-270m-it | 0.39 GiB | benchmark-selected | gemma-3-270m-it-q4k-ehf16-af32 (RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout) | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T200811.json<br>benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-browser-p512-d128-t0-k1-strix-halo-20260627T200811.svg | gemma-3-270m-it-q4k-ehf16-af32 is selected from committed runtime, compare JSON, and SVG receipts. |
| small | google/embeddinggemma-300m | 0.45 GiB | benchmark-comparison-needed | google-embeddinggemma-300m-q4k-ehf16-af32 candidate; benchmark comparison pending | reports/release-claims/google-embeddinggemma-300m-q4k-ehf16-af32/2026-06-24T01-42-03.814Z.json | Run compare/bench receipts for google-embeddinggemma-300m-q4k-ehf16-af32. |
| small | LiquidAI/LFM2.5-1.2B-Instruct | 0.79 GiB | runtime-verification-needed | pending verification | none | Refresh manifest and weight identity for lfm2-5-1-2b-instruct-q4k-ehf16-af32. |
| small | openbmb/MiniCPM4-0.5B | 0.81 GiB | verification-failed | failed verification | none | Fix failed runtime verification for minicpm4-0-5b-f16-af32; keep it unpromoted until a passing receipt exists. |
| small | Qwen/Qwen3-Reranker-0.6B | 0.87 GiB | benchmark-comparison-needed | qwen-3-reranker-0-6b-q4k-ehf16-af32 candidate; benchmark comparison pending | reports/release-claims/qwen-3-reranker-0-6b-q4k-ehf16-af32/2026-07-05T16-28-17.169Z.node.json | Add a compare profile for qwen-3-reranker-0-6b-q4k-ehf16-af32. |
| small | google/gemma-3-1b-it | 0.97 GiB | benchmark-comparison-needed | gemma-3-1b-it-q4k-ehf16-af32 candidate; benchmark comparison pending | reports/release-claims/gemma-3-1b-it-q4k-ehf16-af32/2026-06-24T01-23-10.494Z.json | Run compare/bench receipts for gemma-3-1b-it-q4k-ehf16-af32. |
| small | Qwen/Qwen3.5-0.8B | 1.08 GiB | benchmark-receipts-incomplete | qwen-3-5-0-8b-q4k-ehaf16 candidate; benchmark comparison pending | reports/release-claims/qwen-3-5-0-8b-q4k-ehaf16/2026-05-10T02-22-04.891Z.json | Run compare/bench receipts for qwen-3-5-0-8b-q4k-ehaf16. |
| small | Qwen/Qwen3-Embedding-0.6B | 1.11 GiB | benchmark-comparison-needed | qwen-3-embedding-0-6b-q4k-ehf16-af32 candidate; benchmark comparison pending | reports/release-claims/qwen-3-embedding-0-6b-q4k-ehf16-af32/2026-07-05T16-24-38.048Z.node.json | Run compare/bench receipts for qwen-3-embedding-0-6b-q4k-ehf16-af32. |
| medium | Qwen/Qwen3.5-2B | 2.32 GiB | benchmark-comparison-needed | qwen-3-5-2b-q4k-ehaf16 candidate; benchmark comparison pending | reports/release-claims/qwen-3-5-2b-q4k-ehaf16/2026-05-03T02-33-21.397Z.json | Decide whether qwen-3-5-2b-q4k-ehaf16 needs a claimable generation compare lane or stays capability-only. |
| medium | google/translategemma-4b-it | 2.95 GiB | benchmark-comparison-needed | translategemma-4b-it-q4k-ehf16-af32 candidate; benchmark comparison pending | reports/release-claims/translategemma-4b-it-q4k-ehf16-af32/2026-03-22T14-48-13.935Z.json | Run compare/bench receipts for translategemma-4b-it-q4k-ehf16-af32. |
| medium | google/gemma-4-E2B-it | 3.73 GiB | benchmark-comparison-needed | gemma-4-e2b-it-q4k-ehf16-af16-int4ple candidate; benchmark comparison pending | reports/release-claims/gemma-4-e2b-it-q4k-ehf16-af16-int4ple/2026-05-07T20-15-34.710Z.json | Publish or refresh the Hugging Face manifest/weights for gemma-4-e2b-it-q4k-ehf16-af16-int4ple. |
| medium | google/gemma-4-12B-it | 7.60 GiB | benchmark-comparison-needed | gemma-4-12b-it-text-q4k-ehf16-af16 candidate; benchmark comparison pending | reports/release-claims/gemma-4-12b-it-text-q4k-ehf16-af16/2026-06-29T22-14-43.963Z.json | Publish or refresh the Hugging Face manifest/weights for gemma-4-12b-it-text-q4k-ehf16-af16. |
| large | google/gemma-4-12B-it-qat-w4a16-ct | 9.47 GiB | benchmark-comparison-needed | gemma-4-12b-it-text-w4a16-ct-ehf16-af16 candidate; benchmark comparison pending | reports/release-claims/gemma-4-12b-it-text-w4a16-ct-ehf16-af16/2026-06-29T22-04-29.156Z.json | Publish or refresh the Hugging Face manifest/weights for gemma-4-12b-it-text-w4a16-ct-ehf16-af16. |
| large | google/diffusiongemma-26B-A4B-it | 14.6 GiB | runtime-verification-needed | pending verification | none | Run deterministic runtime verification for diffusiongemma-26b-a4b-it-q4k-ehf16-af16. |
| large | Qwen/Qwen3.6-27B | 15.8 GiB | benchmark-comparison-needed | qwen-3-6-27b-q4k-eaf16 candidate; benchmark comparison pending | reports/program-bundles/qwen-3-6-27b-q4k-eaf16/capture.node.reference.json | Add a compare profile for qwen-3-6-27b-q4k-eaf16. |
| large | google/gemma-4-31B-it | 18.0 GiB | benchmark-comparison-needed | gemma-4-31b-it-text-q4k-ehf16-af16 candidate; benchmark comparison pending | reports/release-claims/gemma-4-31b-it-text-q4k-ehf16-af16/2026-06-29T22-09-42.445Z.json | Publish or refresh the Hugging Face manifest/weights for gemma-4-31b-it-text-q4k-ehf16-af16. |

## Next Commands

These are policy-generated command recipes, not evidence. A command becomes support evidence only after its saved artifact is committed and referenced by the claim lane.

| Tier | Source checkpoint | Next gate | Command |
| --- | --- | --- | --- |
| small | google/embeddinggemma-300m | compare-result | `node tools/compare-embeddings.js --model-id google-embeddinggemma-300m-q4k-ehf16-af32 --warmup 1 --runs 3 --doppler-source quickstart-registry --doppler-surface auto --cache-mode warm --load-mode http --save --json` |
| small | openbmb/MiniCPM4-0.5B | runtime-verify | `node tools/run-registry-verify.js minicpm4-0-5b-f16-af32 --surface auto` |
| small | google/gemma-3-1b-it | compare-result | `node tools/compare-engines.js --model-id gemma-3-1b-it-q4k-ehf16-af32 --workload p064-d064-t0-k1 --mode compute --decode-profile parity --warmup 1 --runs 3 --save --json` |
| small | Qwen/Qwen3.5-0.8B | summary-svg | `node tools/compare-engines.js --model-id qwen-3-5-0-8b-q4k-ehaf16 --workload p064-d064-t0-k1 --mode compute --decode-profile parity --warmup 1 --runs 3 --save --json` |
| small | Qwen/Qwen3-Embedding-0.6B | compare-result | `node tools/compare-embeddings.js --model-id qwen-3-embedding-0-6b-q4k-ehf16-af32 --warmup 1 --runs 3 --doppler-source local --doppler-surface auto --cache-mode warm --load-mode http --save --json` |
| medium | google/translategemma-4b-it | compare-result | `node tools/compare-engines.js --model-id translategemma-4b-it-q4k-ehf16-af32 --workload p064-d064-t0-k1 --mode compute --decode-profile parity --warmup 1 --runs 3 --save --json` |
| medium | google/gemma-4-E2B-it | hf-publish | `node tools/publish-hf-registry-model.js --model-id gemma-4-e2b-it-q4k-ehf16-af16-int4ple --dry-run --bootstrap` |
| medium | google/gemma-4-12B-it | hf-publish | `node tools/publish-hf-registry-model.js --model-id gemma-4-12b-it-text-q4k-ehf16-af16 --dry-run --bootstrap` |
| large | google/gemma-4-12B-it-qat-w4a16-ct | hf-publish | `node tools/publish-hf-registry-model.js --model-id gemma-4-12b-it-text-w4a16-ct-ehf16-af16 --dry-run --bootstrap` |
| large | google/diffusiongemma-26B-A4B-it | runtime-verify | `node tools/run-registry-verify.js diffusiongemma-26b-a4b-it-q4k-ehf16-af16 --surface auto` |
| large | google/gemma-4-31B-it | hf-publish | `node tools/publish-hf-registry-model.js --model-id gemma-4-31b-it-text-q4k-ehf16-af16 --dry-run --bootstrap` |

## Small Models

| Model ID | Family | Size | Architecture | Runtime verify | HF | Compare lane | Benchmark evidence | Next gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-3-270m-it-q4k-ehf16-af32 | gemma3 | 0.39 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | 2026-06-24 (browser+node) | Clocksmith/rdrr@7e194b4a6824a8b71cbd0eaa511d9f2c7e1c0129 | performance_comparable / onnx-community/gemma-3-270m-it-ONNX | benchmarks/vendors/results/compare_20260627T200811.json<br>benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-browser-p512-d128-t0-k1-strix-halo-20260627T200811.svg | preferred-architecture |
| gemma-3-270m-it-f16-af32 | gemma3 | 0.50 GiB | RDRR, f16 weights, f16 embeddings, f16 LM head, f32 compute | 2026-06-29 (node) | missing | performance_comparable / onnx-community/gemma-3-270m-it-ONNX | missing | hf-publish |
| google-embeddinggemma-300m-q4k-ehf16-af32 | embeddinggemma | 0.45 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | 2026-06-24 (node) | Clocksmith/rdrr@95f0b29ec73dea70394c6bcfa8407bc6796df6c9 | performance_comparable / onnx-community/embeddinggemma-300m-ONNX | missing | compare-result |
| lfm2-5-1-2b-instruct-q4k-ehf16-af32 | lfm2 | 0.79 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | missing | missing | performance_comparable / LiquidAI/LFM2.5-1.2B-Instruct-ONNX | missing | manifest-weights |
| minicpm4-0-5b-f16-af32 | minicpm | 0.81 GiB | RDRR, f16 weights, f16 embeddings, f16 LM head, f32 compute | 2026-07-04 failed | missing | missing | missing | runtime-verify |
| qwen-3-reranker-0-6b-q4k-ehf16-af32 | qwen3 | 0.87 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | 2026-07-05 (browser+node) | Clocksmith/rdrr@f86fe245b9bbc275cd69af46b1d45d47ea685a55 | missing | missing | compare-profile |
| qwen-3-reranker-0-6b-f16-af32 | qwen3 | 1.11 GiB | RDRR, f16 weights, f16 embeddings, f16 LM head, f32 compute | 2026-07-04 (node) | Clocksmith/rdrr@cc1fafff8cda609372b608cd92e487f6a2c32bc8 | missing | missing | compare-profile |
| gemma-3-1b-it-q4k-ehf16-af32 | gemma3 | 0.97 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | 2026-06-24 (node) | Clocksmith/rdrr@7c3d30e300bcb02cbd68fb0db3eee64fbf738f99 | performance_comparable / onnx-community/gemma-3-1b-it-ONNX-GQA | missing | compare-result |
| qwen-3-5-0-8b-q4k-ehaf16 | qwen3 | 1.08 GiB | RDRR, q4k weights, f16 embeddings, q4k LM head, f32 compute, row Q4K layout | 2026-06-15 (browser+node) | Clocksmith/rdrr@95a01447eecbf13fc5964986f507b08ded0cd40f | performance_comparable / onnx-community/Qwen3.5-0.8B-ONNX | missing | summary-svg |
| qwen-3-embedding-0-6b-q4k-ehf16-af32 | qwen3 | 1.11 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | 2026-07-05 (browser+node) | Clocksmith/rdrr@049000f49325dca7db2ed2c9de2c8881bd0f4603 | performance_comparable / onnx-community/Qwen3-Embedding-0.6B-ONNX | missing | compare-result |

## Medium Models

| Model ID | Family | Size | Architecture | Runtime verify | HF | Compare lane | Benchmark evidence | Next gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen-3-5-2b-q4k-ehaf16 | qwen3 | 2.32 GiB | RDRR, q4k weights, f16 embeddings, q4k LM head, f32 compute, row Q4K layout | 2026-06-15 (browser+node) | Clocksmith/rdrr@a8c45dd885a789042d3b82c95b471d66ca8d5152 | capability_only / onnx-community/Qwen3.5-2B-ONNX | missing | benchmark-lane-capability-only |
| translategemma-4b-it-q4k-ehf16-af32 | translategemma | 2.95 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | 2026-03-20 (browser+node) | Clocksmith/rdrr@6fc46049882e961a57d1690ba1ffde21677d001a | performance_comparable / onnx-community/translategemma-text-4b-it-ONNX | missing | compare-result |
| gemma-4-e2b-it-q4k-ehf16-af16-int4ple | gemma4 | 3.73 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f16 compute, row Q4K layout | 2026-05-07 (node) | missing | missing | missing | hf-publish |
| gemma-4-e2b-it-q4k-ehf16-af32-int4ple | gemma4 | 3.73 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | 2026-04-22 (node) | Clocksmith/rdrr@dbb354acb00108965c5324ca7b6fecc198d12501 | performance_comparable / onnx-community/gemma-4-E2B-it-ONNX | missing | compare-result |
| gemma-4-e2b-it-q4k-ehf16-af32 | gemma4 | 6.61 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | 2026-04-18 (node) | Clocksmith/rdrr@2070777c77047d54e6eae105f6dcb1891cf6f21a | performance_comparable / onnx-community/gemma-4-E2B-it-ONNX | missing | compare-result |
| gemma-4-12b-it-text-q4k-ehf16-af16 | gemma4 | 7.60 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f16 compute, row Q4K layout | 2026-06-29 (node) | missing | missing | missing | hf-publish |
| gemma-4-12b-it-text-q4k-ehf16-af32 | gemma4 | 7.60 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | 2026-06-29 (node) | missing | missing | missing | hf-publish |

## Large Models

| Model ID | Family | Size | Architecture | Runtime verify | HF | Compare lane | Benchmark evidence | Next gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-4-12b-it-text-w4a16-ct-ehf16-af16 | gemma4 | 9.47 GiB | RDRR, w4a16 weights, f16 embeddings, f16 LM head, f16 compute | 2026-06-29 (node) | missing | missing | missing | hf-publish |
| diffusiongemma-26b-a4b-it-q4k-ehf16-af16 | diffusiongemma | 14.6 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f16 compute, row Q4K layout | missing | missing | missing | missing | runtime-verify |
| qwen-3-6-27b-q4k-eaf16 | qwen3 | 15.8 GiB | RDRR, q4k weights, f16 embeddings, q4k LM head, f16 compute, row Q4K layout | 2026-04-29 (browser+node) | Clocksmith/rdrr@3dee21b3b12d65ac7fef9b24cbf759cacc953a67 | missing | missing | compare-profile |
| qwen-3-6-27b-q4k-ehaf16 | qwen3 | 15.8 GiB | RDRR, q4k weights, f16 embeddings, q4k LM head, f32 compute, row Q4K layout | 2026-04-28 (browser) | Clocksmith/rdrr@b402f6f27837857d51636da5f78c12bcd47e2a03 | missing | missing | compare-profile |
| gemma-4-31b-it-text-q4k-ehf16-af16 | gemma4 | 18.0 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f16 compute, row Q4K layout | 2026-06-29 (node) | missing | missing | missing | hf-publish |
| gemma-4-31b-it-text-q4k-ehf16-af32 | gemma4 | 18.0 GiB | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | 2026-06-29 (node) | missing | missing | missing | hf-publish |

## Conversion-Only Configs

These entries have checked-in conversion configs but are not catalog-supported runtime artifacts yet.

| Config | Model base ID | Family | Architecture | Next action |
| --- | --- | --- | --- | --- |
| src/config/conversion/gemma3/gemma-3-1b-it-f16-af32.json | gemma-3-1b-it-f16-af32 | gemma3 | RDRR, f16 weights, f16 embeddings, f16 LM head, f32 compute | Catalog, verify, and publish gemma-3-1b-it-f16-af32 before claiming runtime support. |
| src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af16.json | gemma-3-270m-it-q4k-ehf16-af16 | gemma3 | RDRR, q4k weights, f16 embeddings, f16 LM head, f16 compute, row Q4K layout | Catalog, verify, and publish gemma-3-270m-it-q4k-ehf16-af16 before claiming runtime support. |
| src/config/conversion/gemma4/gemma-4-12b-it-text-q4k-ehf16-hq4k-af16.json | gemma-4-12b-it-text-q4k-ehf16-hq4k-af16 | gemma4 | RDRR, q4k weights, f16 embeddings, q4k LM head, f16 compute, row Q4K layout | Catalog, verify, and publish gemma-4-12b-it-text-q4k-ehf16-hq4k-af16 before claiming runtime support. |
| src/config/conversion/gemma4/gemma-4-moe-q4k-ehf16-af32.json | gemma-4-moe-q4k-ehf16-af32 | gemma4 | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | Catalog, verify, and publish gemma-4-moe-q4k-ehf16-af32 before claiming runtime support. |
| src/config/conversion/gpt-oss-20b-f16-xmxfp4.json | gpt-oss-20b-f16-xmxfp4 | root | RDRR, f16 weights, f16 embeddings, f16 LM head, f16 compute | Catalog, verify, and publish gpt-oss-20b-f16-xmxfp4 before claiming runtime support. |
| src/config/conversion/janus/janus-pro-1b-text-q4k-ehaf16.json | janus-pro-1b-text-q4k-ehaf16 | janus | RDRR, q4k weights, f16 embeddings, f16 LM head, f16 compute, row Q4K layout | Catalog, verify, and publish janus-pro-1b-text-q4k-ehaf16 before claiming runtime support. |
| src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json | translategemma-4b-1b-enes-q4k-ehf16-af32 | gemma3 | RDRR, q4k weights, f16 embeddings, f16 LM head, f32 compute, row Q4K layout | Catalog, verify, and publish translategemma-4b-1b-enes-q4k-ehf16-af32 before claiming runtime support. |

## Source Files

- models/catalog.json
- src/config/conversion/**
- benchmarks/vendors/support-rollout-policy.json
- benchmarks/vendors/benchmark-policy.json
- benchmarks/vendors/workloads.json
- benchmarks/vendors/compare-engines.config.json
- benchmarks/vendors/embedding-compare.config.json
- benchmarks/vendors/local-inference-claim-matrix.json
- benchmarks/vendors/release-matrix.json
- tools/policies/release-claim-policy.json

