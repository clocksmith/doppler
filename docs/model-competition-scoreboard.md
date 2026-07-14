# Model Competition Scoreboard

Generated from the catalog, support inventory, release matrix, embedding/rerank compare configs, and saved compare receipts.
This file is an evidence ledger: it records what is verified, what is on Hugging Face according to catalog metadata, where Doppler has comparable performance receipts, and which gates remain.

Updated at: 2026-07-11T19:35:41.171Z

## Summary

- Catalog models: 24
- Runtime-verified models: 21
- HF-published models: 15
- Failed models: 1
- Verification-needed models: 2
- Generation compare rows: 15
- Generation compare gap rows: 3
- Embedding compare rows: 2
- Rerank compare rows: 1
- Doppler decode-leading generation rows: 13
- Transformers.js decode-leading generation rows: 2
- Doppler embedding latency-leading rows: 1
- Transformers.js embedding latency-leading rows: 1
- Doppler rerank latency-leading rows: 1
- Transformers.js rerank latency-leading rows: 0
- Evidence-incomplete rows: 29

## Claim Status Rules

- `claim-ready` and `claimable` mean the row has enough committed evidence for that row-level claim.
- `candidate`, `local-comparable`, and `*-missing` rows are useful engineering evidence, but not release claims.
- Missing HF, runtime, compare JSON, or SVG evidence stays visible as a gate instead of being inferred from local state.

## Generation Competition Rows

| Model | Surface | Workload | Correctness | Doppler decode | TJS decode | Decode leader | Doppler prompt | TJS prompt | Prompt leader | Bottleneck | Claim | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-3-270m-it-q4k-ehf16-af32 | browser | p064-d064-t0-k1<br>throughput | exact | 175.82 tok/s | 107.09 tok/s | doppler | 1036.45 tok/s | 840.39 tok/s | doppler | readback map wait | summary-svg-missing | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T202837.json |
| gemma-3-270m-it-q4k-ehf16-af32 | browser | p256-d128-t0-k1<br>throughput | exact | 164.23 tok/s | 113.02 tok/s | doppler | 2071.2 tok/s | 1021.49 tok/s | doppler | readback map wait | summary-svg-missing | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T203348.json |
| gemma-3-270m-it-q4k-ehf16-af32 | browser | p512-d128-t0-k1<br>throughput | exact | 156.51 tok/s | 96.08 tok/s | doppler | 2426.55 tok/s | 1023.97 tok/s | doppler | readback map wait | candidate | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T200811.json<br>benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-browser-p512-d128-t0-k1-strix-halo-20260627T200811.svg |
| gemma-3-270m-it-q4k-ehf16-af32 | bun | p064-d064-t0-k1<br>throughput | exact | 115.53 tok/s | 110.44 tok/s | doppler | 968.41 tok/s | 832.52 tok/s | doppler | command recording | summary-svg-missing | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T202736.json |
| gemma-3-270m-it-q4k-ehf16-af32 | bun | p256-d128-t0-k1<br>throughput | exact | 108.39 tok/s | 109.93 tok/s | transformersjs | 1969.8 tok/s | 992.25 tok/s | doppler | command recording | summary-svg-missing | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T203220.json |
| gemma-3-270m-it-q4k-ehf16-af32 | bun | p512-d128-t0-k1<br>throughput | exact | 105.55 tok/s | 97.66 tok/s | doppler | 2323.66 tok/s | 1003.97 tok/s | doppler | command recording | candidate | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T200603.json<br>benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-bun-p512-d128-t0-k1-strix-halo-20260627T200603.svg |
| gemma-3-270m-it-q4k-ehf16-af32 | node | p064-d064-t0-k1<br>throughput | exact | 112.49 tok/s | 108.16 tok/s | doppler | 1105.97 tok/s | 831.87 tok/s | doppler | command recording | summary-svg-missing | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T202549.json |
| gemma-3-270m-it-q4k-ehf16-af32 | node | p256-d128-t0-k1<br>throughput | exact | 105.99 tok/s | 109.18 tok/s | transformersjs | 2156.26 tok/s | 1008.41 tok/s | doppler | command recording | summary-svg-missing | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T203031.json |
| gemma-3-270m-it-q4k-ehf16-af32 | node | p512-d128-t0-k1<br>throughput | exact | 103 tok/s | 97.15 tok/s | doppler | 2453.29 tok/s | 1021.53 tok/s | doppler | command recording | candidate | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T200323.json<br>benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-node-p512-d128-t0-k1-strix-halo-20260627T200323.svg |
| gemma-4-e2b-it-q4k-ehf16-af32-int4ple | browser | p064-d064-t0-k1<br>parity | mismatch | 16.32 tok/s | 9.24 tok/s | doppler | 125.1 tok/s | 299.55 tok/s | transformersjs | decode orchestration | summary-svg-missing | reports/program-bundles/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/2026-04-22T18-09-10.471Z.reference.json<br>benchmarks/vendors/results/compare_20260707T170557.json |
| qwen-3-5-0-8b-q4k-ehaf16 | browser | p064-d064-t0-k1<br>throughput | exact | 75.87 tok/s | 41.57 tok/s | doppler | 184.6 tok/s | 9.52 tok/s | doppler | readback map wait | summary-svg-missing | reports/release-claims/qwen-3-5-0-8b-q4k-ehaf16/2026-05-10T02-22-04.891Z.json<br>benchmarks/vendors/results/compare_20260705T160226.json |
| qwen-3-5-0-8b-q4k-ehaf16 | browser | p256-d128-t0-k1<br>throughput | exact | 73.74 tok/s | 43.17 tok/s | doppler | 432.94 tok/s | 9.78 tok/s | doppler | readback map wait | summary-svg-missing | reports/release-claims/qwen-3-5-0-8b-q4k-ehaf16/2026-05-10T02-22-04.891Z.json<br>benchmarks/vendors/results/compare_20260705T161224.json |
| qwen-3-5-2b-q4k-ehaf16 | browser | p064-d064-t0-k1<br>throughput | exact | 60.95 tok/s | 40.1 tok/s | doppler | 168.47 tok/s | 9.9 tok/s | doppler | readback map wait | summary-svg-missing | reports/release-claims/qwen-3-5-2b-q4k-ehaf16/2026-05-03T02-33-21.397Z.json<br>benchmarks/vendors/results/compare_20260707T154847.json |
| qwen-3-5-2b-q4k-ehaf16 | browser | p256-d128-t0-k1<br>throughput | exact | 58.98 tok/s | 40.64 tok/s | doppler | 372.61 tok/s | 9.59 tok/s | doppler | readback map wait | summary-svg-missing | reports/release-claims/qwen-3-5-2b-q4k-ehaf16/2026-05-03T02-33-21.397Z.json<br>benchmarks/vendors/results/compare_20260707T155858.json |
| qwen-3-5-2b-q4k-ehaf16 | browser | p512-d128-t0-k1<br>throughput | exact | 56.5 tok/s | 41.5 tok/s | doppler | 446.83 tok/s | 9.61 tok/s | doppler | readback map wait | summary-svg-missing | reports/release-claims/qwen-3-5-2b-q4k-ehaf16/2026-05-03T02-33-21.397Z.json<br>benchmarks/vendors/results/compare_20260707T161623.json |

## Embedding Competition Rows

| Model | Correctness | Doppler median | TJS median | Latency leader | Doppler throughput | TJS throughput | Throughput leader | Load leader | Claim | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| google-embeddinggemma-300m-q4k-ehf16-af32 | semantic-pass | 60.04 ms | 25.42 ms | transformersjs | 16.74 emb/s | 35.13 emb/s | transformersjs | doppler | claimable | benchmarks/vendors/results/embedding_compare_google-embeddinggemma-300m-q4k-ehf16-af32_20260704T154500.json |
| qwen-3-embedding-0-6b-q4k-ehf16-af32 | semantic-pass | 29.4 ms | 145.89 ms | doppler | 33.88 emb/s | 6.69 emb/s | doppler | transformersjs | claimable | benchmarks/vendors/results/embedding_compare_qwen-3-embedding-0-6b-q4k-ehf16-af32_20260710T011455.json |

## Rerank Competition Rows

| Model | Correctness | Doppler median | TJS median | Latency leader | Doppler throughput | TJS throughput | Throughput leader | Load leader | Claim | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen-3-reranker-0-6b-q4k-ehf16-af32 | semantic-pass | 277.6 ms | 469.25 ms | doppler | 3.59 rerank/s | 2.12 rerank/s | doppler | transformersjs | claimable | benchmarks/vendors/results/rerank_compare_qwen-3-reranker-0-6b-q4k-ehf16-af32_20260710T014450.json |

## Support And Competition Gaps

| Model | Mode | HF | Platforms | Competitor | Claim/gate | Missing | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| diffusiongemma-26b-a4b-it-q4k-ehf16-af16 | support | missing | browser:missing<br>node:missing<br>bun:missing | transformersjs | verification-needed<br>runtime-verify | runtime-verify<br>hf-publish<br>compare-profile | none |
| gemma-3-1b-it-q4k-ehf16-af32 | generation | Clocksmith/rdrr@7c3d30e300bcb02cbd68fb0db3eee64fbf738f99<br>models/gemma-3-1b-it-q4k-ehf16-af32 | browser:missing<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/gemma-3-1b-it-ONNX-GQA<br>onnx/q4f16 | candidate<br>compare-result | chromium-webgpu<br>p064-d064-t0-k1<br>p256-d128-t0-k1<br>p512-d128-t0-k1<br>parity<br>throughput<br>chromium-webgpu:p064-d064-t0-k1<br>chromium-webgpu:p256-d128-t0-k1<br>chromium-webgpu:p512-d128-t0-k1 | reports/release-claims/gemma-3-1b-it-q4k-ehf16-af32/2026-06-24T01-23-10.494Z.json |
| gemma-3-270m-it-f16-af32 | support | missing | browser:missing<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/gemma-3-270m-it-ONNX<br>onnx | compare-missing<br>hf-publish | hf-publish<br>claim-lane<br>compare-result<br>summary-svg | reports/release-claims/gemma-3-270m-it-f16-af32/2026-06-29T22-02-14.821Z.json |
| gemma-3-270m-it-q4k-ehf16-af32 | generation | Clocksmith/rdrr@7e194b4a6824a8b71cbd0eaa511d9f2c7e1c0129<br>models/gemma-3-270m-it-q4k-ehf16-af32 | browser:benchmarked<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/gemma-3-270m-it-ONNX<br>onnx/q4f16 | summary-svg-missing<br>summary-svg | summary-svg | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T202837.json |
| gemma-3-270m-it-q4k-ehf16-af32 | generation | Clocksmith/rdrr@7e194b4a6824a8b71cbd0eaa511d9f2c7e1c0129<br>models/gemma-3-270m-it-q4k-ehf16-af32 | browser:benchmarked<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/gemma-3-270m-it-ONNX<br>onnx/q4f16 | summary-svg-missing<br>summary-svg | summary-svg | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T203348.json |
| gemma-3-270m-it-q4k-ehf16-af32 | generation | Clocksmith/rdrr@7e194b4a6824a8b71cbd0eaa511d9f2c7e1c0129<br>models/gemma-3-270m-it-q4k-ehf16-af32 | browser:verified<br>node:verified<br>bun:benchmarked | transformersjs<br>onnx-community/gemma-3-270m-it-ONNX<br>onnx/q4f16 | summary-svg-missing<br>summary-svg | summary-svg | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T202736.json |
| gemma-3-270m-it-q4k-ehf16-af32 | generation | Clocksmith/rdrr@7e194b4a6824a8b71cbd0eaa511d9f2c7e1c0129<br>models/gemma-3-270m-it-q4k-ehf16-af32 | browser:verified<br>node:verified<br>bun:benchmarked | transformersjs<br>onnx-community/gemma-3-270m-it-ONNX<br>onnx/q4f16 | summary-svg-missing<br>summary-svg | summary-svg | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T203220.json |
| gemma-3-270m-it-q4k-ehf16-af32 | generation | Clocksmith/rdrr@7e194b4a6824a8b71cbd0eaa511d9f2c7e1c0129<br>models/gemma-3-270m-it-q4k-ehf16-af32 | browser:verified<br>node:benchmarked<br>bun:missing | transformersjs<br>onnx-community/gemma-3-270m-it-ONNX<br>onnx/q4f16 | summary-svg-missing<br>summary-svg | summary-svg | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T202549.json |
| gemma-3-270m-it-q4k-ehf16-af32 | generation | Clocksmith/rdrr@7e194b4a6824a8b71cbd0eaa511d9f2c7e1c0129<br>models/gemma-3-270m-it-q4k-ehf16-af32 | browser:verified<br>node:benchmarked<br>bun:missing | transformersjs<br>onnx-community/gemma-3-270m-it-ONNX<br>onnx/q4f16 | summary-svg-missing<br>summary-svg | summary-svg | reports/release-claims/gemma-3-270m-it-q4k-ehf16-af32/2026-06-24T01-11-24.275Z.json<br>benchmarks/vendors/results/compare_20260627T203031.json |
| gemma-4-12b-it-text-q4k-ehf16-af16 | text | missing | browser:missing<br>node:verified<br>bun:missing | transformersjs | verified-no-compare<br>hf-publish | hf-publish<br>compare-profile | reports/release-claims/gemma-4-12b-it-text-q4k-ehf16-af16/2026-06-29T22-14-43.963Z.json |
| gemma-4-12b-it-text-q4k-ehf16-af32 | text | missing | browser:missing<br>node:verified<br>bun:missing | transformersjs | verified-no-compare<br>hf-publish | hf-publish<br>compare-profile | reports/release-claims/gemma-4-12b-it-text-q4k-ehf16-af32/2026-06-29T22-13-57.102Z.json |
| gemma-4-12b-it-text-w4a16-ct-ehf16-af16 | text | missing | browser:missing<br>node:verified<br>bun:missing | transformersjs | verified-no-compare<br>hf-publish | hf-publish<br>compare-profile | reports/release-claims/gemma-4-12b-it-text-w4a16-ct-ehf16-af16/2026-06-29T22-04-29.156Z.json |
| gemma-4-31b-it-text-q4k-ehf16-af16 | text | missing | browser:missing<br>node:verified<br>bun:missing | transformersjs | verified-no-compare<br>hf-publish | hf-publish<br>compare-profile | reports/release-claims/gemma-4-31b-it-text-q4k-ehf16-af16/2026-06-29T22-09-42.445Z.json |
| gemma-4-31b-it-text-q4k-ehf16-af32 | text | missing | browser:missing<br>node:verified<br>bun:missing | transformersjs | verified-no-compare<br>hf-publish | hf-publish<br>compare-profile | reports/release-claims/gemma-4-31b-it-text-q4k-ehf16-af32/2026-06-29T22-07-45.149Z.json |
| gemma-4-e2b-it-q4k-ehf16-af16-int4ple | support | Clocksmith/rdrr@2d326649ae3d04b83323de0e6247e1e42fb1d9af<br>models/gemma-4-e2b-it-q4k-ehf16-af16-int4ple | browser:missing<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/gemma-4-E2B-it-ONNX<br>onnx/q4f16 | compare-missing<br>claim-lane | claim-lane<br>compare-result<br>summary-svg | reports/release-claims/gemma-4-e2b-it-q4k-ehf16-af16-int4ple/2026-05-07T20-15-34.710Z.json |
| gemma-4-e2b-it-q4k-ehf16-af32 | generation | Clocksmith/rdrr@2070777c77047d54e6eae105f6dcb1891cf6f21a<br>models/gemma-4-e2b-it-q4k-ehf16-af32 | browser:missing<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/gemma-4-E2B-it-ONNX<br>onnx/q4f16 | candidate<br>compare-result | chromium-webgpu<br>p064-d064-t0-k1<br>p256-d128-t0-k1<br>p512-d128-t0-k1<br>parity<br>throughput<br>chromium-webgpu:p064-d064-t0-k1<br>chromium-webgpu:p256-d128-t0-k1<br>chromium-webgpu:p512-d128-t0-k1 | reports/release-claims/gemma-4-e2b-it-q4k-ehf16-af32/2026-04-18T19-17-45.997Z.json |
| gemma-4-e2b-it-q4k-ehf16-af32-int4ple | generation | Clocksmith/rdrr@16256bf16dc8f92b8fe5105b07628cef91587f0a<br>models/gemma-4-e2b-it-q4k-ehf16-af32-int4ple | browser:benchmarked<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/gemma-4-E2B-it-ONNX<br>onnx/q4f16 | summary-svg-missing<br>summary-svg | summary-svg | reports/program-bundles/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/2026-04-22T18-09-10.471Z.reference.json<br>benchmarks/vendors/results/compare_20260707T170557.json |
| lfm2-5-1-2b-instruct-q4k-ehf16-af32 | text | missing | browser:missing<br>node:missing<br>bun:missing | transformersjs<br>LiquidAI/LFM2.5-1.2B-Instruct-ONNX<br>onnx | verification-needed<br>manifest-weights | manifest-weights<br>runtime-verify<br>hf-publish<br>claim-lane<br>compare-result<br>summary-svg | none |
| minicpm4-0-5b-f16-af32 | text | missing | browser:missing<br>node:verified<br>bun:missing | transformersjs | failed<br>runtime-verify | runtime-verify<br>hf-publish<br>compare-profile | none |
| qwen-3-5-0-8b-q4k-ehaf16 | generation | Clocksmith/rdrr@f58f1d0b58641c84e7ea50d13fea0dd4dc91389a<br>models/qwen-3-5-0-8b-q4k-ehaf16 | browser:benchmarked<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/Qwen3.5-0.8B-ONNX<br>onnx/q4f16 | summary-svg-missing<br>summary-svg | summary-svg | reports/release-claims/qwen-3-5-0-8b-q4k-ehaf16/2026-05-10T02-22-04.891Z.json<br>benchmarks/vendors/results/compare_20260705T160226.json |
| qwen-3-5-0-8b-q4k-ehaf16 | generation | Clocksmith/rdrr@f58f1d0b58641c84e7ea50d13fea0dd4dc91389a<br>models/qwen-3-5-0-8b-q4k-ehaf16 | browser:benchmarked<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/Qwen3.5-0.8B-ONNX<br>onnx/q4f16 | summary-svg-missing<br>summary-svg | summary-svg | reports/release-claims/qwen-3-5-0-8b-q4k-ehaf16/2026-05-10T02-22-04.891Z.json<br>benchmarks/vendors/results/compare_20260705T161224.json |
| qwen-3-5-2b-q4k-ehaf16 | generation | Clocksmith/rdrr@977d145bf2478a7fb542e6aca65030585620ca60<br>models/qwen-3-5-2b-q4k-ehaf16 | browser:benchmarked<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/Qwen3.5-2B-ONNX<br>onnx/q4f16 | summary-svg-missing<br>summary-svg | summary-svg | reports/release-claims/qwen-3-5-2b-q4k-ehaf16/2026-05-03T02-33-21.397Z.json<br>benchmarks/vendors/results/compare_20260707T154847.json |
| qwen-3-5-2b-q4k-ehaf16 | generation | Clocksmith/rdrr@977d145bf2478a7fb542e6aca65030585620ca60<br>models/qwen-3-5-2b-q4k-ehaf16 | browser:benchmarked<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/Qwen3.5-2B-ONNX<br>onnx/q4f16 | summary-svg-missing<br>summary-svg | summary-svg | reports/release-claims/qwen-3-5-2b-q4k-ehaf16/2026-05-03T02-33-21.397Z.json<br>benchmarks/vendors/results/compare_20260707T155858.json |
| qwen-3-5-2b-q4k-ehaf16 | generation | Clocksmith/rdrr@977d145bf2478a7fb542e6aca65030585620ca60<br>models/qwen-3-5-2b-q4k-ehaf16 | browser:benchmarked<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/Qwen3.5-2B-ONNX<br>onnx/q4f16 | summary-svg-missing<br>summary-svg | summary-svg | reports/release-claims/qwen-3-5-2b-q4k-ehaf16/2026-05-03T02-33-21.397Z.json<br>benchmarks/vendors/results/compare_20260707T161623.json |
| qwen-3-6-27b-q4k-eaf16 | text | Clocksmith/rdrr@3dee21b3b12d65ac7fef9b24cbf759cacc953a67<br>models/qwen-3-6-27b-q4k-eaf16 | browser:verified<br>node:verified<br>bun:missing | transformersjs | verified-no-compare<br>compare-profile | compare-profile | reports/program-bundles/qwen-3-6-27b-q4k-eaf16/capture.node.reference.json |
| qwen-3-6-27b-q4k-ehaf16 | text | Clocksmith/rdrr@b402f6f27837857d51636da5f78c12bcd47e2a03<br>models/qwen-3-6-27b-q4k-ehaf16 | browser:verified<br>node:missing<br>bun:missing | transformersjs | verified-no-compare<br>compare-profile | compare-profile | reports/program-bundles/qwen-3-6-27b-q4k-ehaf16/2026-04-28T01-19-10.497Z.reference.json |
| qwen-3-reranker-0-6b-f16-af32 | rerank | Clocksmith/rdrr@cc1fafff8cda609372b608cd92e487f6a2c32bc8<br>models/qwen-3-reranker-0-6b-f16-af32 | browser:missing<br>node:verified<br>bun:missing | transformersjs | verified-no-compare<br>compare-profile | compare-profile | reports/release-claims/qwen-3-reranker-0-6b-f16-af32/2026-07-04T02-00-00.000Z.json |
| translategemma-4b-1b-enes-q4k-ehf16-af32 | translate | Clocksmith/rdrr@1e2c047357f07dfe45a8effc2fa915acd73999ee<br>models/translategemma-4b-1b-enes-q4k-ehf16-af32 | browser:verified<br>node:verified<br>bun:missing | transformersjs | capability-only<br>benchmark-lane-capability-only | benchmark-lane-capability-only | benchmarks/vendors/results/translategemma-nativekd2/hosted-browser/translategemma-4b-1b-enes-q4k-ehf16-af32_20260711T115839.json |
| translategemma-4b-it-q4k-ehf16-af32 | generation | Clocksmith/rdrr@6fc46049882e961a57d1690ba1ffde21677d001a<br>models/translategemma-4b-it-q4k-ehf16-af32 | browser:verified<br>node:verified<br>bun:missing | transformersjs<br>onnx-community/translategemma-text-4b-it-ONNX<br>onnx/q4f16 | candidate<br>compare-result | chromium-webgpu<br>p064-d064-t0-k1<br>p256-d128-t0-k1<br>p512-d128-t0-k1<br>parity<br>throughput<br>chromium-webgpu:p064-d064-t0-k1<br>chromium-webgpu:p256-d128-t0-k1<br>chromium-webgpu:p512-d128-t0-k1 | reports/release-claims/translategemma-4b-it-q4k-ehf16-af32/2026-03-22T14-48-13.935Z.json |

## Next Commands

These commands are gates, not evidence. A row becomes evidence only after its saved artifact is committed and referenced by the catalog, release matrix, or support inventory.

| Model | Gate | Command |
| --- | --- | --- |
| diffusiongemma-26b-a4b-it-q4k-ehf16-af16 | runtime-verify | `node tools/run-registry-verify.js diffusiongemma-26b-a4b-it-q4k-ehf16-af16 --surface auto` |
| gemma-3-1b-it-q4k-ehf16-af32 | compare-result | `node tools/compare-engines.js --model-id gemma-3-1b-it-q4k-ehf16-af32 --workload p064-d064-t0-k1 --mode compute --decode-profile parity --warmup 1 --runs 3 --save --json` |
| gemma-3-270m-it-f16-af32 | hf-publish | `node tools/publish-hf-registry-model.js --model-id gemma-3-270m-it-f16-af32 --dry-run --bootstrap` |
| gemma-3-270m-it-q4k-ehf16-af32 | summary-svg | `node tools/compare-engines.js --model-id gemma-3-270m-it-q4k-ehf16-af32 --workload p064-d064-t0-k1 --mode compute --decode-profile parity --warmup 1 --runs 3 --save --json` |
| gemma-4-12b-it-text-q4k-ehf16-af16 | hf-publish | `node tools/publish-hf-registry-model.js --model-id gemma-4-12b-it-text-q4k-ehf16-af16 --dry-run --bootstrap` |
| gemma-4-12b-it-text-q4k-ehf16-af32 | hf-publish | `node tools/publish-hf-registry-model.js --model-id gemma-4-12b-it-text-q4k-ehf16-af32 --dry-run --bootstrap` |
| gemma-4-12b-it-text-w4a16-ct-ehf16-af16 | hf-publish | `node tools/publish-hf-registry-model.js --model-id gemma-4-12b-it-text-w4a16-ct-ehf16-af16 --dry-run --bootstrap` |
| gemma-4-31b-it-text-q4k-ehf16-af16 | hf-publish | `node tools/publish-hf-registry-model.js --model-id gemma-4-31b-it-text-q4k-ehf16-af16 --dry-run --bootstrap` |
| gemma-4-31b-it-text-q4k-ehf16-af32 | hf-publish | `node tools/publish-hf-registry-model.js --model-id gemma-4-31b-it-text-q4k-ehf16-af32 --dry-run --bootstrap` |
| gemma-4-e2b-it-q4k-ehf16-af16-int4ple | claim-lane | `node tools/compare-engines.js --model-id gemma-4-e2b-it-q4k-ehf16-af16-int4ple --workload p064-d064-t0-k1 --mode compute --decode-profile parity --warmup 1 --runs 3 --save --json` |
| gemma-4-e2b-it-q4k-ehf16-af32 | compare-result | `node tools/compare-engines.js --model-id gemma-4-e2b-it-q4k-ehf16-af32 --workload p064-d064-t0-k1 --mode compute --decode-profile parity --warmup 1 --runs 3 --save --json` |
| gemma-4-e2b-it-q4k-ehf16-af32-int4ple | summary-svg | `node tools/compare-engines.js --model-id gemma-4-e2b-it-q4k-ehf16-af32-int4ple --workload p064-d064-t0-k1 --mode compute --decode-profile parity --warmup 1 --runs 3 --save --json` |
| minicpm4-0-5b-f16-af32 | runtime-verify | `node tools/run-registry-verify.js minicpm4-0-5b-f16-af32 --surface auto` |
| qwen-3-5-0-8b-q4k-ehaf16 | summary-svg | `node tools/compare-engines.js --model-id qwen-3-5-0-8b-q4k-ehaf16 --workload p064-d064-t0-k1 --mode compute --decode-profile parity --warmup 1 --runs 3 --save --json` |
| qwen-3-5-2b-q4k-ehaf16 | summary-svg | `node tools/compare-engines.js --model-id qwen-3-5-2b-q4k-ehaf16 --workload p064-d064-t0-k1 --mode compute --decode-profile parity --warmup 1 --runs 3 --save --json` |
| translategemma-4b-it-q4k-ehf16-af32 | compare-result | `node tools/compare-engines.js --model-id translategemma-4b-it-q4k-ehf16-af32 --workload p064-d064-t0-k1 --mode compute --decode-profile parity --warmup 1 --runs 3 --save --json` |

## Source Files

- models/catalog.json
- benchmarks/vendors/model-support-inventory.json
- benchmarks/vendors/release-matrix.json
- benchmarks/vendors/embedding-compare.config.json
- benchmarks/vendors/rerank-compare.config.json
- benchmarks/vendors/results/embedding_compare_*.json
- benchmarks/vendors/results/rerank_compare_*.json

