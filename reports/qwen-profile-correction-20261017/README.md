# Qwen browser profile correction and matched comparisons

Retain one configuration correction: explicitly select existing Qwen decode fusions in the Linux parity comparison lane. No runtime JavaScript or shader changed. The original Vulkan prefill projection kernels and 4x4 decode cadence remain selected. Metal profiles are unchanged controls. The prior [default-profile AMD loss](../qwen-prepared-execution-followup-20261017/README.md) remains valid and unchanged.

All eight comparison receipts pass the configured fairness gates and match all 128 generated token IDs on this synthetic workload. Doppler wins decode throughput and complete-generation latency in all eight observations. First-token latency, loading and token-delivery gaps do not show universal wins. These are local artifact comparisons, not hosted release claims, general numerical correctness, application quality or independent adoption.

## Exact scope

- Qwen3.5 0.8B and 2B: Doppler Q4_K weights, F32 compute, F16 KV; Transformers.js 4.2.0 ONNX q4f16. Product formats and quantization implementations differ; this is not identical-weight engine isolation. Exact model/tokenizer/source identities are in every receipt.
- Apple M3, macOS, Chrome 155, Node 22.20.0; Radeon 8060S / RADV STRIX_HALO, Linux Vulkan, Chrome 146, Node 22.22.1. Full device and library identities are in summary and raw receipts.
- Identical 64-token raw input, 128-token greedy output cap, seed 0, no chat template, two warmups and three measured runs. Both engine orders; jobs serialized per GPU. Hosts were not exclusively reserved.
- HTTP loading from provisioned local artifacts. Initial engine loading is reported independently; filesystem/driver caches were not purged. This is not network-cold startup qualification.
- Each row is the harness aggregate from one process per engine. Two engine orders expose substantial variability, especially Metal 2B; they do not establish confidence bounds for universal superiority. No outlier removal.

## Measurements

Each cell is **Doppler / Transformers.js**. Times are milliseconds; decode is tokens/second. Order D means Doppler first, T means Transformers.js first.

| GPU | Model | Order | Loading | Warm first token | Decode | Complete answer | Delivery gap p95 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| metal | 0.8B | D | 1322.6 / 2521.8 | 181.7 / 146.7 | 57.1 / 40.7 | 2403.0 / 3265.9 | 274.2 / 70.8 |
| metal | 0.8B | T | 1459.3 / 2062.7 | 192.6 / 142.8 | 52.6 / 45.3 | 2641.7 / 2949.1 | 288.2 / 24.2 |
| metal | 2B | D | 3083.9 / 6853.0 | 503.0 / 784.8 | 32.5 / 10.8 | 4410.5 / 12494.8 | 442.6 / 150.0 |
| metal | 2B | T | 4117.0 / 7210.8 | 667.4 / 424.2 | 22.0 / 20.8 | 6436.8 / 6516.4 | 661.8 / 66.0 |
| vulkan | 0.8B | D | 2343.8 / 2303.8 | 144.4 / 101.7 | 98.4 / 26.9 | 1448.2 / 4816.1 | 148.3 / 62.9 |
| vulkan | 0.8B | T | 1724.5 / 2896.6 | 179.9 / 123.6 | 82.5 / 26.6 | 1871.3 / 4900.7 | 173.2 / 65.8 |
| vulkan | 2B | D | 4323.6 / 3190.0 | 213.1 / 279.1 | 56.2 / 34.4 | 2529.4 / 3967.3 | 218.6 / 50.3 |
| vulkan | 2B | T | 4622.8 / 4291.6 | 269.9 / 286.1 | 53.9 / 24.3 | 2621.5 / 5462.5 | 249.1 / 74.4 |

The missing Linux profile explains an execution-path difference: the former default comparison omitted the previously tested fusions and readback configuration. This is not a contemporaneous baseline/candidate A/B estimate of a fivefold optimization; host conditions and comparator timings also changed. The eight new rows establish their own scoped product comparisons.

## Where time goes

Warm prefill wall time is retained in each uninstrumented comparison. A separate GPU-timestamp diagnostic records one **fresh** generation with the same 64-token input and one output token. The debug command does not execute the benchmark warmup loop even though its requested config retained those fields. These diagnostic timings cannot be substituted for warm TTFT or summed into an uninstrumented latency budget.

| GPU | Model | Fresh prefill wall | GPU timestamps | Layer recording wall | Submission wait wall |
| --- | --- | ---: | ---: | ---: | ---: |
| metal | 0.8B | 331.1 | 152.8 | 122.8 | 156.0 |
| metal | 2B | 608.7 | 325.6 | 143.7 | 312.2 |
| vulkan | 0.8B | 482.2 | 71.6 | 338.3 | 98.2 |
| vulkan | 2B | 820.4 | 262.3 | 464.1 | 260.1 |

On Metal, the three FFN projections account for about 49.7 ms / 163.6 ms of observed GPU work for 0.8B / 2B; recurrent attention accounts for 37.9 / 37.8 ms. On Vulkan, recurrent attention accounts for 37.4 / 118.6 ms; linear A/B projections together account for 10.0 / 51.2 ms. Exact grouped timings are in the summary. These identify measured work, not an authorized second kernel correction.

Recording wall time includes awaited preparation and can contain CPU/GPU synchronization; it is not pure CPU work. Submission wait includes completion. Instrumentation changes pass boundaries and adds overhead. The initial Vulkan diagnostic mistakenly requested instrumentation through `bench`; the command rejected it before inference. Both failed attempts are retained, followed by successful `debug` executions.

Delivery stalls have a separate source: `generator/decode-runtime.js` requests `batchSize * readbackInterval` tokens, awaits `_generateNTokensGPU`, then yields its returned tokens. `generator-steps.js` submits partial command buffers while recording, but copies and maps the token staging buffer only after the complete requested batch. A 4x4 cadence therefore delivers up to 16 tokens together. Faster compute reduces a burst interval without making it token-by-token delivery. The existing readback/map-wait observation includes GPU completion and does not prove that copying bytes itself is the bottleneck.

## Reuse, cancellation and limitations

The selected runtime configurations separately passed the Node physical prepared-sequence test for both models on both GPUs. Four varied requests per model compare retained parameters with full reconstruction, prove buffer reuse, and verify cancellation followed by correct fresh-request reuse. These Node lifecycle receipts do not establish browser device-loss recovery. Browser warm comparisons independently exercise repeated generation.

Focused configuration validation, comparison CLI contracts, onboarding validation and prepared-sequence ownership tests passed. No broad release qualification was run. The existing architecture-limit failure and skipped checks are not closed by this report. GPU-memory comparability and platform telemetry limits remain explicit in `summary.json`; device-global memory is not process residency.

The Reploid movie campaign remains rejected at 5/12. Independent CPU controls reproduce two wrong Q4_K answers; agreement on synthetic benchmark tokens does not rescue useful-answer quality. A matching F16 control is the next discriminator for those specific failures.

## Evidence and reproduction

- `contract.json`: frozen hypothesis, workload and source hashes; base source `e6b97ddd`, whose executable runtime matches `64e27ed4`.
- `executed-config/`: exact configuration bytes used in isolated snapshots. The checked-in comparison config retains the original Unicode spelling of an unrelated description; its JSON meaning matches the executed copy.
- `summary.json`: all eight rows, artifact/device identities, resource limitations, four fresh diagnostics and four Node lifecycle results.
- `raw-receipts.tar.gz`: exact comparison JSON, commands/configs, success/failure logs and process results. Extract into an empty directory. Run recorded argument arrays with only machine-local snapshot/model/browser/output paths adapted; use already provisioned artifacts and serialize GPU work.
- Reproduction driver scripts retain the actual orchestration. The first prefill driver intentionally remains as the failed historical attempt; the debug driver is the corrected command surface. Its requested warmup fields do not establish executed warmups.
- `SHA256SUMS`: retained bundle integrity. No vendor matrix or hosted package was promoted.

Component: doppler.runtime-source.config, doppler.benchmarks.vendors. Intent: preserved. Acceptance evidence: scoped comparisons, focused checks, diagnostic and lifecycle receipts above. Boundary effects: explicit Linux parity-profile selection only; no model semantics, public service, runtime default, kernel, application policy or training change.
