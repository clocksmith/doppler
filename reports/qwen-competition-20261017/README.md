# Qwen3.5 0.8B and 2B: matched browser measurements, 2026-10-17

Doppler decoded faster than Transformers.js 4.2.0 on both measured GPUs and both models: **1.16x/1.06x on Apple M3 Metal and 2.31x/1.72x on Radeon 8060S Vulkan** (0.8B/2B). Transformers.js had lower warm first-token latency in every lane. Loading favored Transformers.js in three lanes and Doppler for M3 2B. These are local workload results, not a release qualification or an every-metric win.

## Exact workload and platforms

- **Apple M3, 24 GiB unified memory**, macOS 26.6.2, Chrome 155.0.8059.80, Node 22.20.0, WebGPU Metal (`apple/metal-3`).
- **AMD Ryzen AI MAX+ 395 / Radeon 8060S, 128 GB UMA**, Linux 7.0.0-38-generic, Chrome 146.0.7680.177, Node 22.22.1, WebGPU Vulkan (`RADV STRIX_HALO`, device `0x1586`). The exact browser patch was additionally queried from the installed binary.
- **Qwen3.5 0.8B and 2B**, Doppler IDs `qwen-3-5-0-8b-q4k-ehaf16` and `qwen-3-5-2b-q4k-ehaf16`. Despite the legacy suffix, manifests specify **Q4_K weights, f16 embeddings, Q4_K output head, f32 compute**. Transformers.js uses the registered `onnx-community/Qwen3.5-{0.8B,2B}-ONNX-OPT` **q4f16** conversions.
- Same raw 64-token prompt, no chat template, 128 generated tokens, greedy temperature 0/top-k 1/top-p 1, seed 0, KV budget 193. One warmup and three timed runs per primary lane. Decode counts **127 tokens after the first prefill-produced token**.
- Same browser within each host; browser versions differ between hosts. Both hosts were shared with other work. This is a comparison of optimized product formats, with different embedding quantization, layouts and runtime implementations—not identical quantized tensors. Models were already provisioned; no network weight-download time is included.

## Warm request, loading and decode are separate

Values are Doppler / Transformers.js. TTFT and decode are the harness medians after warmup; loading is the separately recorded model-open interval. Loading plus warm TTFT is not a cold first-answer measurement. Browser process launch is outside model loading.

| Hardware | Exact model | Decode tokens/s | Warm TTFT ms | Model loading ms |
|---|---|---:|---:|---:|
| M3 Metal | Qwen3.5 0.8B | 29.80 / 25.75 | 448.00 / 241.10 | 3295.70 / 3265.93 |
| M3 Metal | Qwen3.5 2B | 20.46 / 19.27 | 997.40 / 472.64 | 5426.50 / 9299.95 |
| 8060S Vulkan | Qwen3.5 0.8B | 97.66 / 42.21 | 308.90 / 84.25 | 3498.30 / 2350.89 |
| 8060S Vulkan | Qwen3.5 2B | 71.28 / 41.52 | 337.80 / 257.40 | 6172.30 / 3294.75 |

All four primary receipts pass the harness local-comparability gate and have exact agreement on all 128 generated token IDs. The prompt is a synthetic repeated phrase: this supports only that workload, not blanket correctness or useful-answer quality. `releaseClaimable` remains false. The raw receipts preserve the complete prompt, actual IDs, runtime configuration, fairness decisions and environment; [comparison-summary.json](comparison-summary.json) points to each receipt inside [raw-receipts.tar.gz](raw-receipts.tar.gz).

## First request in a fresh browser profile

These **single observations** use zero warmups, one timed run and local HTTP model files. OS and GPU-driver caches were not evicted. Times are Doppler / Transformers.js; first answer is model loading plus request TTFT, excluding browser launch. No Vulkan zero-warmup first-answer receipt is available in this bundle; its mode-cold runs still had a warmup and must not be described as cold first requests.

| M3 model | Loading ms | First-request TTFT ms | Load + first token ms | Decode tokens/s |
|---|---:|---:|---:|---:|
| Qwen3.5 0.8B | 3015.10 / 2275.58 | 528.40 / 360.33 | 3543.50 / 2635.91 | 36.01 / 27.91 |
| Qwen3.5 2B | 5084.40 / 7663.43 | 875.80 / 591.21 | 5960.20 / 8254.64 | 19.10 / 17.80 |

## Where prefill time goes

An adjacent M3 browser 0.8B profile attributes **449.2 ms wall prefill** to **222.80 ms measured GPU work** and **171.0 ms CPU command recording**, with roughly **55.4 ms remaining** outside those two counters. GPU work is dominated by projections (171.58 ms), followed by recurrent attention (38.41 ms), full attention (5.81 ms) and convolution (4.63 ms). Remaining time is not yet a proven causal breakdown; timestamp instrumentation changes the run. Submit/readback waits overlap and must not be added twice.

A narrower-tile candidate measured 438.6 ms wall, 224.80 ms GPU and 160.4 ms CPU recording. The 1.4–1.5x Node WebGPU operator win therefore **did not reproduce in browser GPU prefill**. Unprofiled ABBA runs confirm neutral 0.8B TTFT (409–412 ms). 2B varied enough that a candidate benefit is not yet causally established. Existing tile defaults remain unchanged. Next work is runtime-matched attribution of projection dispatch, CPU recording and the remaining prefill wall interval, then prompt-length scaling. More kernel edits are paused until that diagnosis.

## Reproduction

[reproduce.py](reproduce.py) reconstructs the measured source from base commit `5beddd5d3afafa0c4cbe1c6c4b9d6b3d7f950dad` plus the small [source overrides](source-overrides.tar.gz), checks source hashes and provisioned model identities, and invokes the recorded comparison contract. It downloads no models. Use the pinned Node dependencies from that source's lockfile, including Transformers.js 4.2.0; model revisions and file hashes are in [comparator-pins.json](comparator-pins.json) and [doppler-model-pins.json](doppler-model-pins.json). Provision Git LFS tokenizer files where needed. Model roots must contain the exact signed manifests and shards, not a reconversion.

```sh
python3 reports/qwen-competition-20261017/reproduce.py --size 08 \
  --model-root /absolute/path/to/models/local \
  --tjs-root /absolute/path/to/cache-root-containing-onnx-community \
  --browser '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
  --output /absolute/path/to/new-receipts
# Repeat with --size 2b; Vulkan uses --browser /opt/google/chrome/chrome.
# Add --cold for the zero-warmup, single first-request observation.
```

Use a fresh output directory. The script's `--prepare-only` mode verifies reconstruction without launching GPU work. Reproduction establishes repeatable inputs and commands, not identical timings on a shared host. [SHA256SUMS](SHA256SUMS) binds the published bundle.

## Preserved failures and scope limits

The first M3 2B Transformers.js child timed out after 240 seconds. A one-token diagnostic and full retry subsequently passed; cause remains unknown and the failed log is retained. An initial Vulkan comparison selected a missing browser binary and was rerun with the explicit installed Chrome. An earlier M3 comparison tripped the model-override fairness gate; it is diagnostic only. None of these failures count as a performance win.

Native MLX 0.32.4 / mlx-lm 0.32.1 and llama.cpp observations remain separate in raw receipts. MLX warm 0.8B spans 49–107 tokens/s and 2B about 39.5–39.8 tokens/s; llama.cpp Vulkan 2B about 101–102 tokens/s. They expose a remaining performance gap. Native startup/API/HTTP timing scopes differ, and the retained Vulkan llama.cpp run uses older commit `f4884293809b5227d7307140a942f1bc4176a603`. The current-build Vulkan attempt failed configuration because SPIRV-Headers CMake metadata was missing; full native coverage is not established. JS heap, GPU allocator bytes and process RSS are not interchangeable memory measures.

## Implementation and numerical evidence


- `46f5496a`: fused attention cache ownership fix. Direct K/V cache writes have no temporary tensors to own; preserve cache ownership. Focused cleanup, projection and resource-scope tests passed and both optimized models generated 128 matching tokens.
- `ebbe275b`: decode phase accounting excludes the prefill token, including immediate stopping and prefix-KV generation. A targeted three-surface check covers budgets 1/128 and immediate/later EOS; sequential/batched stop accounting tests pass. The old full behavioral parity test has a pre-existing fake-tokenizer incompatibility (`createIncrementalDecoder` missing), retained in its failure log; it is not reported as passing.
- `ebbe275b`: TILE_N specialization updates shader workgroup width and output-column coverage. Focused dispatch tests cover all five WideTile variants and partial tiles.
- `d8b6df24` (pushed): scalar fused recurrent state reuse is byte-exact for output, convolution state and recurrent state over 128 steps. Paired operator tests show about 1.4–1.5x speedup. Reordering compensated arithmetic regressed Vulkan; restoring only the loop order still left a 3–6% regression. Production selection now routes two-component state to byte-identical original shader sources. Separate precision variants are necessary because source changes affected compiler performance even when arithmetic was unchanged.
- A real-weight Metal tile search tests QKV `[6144,1024]` and FFN-down `[1024,3584]`, with 64/256 input rows, alternating baseline/candidate order, three warmups and twelve pairs. All output bytes match. TILE_M=4/TILE_N=128 gives about 1.4–1.5x operator gains. The Node microbenchmark gain did not reproduce in browser 0.8B GPU timings: 222.80 ms baseline versus 224.80 ms candidate. Browser CPU recording was 171.0 versus 160.4 ms and total prefill 449.2 versus 438.6 ms. Actual QKV dispatch changed from [24,16,1] to [48,16,1]. The unprofiled ABBA 0.8B runs were neutral (409–412 ms TTFT); 2B runs suggest improvement but do not explain its source. No tile profile was adopted. Further kernel changes are paused pending runtime-matched prefill attribution.

The prior recurrent-prefill, GPU-resident clone and uniform-validation-cache improvements are in [the preceding report](../qwen-optimization-20261017/README.md). No signed model artifact was rewritten by this campaign. Recipe digest synchronization is not qualification of the 27B model.


## Open evidence

Vulkan zero-warmup first requests, longer prompts, extended quality tasks, transfer/import/export latency, and current-build native coverage remain open. Existing full-test fake-tokenizer failure and the untouched attention architecture budget failure are not passing release gates.

Component: Doppler benchmark evidence. Intent: preserved. Acceptance: four local matched browser lanes, two fresh-profile M3 observations, retained source/model hashes and focused implementation checks. Boundary effects: none from this report; no kernel profile or model artifact is changed by publication.
