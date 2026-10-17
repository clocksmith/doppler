# Qwen installed candidate and application evidence, 2026-10-17

**Qwen3.5 0.8B and 2B, Apple M3 Metal and Radeon 8060S Vulkan:** the prior matched browser runs measured faster Doppler decode in all four lanes, while Transformers.js had lower warm first-token latency in all four. The new installed-candidate campaign passes eight scoped lifecycle lanes, including Node, and exposes remaining application-quality failures. Neither result establishes an every-metric win or independent application retention.

## Matched comparison retained separately

These are the previously published one-warmup/three-run medians, **Doppler / Transformers.js 4.2.0**, with the same 64-token raw prompt and 128 generated tokens. Decode counts the 127 tokens after the first. Doppler uses Q4_K weights/output head, f16 embeddings/KV and f32 computation; Transformers.js uses ONNX q4f16, a different quantization representation. No model-download or browser-launch time is included.

| Hardware | Model | Decode tokens/s | Warm first token ms | Model-open ms |
|---|---|---:|---:|---:|
| Apple M3, 24 GiB, Metal | Qwen3.5 0.8B | 29.80 / 25.75 | 448 / 241 | 3296 / 3266 |
| Apple M3, 24 GiB, Metal | Qwen3.5 2B | 20.46 / 19.27 | 997 / 473 | 5427 / 9300 |
| Radeon 8060S, 128 GB UMA, Vulkan | Qwen3.5 0.8B | 97.66 / 42.21 | 309 / 84 | 3498 / 2351 |
| Radeon 8060S, 128 GB UMA, Vulkan | Qwen3.5 2B | 71.28 / 41.52 | 338 / 257 | 6172 / 3295 |

[Original report, exact hardware/software, source reconstruction and raw receipts](../qwen-competition-20261017/README.md). All 128 generated IDs agree on those synthetic workloads; that is not blanket correctness. This table does not relabel the later candidate's instrumented qualification timings as the original benchmark. MLX and llama.cpp remain faster on several retained native observations; current-build complete native coverage remains open.

## Exact installed candidate

- Runtime source: `8b211c3f`; package `doppler-gpu` 0.6.31, [archive](doppler-gpu-0.6.31.tgz), SHA-256 **`6b057b0e526a7a18afcefd655cb2fc4acb96e958962eee5a67b3caee231be1f9`**. The archive hash, not the reused development version label, identifies this candidate. It was not uploaded over an existing registry version.
- [Packaging receipt](package-receipt.json): 2028 files, 3,501,433 bytes; installed import/type/application smokes passed. Exact model manifests and full configurations are retained in the raw bundle. Models are the same pinned artifacts as the preceding comparison.
- Qualification harness: `4db41ca5`; physical partial-tile test: `201bcbbc`. The browser relay's range fix is in the harness/source at `4db41ca5`, **not inside the older installed archive**. Reploid uses its native whole-model execution surface. An application embedding the older archive's browser relay needs the serving fix separately; this report does not qualify a newer archive by inference.
- Apple M3: Chrome 155.0.8059.80, Node 22.20.0; AMD 8060S: Chrome 146.0.7680.177, Node 22.22.1. Both hosts were shared with existing contributors; this was one owned GPU job per host, not an idle-machine benchmark.

## Physical lifecycle, numerical and memory evidence

All **eight lanes**—two models × two devices × Node/browser—passed two exact 128-token reference repeats, cancellation, same-session reuse, explicit `device.destroy()`, rejection of the old session and recovery on a new device/session with the exact reference tokens. This exercises controlled device loss, not a real driver crash. [Summary](qualification-summary.json), [contract](contract.json), [raw receipts](raw-receipts.tar.gz).

Stream cancellation originally bypassed accounting and left previous-request token counts behind. `8b211c3f` settles counts and aborted status in cleanup, snapshots batching state, and resets each request across streaming, token-ID and prefix-KV generation. The existing behavioral test's fake tokenizer was also updated to support its required incremental decoder. Focused cancellation/parity, output, stop-token and harness tests passed; the former fake-tokenizer failure is closed specifically by these tests, not by a full release run.

The four scalar fused recurrent variants preserve **every output, convolution-state and recurrent-state byte over 128 steps**, with three warmup and twelve measured alternating pairs, against the original source on both devices. [Metal](metal-recurrent.json), [Vulkan](vulkan-recurrent-original.json), [original shader bundle](recurrent-baseline.tar.gz). The first AMD run accidentally used the already-updated baseline; its same-source receipt remains in the raw bundle and is not the original-versus-candidate evidence. The corrected run binds the same original hashes as Metal. Compensated two-component shader sources remain unchanged from the previously retained qualification; this new paired probe covers scalar state only.

Physical partial-tile validation uses `M=19, K=512, N=517`, `TILE_M=8, TILE_N=128`, production configuration/dispatch helpers, CPU dequantized reference arithmetic and NaN-filled output. All rows/columns must be overwritten. Maximum absolute errors were **0.002120 M3 / 0.003907 AMD**, with no NaNs/Infs, within the unchanged existing f16 tolerance of 1.0. The original `M=4,K=1536,N=6144` case also passes on both. [Metal partial log](metal-matmul-partial.log), [AMD partial log](vulkan-matmul-partial.log). Five-variant dispatch contract tests pass; only the f16-activation WideTile variant received this new physical partial-tile run.

Managed buffer-pool allocations and process RSS are both retained and explicitly separate. RSS includes shared mappings and is not physical GPU residency. Cancellation after one streamed chunk had accurate request counts and `stopReason=aborted`; GPU work is not claimed to terminate instantaneously. Node RSS spans the whole fresh process; browser RSS is periodically sampled, so the figures are not identical memory metrics.

## Cold model opening, first request and warm request

These are **single instrumented qualification observations**, not competitor medians: a fresh process/browser profile using provisioned local files, followed by a second request. OS and driver caches were not evicted. The browser rows use the corrected external range server. Browser launch is excluded. Warm decode uses 127/decode-time; no warmup distribution or performance promotion is inferred.

| Device / surface | Model | Model-open ms | First request TTFT ms | Warm TTFT ms | Warm decode tok/s | Peak/sampled RSS GiB |
|---|---|---:|---:|---:|---:|---:|
| M3 Node | 0.8B | 1271 | 564 | 564 | 21.57 | 2.07 |
| M3 Node | 2B | 2745 | 1450 | 1489 | 10.48 | 2.10 |
| M3 browser | 0.8B | 4066 | 527 | 465 | 35.70 | 2.10 |
| M3 browser | 2B | 7790 | 958 | 966 | 21.63 | 2.19 |
| AMD Node | 0.8B | 2067 | 1144 | 1117 | 33.59 | 1.29 |
| AMD Node | 2B | 3305 | 701 | 399 | 28.35 | 1.72 |
| AMD browser | 0.8B | 7866 | 867 | 789 | 22.86 | 1.19 |
| AMD browser | 2B | 15922 | 1112 | 706 | 59.73 | 1.58 |

The variability from the preceding benchmark is material. Shared hosts, sampling, cache state and different runtime surfaces preclude attributing it to a kernel change. The M3 Node 2B run also overlapped CPU-side source archiving; its timing is not an uncontended performance claim.

The qualification HTTP server previously ignored Range and returned complete shards. Correct single-byte-range responses reduce redundant transfer; unsatisfiable ranges return 416, and interrupted streams are closed. Focused real-HTTP range and existing relay tests pass. Mac browser model-open observations changed from **17.1 to 4.1 seconds (0.8B)** and **43.3 to 7.8 seconds (2B)**; AMD 2B changed from **72.4 to 15.9 seconds**. These are before/after serving observations on shared hosts, not a controlled runtime speed ratio. Corrected first-load transfer receipts show about **1.16 GB / 2.45 GB** received for 0.8B/2B, including HTTP overhead and repeated tensor reads. This is model acquisition, not a measurement of KV-cache transfer.

## Prefill diagnosis before further kernel changes

The existing M3 browser 0.8B timestamp profile attributes 449.2 ms wall prefill to 222.8 ms GPU work and 171.0 ms CPU command recording, leaving roughly 55.4 ms outside those counters. Projections account for 171.6 ms of GPU time and recurrent attention 38.4 ms. The narrower tile did not reproduce its Node operator win in browser prefill; it remains unadopted.

The new unprofiled warm browser counters provide a separate boundary check:

| Device / model | Prefill wall ms | CPU recording ms | Submit-wait ms |
|---|---:|---:|---:|
| M3 0.8B | 455 | 195 | 199 |
| M3 2B | 955 | 165 | 619 |
| AMD 0.8B | 779 | 223 | 465 |
| AMD 2B | 699 | 211 | 374 |

Submit wait is not a GPU timestamp and may include queue contention. Wait/readback counters overlap and must not be summed as disjoint phases. Next numerical optimization requires runtime-matched projection/recording attribution and prompt-length scaling; these measurements do not authorize another tile by analogy. KV import/export and prefill handoff latency remain unmeasured here.

## Application quality and 27B disposition

The [Reploid journey](https://github.com/clocksmith/reploid/tree/main/artifacts/model-service-20261017) exercised this exact installed archive with a whole-model M3 contributor and two browser conversations. Recipient approval, follow-up, reload, overlapping requests, isolated cancellation, provider death, reconnect and explicit retry passed; unrelated history survived and requester weights were not fetched. It uncovered and repaired native readiness disappearing during execution (`721cd862`). It is a real application integration, not independent retention or a complete cost comparison against another service.

Fourteen frozen [single-turn tasks](service-tasks.json) scored Reploid 2B **9/14** with repetition penalty 1.1 and **11/14** at 1.0. The identical rendered prompts under independent Transformers.js 4.2.0 q4f16 scored **9/14** ([receipt](tjs-2b-service-control.json), [provisioning hashes](tjs-service-pins.json)). Both still fail code tracing, exact Unicode copying and counting; exact outputs differ. The two formats are not identical quantized tensors. The control is quality evidence, not an apples-to-apples service latency claim. The one-setting counterfactual was isolated; no public generation policy or acceptance threshold was weakened.

The deployed **Qwen3.8 27B Q4_K** provider scored **13/14**, still answering `22` where the code evaluates to `26`. The missing matching **F16 control is now complete for this prompt**. With source revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, identical prompt/tokenizer and independent CPU Transformers equations, the retained self-consistent greedy paths are Q4 `[17,17,248046]` (`22`, stop) versus F16 `[17,21,248046]` (`26`, stop). At the common second-token prefix, Q4 favors `2` by 1.135 logit while F16 favors `6` by 1.758. [Comparison](f16-code-comparison.json), [contracts, full F16 inventory and reference logits](f16-code-control.tar.gz).

This is CPU F32 arithmetic with F16 KV, not Doppler F16 GPU qualification. Both reference paths use the same independent tool; the [matching Q4 contract and raw logits](q4-code-control.tar.gz) are retained too. One-prompt attribution implicates quantization here; it does not establish broad F16 quality or explain every Q4 failure. **27B Q4 remains unaccepted. 256-token chunking is mandatory; unchunked long prefill is not repaired.** Prior packed residency (401 packed matrices; approximately 14.4 GB weights and 18.187 GB peak managed allocation) remains separate evidence. The existing public 27B contributor was not replaced with this new small-model candidate.

## Reproduction and open gates

Use the source lockfile dependencies and provision the exact models; no model download is performed. Extract [raw receipts](raw-receipts.tar.gz) into this report directory to restore recorded configurations. The helper validates archive/manifest/reference hashes, installs the exact dependency offline into a fresh consumer, then runs one lane. It copies the retained packaging receipt as archive provenance; it does not label that copy a new packaging smoke result.

```sh
tar -xzf reports/qwen-service-qualification-20261017/raw-receipts.tar.gz \
  -C reports/qwen-service-qualification-20261017
python3 reports/qwen-service-qualification-20261017/reproduce.py \
  --host metal --surface browser --size 2b --models /absolute/models/local \
  --browser '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
  --output /absolute/new-run
# Use --host vulkan and /usr/bin/google-chrome on the retained AMD host.
# --surface node uses the lockfile's provisioned WebGPU provider.
# --prepare-only verifies/installs inputs without launching a GPU campaign.
node tests/kernels/fused-matmul-q4-widetile-f16a-parity.test.js --partial-tile
node reports/qwen-service-qualification-20261017/tjs-service-control.mjs \
  reports/qwen-service-qualification-20261017/service-tasks.json /absolute/tjs-control.json
```

The TJS control serves the already-provisioned `.cache/qwen-comparator-relay`, rejects remote asset requests and caps execution at 600 seconds. Its supplied-list prompts are not a live catalog or streaming-availability test. Archive and receipt hashes are in [SHA256SUMS](SHA256SUMS).

Still open: broad answer-quality acceptance; a qualified recurring movie-selection workload and five actual users; independent return usage; complete service cost and lifecycle comparison with competitors; arbitrary driver-loss recovery; newer-archive qualification including the relay fix; longer prompts and transfer/import/export timings. Reploid's latest hosted unit/integration failures and skipped browser/package checks remain open. Q2/Q3, Flash-Next, catalog expansion and new distributed mechanisms were not started.

Component: installed generation, serving and verification evidence. Intent: preserved. Acceptance: scoped eight-lane lifecycle, scalar recurrent equality, physical partial tiles and retained failed quality results. Boundary effects: no new execution/application ownership; no universal support, quality or promotion claim.
