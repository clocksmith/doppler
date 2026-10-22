# Issue triage: Doppler vs Transformers.js, complete application (2026-10-22)

One small application, [examples/issue-triage](../../examples/issue-triage/README.md), suggests
`bug`, `feature` or `question` for a GitHub issue; the maintainer decides. The same application
code (prompts, policy, decision rule) runs on two engines; only the scoring adapter differs.
[tools/product-compare.js](../../examples/issue-triage/tools/product-compare.js) drove every run
and [tools/summarize-compare.js](../../examples/issue-triage/tools/summarize-compare.js) produced
[summary.json](summary.json).

## Setup

- Jupiter: Radeon 8060S (RADV STRIX_HALO), Linux 7.0.0-38, Chrome 146.0.7680.177 headless,
  Node 22.22.1. Engines ran one at a time; order alternated per repetition (3 each).
  Two Reploid native participants stayed running and did no GPU work during the run; the Codex
  desktop app used 0.56 s of GPU over 7.8 minutes ([counters](other-gpu-clients.txt)).
- **Doppler:** installed candidate archive `doppler-gpu-0.6.32.tgz` (`2cc4d07c…`, source
  `c704cc58`) and the test-signed Capsule of the
  [Q4K reranker successor](../qwen-reranker-successor-20261021/README.md), qualified for
  `scoreChoices` on Node and Chrome ([qualification](../choice-scoring-successor-20261021/README.md)).
  Artifacts are installed into OPFS through the application's installation adapter.
- **Transformers.js 4.2.0:** the fp16 ONNX export of the same source checkpoint
  (`Qwen/Qwen3-Reranker-0.6B@e61197ed`, ONNX `e145ff16…`) that passed the earlier
  [rerank oracle](../reranker-comparison/20260907/README.md), plus the source revision's
  `generation_config.json` (`81051cd3…`), which the export omitted. Default browser cache.
- Model files came from a local HTTP origin that counted bytes served. "Offline" blocks that
  origin. Crash recovery kills every Chrome process of the profile mid-request, then relaunches
  offline. Peak memory is sampled every 200 ms from `/proc` (RSS) and amdgpu `fdinfo` (GTT+VRAM).
- Quality: NLBSE'24 issue report classification at revision `2927bc67`
  ([dataset pin](../../examples/issue-triage/dataset.js)); 150 held-out test issues
  (10 per repository × label) and 90 train issues for label-free route centering. Issue text
  is not redistributed; runs record indices, labels and scores only.

## Results ([summary.json](summary.json), [runs](runs/))

| | Doppler | Transformers.js |
|---|---|---|
| Install, browser-only (`--omit=optional`) ([install.json](install.json)) | 1 package, 14.2 MB, 0.95 s | 48 packages, 690 MB, 31.7 s |
| Install, default | 7 packages, 96.7 MB, 3.6 s | 52 packages, 724 MB, 33.1 s |
| Model bytes acquired | 944.6 MB | 1,514.9 MB |
| Open from empty profile (3 runs) | 14.5, 15.0, 14.6 s | 6.4, 6.3, 6.3 s |
| First decision | 932 ms | 618 ms |
| Warm decision, median / p90 (60) | 648 / 1,024 ms | 236 / 400 ms |
| Peak browser RSS (all processes) | 1.99 GB | 7.41 GB |
| Peak GPU memory (GTT+VRAM) | 6.48 GB | 3.10 GB |
| Cancellation 60 ms in: abort → settled | 232 ms (3/3), `GenerationError` `DOPPLER_GENERATION_ABORTED` | 44 ms (3/3), `AbortError` |
| Same session reused after cancellation | identical scores (3/3) | identical scores (3/3) |
| Offline reopen | 13.5 s, identical scores, 0 bytes fetched (3/3) | fails (3/3) |
| Killed mid-request, relaunched offline | 14.4 s to first decision, identical (3/3) | fails (3/3) |
| Test accuracy, raw / label-free centered | 0.573 / 0.607 | 0.613 / 0.653 |
| Test macro-F1, raw / centered | 0.523 / 0.592 | 0.563 / 0.643 |

Doppler minus Transformers.js test accuracy: −0.040 raw (95% paired bootstrap −0.10 to +0.02),
−0.047 centered (−0.10 to +0.007); routes agree on 126 and 130 of 150. The representations differ
(Q4K attention projections vs fp16 everywhere). The published NLBSE SetFit baseline, a trained
classifier, reports cross-repository F1 0.827; it was not reproduced here.

Transformers.js offline failure: Chrome rejected the 1.5 GB `model_fp16.onnx` Cache Storage write
(`UnknownError: Unexpected internal error`, quota 39.6 GB); Transformers.js logs a warning and
continues, so reopening refetches the file. A custom storage layer could change this.
Transformers.js also loaded its ONNX Runtime WebAssembly from the jsDelivr CDN by default.

## Reading

Transformers.js is faster to open (2.3×), faster per decision (2.7×), settles cancellation sooner,
uses less GPU memory and scores somewhat better on this task. Doppler installs as one 14 MB
package, acquires 38% fewer bytes, uses 3.7× less process memory, and is the only one that reopened
offline and recovered from a crash with identical outputs. Neither engine reaches a trained
classifier's quality zero-shot. This is one device and one model; it is not a general ranking.

Open Doppler walls, measured here: reopening from local storage takes 13.5 s with no network,
and GPU memory is about twice the competitor's for a 0.6B model.

Follow-up attribution: the amdgpu GTT+VRAM figure is device-global per process and includes
Dawn upload staging and reclaim lag. Wrapping `GPUDevice.createBuffer` during an offline reopen
and one decision found 1.49 GiB of live Doppler buffers (2.79 GiB created over the run): the
296 MiB F16 embedding table, 448 MiB of KV, the F16 FFN weights and Q4_K projections
([doppler-gpu-buffers.json](doppler-gpu-buffers.json)). About 10 s of the reopen was JavaScript
SHA-256 and BLAKE3 re-hashing; [be482a91](../capsule-open-hashing-20261022/README.md) reduced the
reopen to 8.5-8.7 s after this comparison was run. The table above is unchanged.

Component: `doppler.examples` (reference application), `doppler` (evidence).
Intent: preserved.
Acceptance evidence: this directory; `examples/issue-triage/tools/product-compare.js` with `config.json`.
Boundary effects: none.
