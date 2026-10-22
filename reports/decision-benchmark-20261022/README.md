# Decision benchmark: issue triage, Linux half (2026-10-22)

The owner asked for a head-to-head benchmark before more packaging work and will choose the
workload from it. This is the Jupiter (Linux, AMD) half under the pre-registered
[contract](contract.json); the Mars half (Safari, macOS Chrome and Firefox, Node on M3) uses the
same harness ([below](#mars-half)). Two comparisons stay separate: (a) runtime parity, the same
model and weights on Doppler and Transformers.js; (b) product, Doppler with its chosen model
against the best open alternative.

## Setup

- Task: route 150 held-out NLBSE'24 GitHub issues to bug, feature or question (10 per repository
  × label); 90 training issues are used only for label-free centering and for choosing the best
  open alternative. Every lane sees the same issue text (title plus 1,600 body characters).
- Harness: [examples/issue-triage/benchmark](../../examples/issue-triage/benchmark/). One page per
  run reports to a local server. That server serves the application, the issues and model files,
  and counts model bytes. Each run launches a fresh browser profile; memory is sampled every
  200 ms over the browser's process tree. Lanes ran one at a time, in alternating order, 3
  repetitions each; the first repetition decided all 240 issues, later ones the first 10.
- Machine: Radeon 8060S (RADV STRIX_HALO, Mesa 26.0.3), Linux 7.0.0-38. Google Chrome
  146.0.7680.177 (Vulkan); Firefox Nightly 160.0a1, build `20261022084508` (source
  `00616311`); Node 22.22.1. Reploid's participants did no GPU work during the run; Xwayland
  used 17 s and the ChatGPT app 9 s of GPU time over the whole run
  ([start](other-gpu-clients-start.txt), [end](other-gpu-clients-end.txt), [platform](platform.json)).
- Versions: Doppler 0.6.32 candidate archive `3f87f551…` with the test-signed Capsule
  `…-92b23b6a7b37aa70` ([qualification](../choice-scoring-successor-20261022/README.md));
  Transformers.js 4.3.1 with onnxruntime-web 1.31.0-dev.20260914 and onnxruntime-node 1.30.0;
  open-jev 0.1.2.

### Lanes

| Lane | Comparison | Model and precision |
|---|---|---|
| `doppler-capsule-q4k` | product | Qwen3-Reranker-0.6B, Q4K attention / F16 FFN, signed Capsule installed into OPFS (Node: compat `load()` of the same manifest, the path Node qualification uses) |
| `doppler-f16` | parity | Qwen3-Reranker-0.6B F16 weights, **F32 activations**, compat `load()` |
| `transformersjs-fp16` | parity | the same checkpoint's fp16 ONNX export (fp16 weights and **fp16 activations** on WebGPU) |
| `open-jev-kev-0.6b`, `open-jev-kev-4b` | product | Qwen3-0.6B/4B-Base decision fine-tunes, q4f16 on WebGPU, q4 on CPU, open-jev's own choice |
| `open-jev-deberta` | product | DeBERTa-v3-large decision model, fp16 on WebGPU, q4 on CPU |

Reranker lanes score each route with the application's prompt and use softmax(yes − no) over
routes as probabilities. open-jev lanes ask one typed question: `choice("What type of GitHub
issue is this?", [bug, feature, question])`, with the route descriptions. Models were fetched at
pinned revisions with every LFS file checked against its SHA-256
([fetch-models.py](../../examples/issue-triage/benchmark/fetch-models.py)).

The F16 artifact is a fresh conversion of the pinned source safetensors
(`model.safetensors` LFS `27cd75a4…`). The conversion configuration now uses the tiled F16
matmul and head-128 prefill attention, as the Q4K reranker already does (variant
`…-mv-exec-v2-tiled-head128`). Its weights are bit-identical to an independent numpy
round-to-nearest BF16→F16 conversion of the source. The July copy on the external drive differs
from it by up to 6e-8.

## Results ([summary.json](summary.json), [table](summary.md), [runs](runs/))

| Linux | Test accuracy | Macro-F1 | ECE / Brier (raw) | Cold, launch to first decision | Warm decision, median / p90 | Peak RSS | GPU memory | Model bytes |
|---|---|---|---|---|---|---|---|---|
| **Chrome** doppler-capsule-q4k | 0.573 | 0.523 | 0.218 / 0.621 | 11.2 s | 542 / 781 ms | 2.42 GiB | 5.99 GiB | 945 MB |
| Chrome doppler-f16 | 0.587 | 0.528 | 0.180 / 0.563 | 4.6 s | 578 / 829 ms | 1.48 GiB | 7.07 GiB | 1,208 MB |
| Chrome transformersjs-fp16 | 0.613 | 0.563 | 0.160 / 0.557 | 6.9 s | 255 / 367 ms | 7.80 GiB | 3.40 GiB | 1,515 MB |
| Chrome open-jev-kev-0.6b | 0.567 | 0.518 | 0.275 / 0.660 | 3.0 s | 83 / 118 ms | 2.23 GiB | 0.94 GiB | 346 MB |
| Chrome open-jev-deberta | 0.580 | 0.480 | 0.136 / 0.587 | 3.6 s | 81 / 86 ms | 3.43 GiB | 1.98 GiB | 884 MB |
| Chrome **open-jev-kev-4b** | **0.720** | **0.697** | 0.187 / 0.469 | 7.7 s | 356 / 553 ms | 5.32 GiB | 4.94 GiB | 2,277 MB |
| Firefox Nightly, Doppler (both lanes) | fails | | | | | | | |
| Firefox Nightly transformersjs-fp16 | 0.613 | 0.563 | 0.160 / 0.557 | 9.4 s | 501 / 660 ms | 8.20 GiB | 3.95 GiB | 1,515 MB |
| Firefox Nightly open-jev-kev-0.6b | 0.580 | 0.530 | 0.294 / 0.661 | 4.1 s | 699 / 703 ms | 2.60 GiB | 1.68 GiB | 346 MB |
| Firefox Nightly open-jev-deberta | fails | | | | | | | |
| Firefox Nightly open-jev-kev-4b | 0.720 | 0.697 | 0.193 / 0.472 | 8.9 s | 1,115 / 1,529 ms | 7.48 GiB | 5.65 GiB | 2,277 MB |
| **Node** doppler-q4k (Dawn WebGPU) | 0.573 | 0.523 | 0.218 / 0.621 | 2.0 s | 589 / 920 ms | 0.57 GiB | 5.37 GiB | local |
| Node doppler-f16 (Dawn WebGPU) | 0.587 | 0.528 | 0.180 / 0.563 | 2.2 s | 641 / 963 ms | 0.59 GiB | 6.50 GiB | local |
| Node transformersjs-fp16, WebGPU | fails | | | | | | | |
| Node transformersjs-fp16, CPU | 0.587 | 0.528 | 0.180 / 0.563 | 6.3 s | 1,756 / 3,193 ms | 3.59 GiB | – | local |
| Node open-jev-kev-0.6b (CPU q4, its default) | 0.587 | 0.536 | 0.289 / 0.662 | 2.0 s | 463 / 854 ms | 0.75 GiB | – | local |
| Node open-jev-deberta (CPU q4) | 0.607 | 0.528 | 0.140 / 0.578 | 1.7 s | 494 / 613 ms | 1.20 GiB | – | local |
| Node open-jev-kev-4b (CPU q4) | 0.720 | 0.697 | 0.193 / 0.471 | 5.6 s | 1,899 / 3,184 ms | 3.53 GiB | – | local |

Every lane that ran was deterministic: later repetitions reproduced the first 10 decisions
exactly. GPU memory is amdgpu GTT+VRAM per process, which includes allocator staging. RSS covers
every process of the browser profile.

### (a) Runtime parity: Qwen3-Reranker-0.6B, F16 weights

- **Doppler matches the full-precision model, and Transformers.js on WebGPU does not.** On the
  40-issue [numerics subset](../issue-triage-numerics-20261022/README.md), Doppler F16 differs from
  the FP32 ONNX oracle by a median 0.003 logits (max 0.011); Transformers.js fp16 on WebGPU differs
  by 0.32 (max 1.02). Transformers.js fp16 on onnxruntime-node CPU agrees with Doppler F16 on all
  150 routes, with route scores within 0.022.
- On Chrome, Transformers.js leads by 2.7 points (95% paired bootstrap −5.3 to −0.7 for Doppler
  minus Transformers.js), from 4 of 150 issues. All four are near-ties (top-two margin 0.02–0.13).
  On the three in the oracle subset, the FP32 model chooses what Doppler chose. The lead comes from
  fp16-activation rounding that happens to land on the correct label, not from a better model.
- Precision is not fully identical. Transformers.js's WebGPU graph computes in fp16, and Doppler
  computes this model in F32. Requesting `activationDtype: f16` at runtime had no effect: the af32
  manifest's execution plan overrides it. Doppler has no af16 lane for this model.
- Speed and memory: Transformers.js is 2.3× faster per decision (255 vs 578 ms). Doppler opens
  faster from an empty profile (4.6 vs 6.9 s), uses 5.3× less browser RSS (1.48 vs 7.80 GiB) but
  2.1× more GPU memory (7.07 vs 3.40 GiB), and fetches 20% fewer bytes.
- With label-free centering on the training split, Doppler F16 reaches 0.667 (ECE 0.071) and
  Transformers.js 0.660 (ECE 0.054).

### (b) Product: Doppler Q4K Capsule against the best open alternative

- The best open alternative by training accuracy is **open-jev with kev-4b**. On Chrome it beats
  Doppler by **+14.7 points** (0.720 vs 0.573; 95% paired bootstrap +8.0 to +21.3), with the same
  result on Firefox Nightly and Node. It is also 1.5× faster per decision on Chrome (356 vs
  542 ms) and opens sooner (7.7 vs 11.2 s). It costs 2.4× the download (2.28 GB vs 0.95 GB) and
  2.2× the browser RSS (5.32 vs 2.42 GiB).
- **kev-0.6b ties Doppler** on accuracy (0.567 vs 0.573; −0.7 points, −8.0 to +6.7), while being
  6.5× faster per decision (83 ms), 3.7× faster to open (3.0 s), downloading 2.7× less (346 MB)
  and using 6× less GPU memory. deberta also ties on accuracy (0.580) at 81 ms per decision.
- Why Doppler is slower here: the application asks three yes/no questions, each a full
  reranker forward over the issue text. open-jev answers all options in one forward pass. The
  model-per-forward speed gap from (a) compounds with this 3× structural difference.
- Calibration (raw): kev-4b is overconfident (mean confidence 0.90 against accuracy 0.72, ECE
  0.187); deberta is the best calibrated (ECE 0.136); Doppler Q4K's raw softmax has ECE 0.218,
  0.093 after label-free centering.
- Doppler Q4K on Node is the smallest resident footprint measured: 0.57 GiB RSS and a 2.0 s
  cold start. Its 5.4 GiB of GPU memory is a measured wall; the live buffers at a reopen are
  1.49 GiB ([attribution](../issue-triage-comparison-20261022/README.md)).

### GPU provider and CPU fallback

| | Chrome 146 (Vulkan) | Firefox Nightly 160 (wgpu) | Node 22 |
|---|---|---|---|
| Doppler | Runs; adapter `amd / rdna-3` | **Fails.** The adapter has no `subgroups`. The Capsule refuses before downloading (`no prequalified target plan … hasSubgroups: false`). Compat F16 downloads 1.2 GB and then fails to compile `rmsnorm_default` (Naga: `enable subgroups` not supported) | Runs on Dawn (`webgpu@0.4.0`, RADV) |
| Transformers.js fp16 | Runs (WebGPU) | Runs (WebGPU) | **WebGPU fails** (`"present.0.key" is not a valid output name`); CPU runs |
| open-jev | All three run | kev-0.6b and kev-4b run; **deberta fails** (`ShaderModule with 'Clip' label is invalid`) | Defaults to CPU q4 even though onnxruntime-node bundles WebGPU |
| Without WebGPU (page removes `navigator.gpu`) | Doppler fails explicitly (`WebGPU is unavailable`). The application's Transformers.js adapter fails (`The device (webgpu) does not support fp16`). open-jev's WebAssembly fallback **fails** for kev-0.6b and kev-4b (`GatherBlockQuantized` has no WebAssembly kernel) and **hung** for deberta (20 min at session creation) | Same, except deberta fails fast with the same `GatherBlockQuantized` error | n/a |

Firefox exposes empty adapter info strings. Neither engine has a working browser CPU path for
these models with these packages.

## Defects found along the way

- **Fixed:** `examples/issue-triage/engines.js` read fp16 logits as raw `Uint16Array` bits on
  Node 22, which has no `Float16Array`. Choices mostly survived because positive values keep their
  order, but scores and probabilities were wrong. It now converts to float32. The Node CPU lane was
  re-run; the defective runs are not in `runs/`.
- **Open, Doppler:** `rmsnorm.wgsl` and `rmsnorm_pair.wgsl` declare `enable subgroups` at module
  scope, while their non-subgroup registry variants declare `requires: []`. So every model needs
  WebGPU subgroups, and a device without them fails at shader compilation rather than at
  selection. Fixes: split the non-subgroup entry points into modules without the extension (this
  changes kernel digests and every manifest pin), or declare the requirement so selection fails
  early with an actionable message.
- **Open, Doppler:** `runtime/default.json` sets `inference.compute.activationDtype: "f16"`, and an
  af32 manifest's execution plan overrides any runtime value, so an explicit f16 request is
  silently ignored. Lane-mismatch guards exist only for named Gemma 4 manifests.
- **Not reproduced:** the first full run's `doppler-f16` Chrome lane stalled for 30 minutes after
  the Capsule lane. It ran normally alone and in the complete re-run. Runs now send progress
  heartbeats, so a repeat would show where it stopped.

## Not run

- TypeSafe Jev: a hosted, paid API with new signups paused. Running it would send issue text to a
  third party; it needs an owner-provided account and approval.
- Chrome Decisions API: not present in Chrome 146.0.7680.177 (no `decisions-api` flag in the
  binary). The explainer says it is not approved to ship.
- Laya builds (kevala, @r4ai/laya-web, open-jev-laya) and MediaPipe DecisionMaker were not run.

## Mars half

Copy `dist/qual-0632-20261022/` (archive, Capsule, descriptor) and the F16 conversion from
Jupiter, or rebuild them with the commands in the qualification report. Then:

```sh
cd examples/issue-triage/benchmark && npm ci
python3 -I fetch-models.py <models root>   # plus the reranker fp16 export for transformersjs-fp16
cp config.example.json config.local.json   # fill paths and the Safari version
node run.js config.local.json              # Chrome, Firefox, Safari (manual kind)
node node-run.js <node config>             # same keys as the Linux node config
node summarize.js <outDir> summary.json
```

Safari has no per-run fresh profile from the command line, and the launcher cannot sample
Safari's memory from the `open` process. Clear website data between cold runs, or use a Safari
WebDriver session, and record memory separately. The Capsule's only qualified target is
`webgpu-f32-f16-subgroups`, so Apple devices whose WebGPU lacks subgroups will refuse it, as
Firefox did.

Component: `doppler.examples` (benchmark harness, application adapter fix), `doppler.runtime-source.config` (F16 reranker routing), `doppler` (evidence).
Intent: preserved.
Acceptance evidence: this directory; `examples/issue-triage/benchmark/summarize.js runs summary.json`.
Boundary effects: none; the F16 reranker configuration has no checked-in manifest, catalog entry or qualification fixture.
