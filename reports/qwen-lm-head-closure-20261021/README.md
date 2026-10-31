# Q4_K LM-head shared-memory cache: AMD closure (2026-10-21)

Closes the AMD half of the [0e0c5276 optimization](../qwen-lm-head-optimization-20261019/README.md)
under a pre-registered, balanced, independent-process comparison. **The AMD half fails promotion:**
exact parity and lifecycle pass in v2 and in the restated v3 gate (cancellation, reuse, device
recreation), but there is no complete-request improvement. The kernel change is
removed from the shared default and retained here as [candidate.patch](candidate.patch). The M3 half
runs the same harness on mac-host with [contract-v3-metal.json](contract-v3-metal.json).

## Setup

- amd-host: AMD Ryzen AI MAX+ 395 / Radeon 8060S (RADV STRIX_HALO), Linux 7.0.0-38, Chrome
  146.0.7680.177, Playwright 1.58.2, Node 22.22.1.
- Model: `qwen-3-5-2b-q4k-ehaf16` standard-BLAKE3 variant, local NVMe copy
  `.cache/qwen-2b-standard-blake3-20261018-v2` (same weight bytes as the Mac contract's variant).
  LM head is Q4_K `[248320, 2048]`, so the shared path is eligible (`hidden 2048 ≤ 2048`). The prior
  report's workgroup arithmetic used a 151,936-entry vocabulary; this model has 248,320.
- Arm A: HEAD `0428c15a` with only `lm_head_argmax_q4k.wgsl` and its digest reverted to `0e0c5276^`.
  Arm B: HEAD unchanged. Separate detached worktree; no other difference.
- One sample = one Node process, one fresh headless Chrome, one model opening, one cold and five
  warm complete requests (`generate()` start to iterator end). Ten ABBA/BAAB blocks, 20 processes
  per arm, 20 adjacent pairs. Primary metric: per-process median warm complete request; mean paired
  log ratio with 95% t and 10,000-resample bootstrap intervals ([compare.js](compare.js)).
- Promotion rule (frozen before data): identical tokens and text in every request, clean
  unload/destroy/browser exit in every process, and bootstrap upper bound below 0.

## Results

| Run | Lane | Kernel dispatched? | Warm complete request (B vs A) | Warm decode | Parity | Lifecycle |
|---|---|---|---|---|---|---|
| [v1](amd-20261021/summary.json), [contract](contract.json) | empty runtime config | **No** (`useGreedyLmHeadArgmaxFusion` defaults false) | +0.08% [−0.71, +0.87] | +0.29% [−0.72, +1.26] | 240/240 | 40/40 |
| [v2](amd-20261021-v2/summary.json), [contract](contract-v2.json) | `profiles/qwen-3-5-2b-vulkan-parity` | **Yes**, 31 `lm_head_argmax_phase1_q4k_f32a` dispatches with `USE_FULL_BLOCK_FAST_PATH=1` per request in both arms ([precondition](amd-20261021-v2-precondition/)) | **+0.47% [−0.14, +1.09]**, medians 439.8 vs 441.7 ms | +0.29% [−0.10, +0.69] | 240/240 | 40/40 |
| [v3](amd-20261022-v3/summary.json), [contract](contract-v3.json) | same | **Yes**, 31 per request in both arms ([precondition](amd-20261022-v3-precondition/)) | **−0.33% [−0.70, +0.07]**, medians 440.4 vs 437.7 ms | −0.06% [−0.41, +0.28] | 240/240 | 40/40, including cancellation, reuse and device recreation |

Every request in both runs produced the Mac's retained 32 tokens and text. Cold request, prefill
and opening intervals all include zero (see each `summary.json`).

v1 was a design error, kept as evidence: the retained Mac probe and v1 both loaded without a runtime
profile, and only runtime profiles enable the fused argmax. v1 therefore measured an unchanged
path. v2 was pre-registered as an amendment before its data and checked the dispatch precondition first.

**Kernel timing on AMD is not available from current source.** Timestamp profiling sets operator
diagnostics, which disables the fused argmax (`generator-steps.js` `shouldUseGreedyLmHeadArgmaxFusion`),
so profiled requests use the full-vocabulary GEMV in both arms. The Mac report's 10.1% kernel figure
predates that gating. The v2 decode interval excludes a decode reduction larger than about 0.1%.

## v3: restated gate (2026-10-22)

The owner restated the gate to require exact cancellation, reuse, unload and device recreation
as well as tokens and text. [contract-v3.json](contract-v3.json) amends v2 with those checks and
moves both arms to current main (`36ce6036` source; arm B = A + [candidate.patch](candidate.patch)).
After its timed requests, every process aborts one request from `onToken` after its 4th token,
runs one full request in the same session, unloads, destroys the device, reopens on a new
device and runs one more request. These checks are not part of the timed metric.

- All 40 processes passed. Every cancelled request rejected with `AbortError`, the compat
  facade's [abort contract](../../src/inference/pipelines/text/abort-contract.js), after exactly the
  first 4 expected tokens, and settled 0.7 ms after the abort (median, both arms). The reuse and
  recreated-device requests reproduced the 32 expected tokens and text exactly. Reopening on a
  new device took 6.1 s (median).
- The warm complete request changed by −0.33% [−0.70, +0.07]. The upper bound is not below
  zero, so **v3 also fails promotion**. Decode time, the only phase the kernel touches, changed by
  −0.06% [−0.41, +0.28].
- Together, v2 and v3 bound any AMD complete-request effect within about ±1%, with no
  detectable decode change. The candidate stays an operator-level result on AMD, not a
  release claim.
- Run: 2026-10-22 18:40–18:52 UTC, Chrome 146.0.7680.177, same model copy. Reploid
  participants did no GPU work; Xwayland used 1.1 s and the ChatGPT app 0.95 s of GPU over
  12.4 minutes ([start](amd-20261022-v3-precondition/other-gpu-clients-start.txt),
  [end](amd-20261022-v3-precondition/other-gpu-clients-end.txt)). The `404` per sample is the
  static server's missing root page, also present in v2.
- `kernel-probe.js` now records the profiled third request's failure instead of aborting. Since
  [e667f3ca](../qwen-numerical-localization-20261022/README.md), profiled batched decode on this
  recurrent model rejects when timestamp capacity runs out, so timing on AMD is still unavailable.
  The precondition counts come from the unprofiled request.

## Contention

Two Reploid native participants (Qwen 0.8B/2B) stayed running; their cumulative amdgpu engine time
did not change during either run (`amd-20261021-v2-precondition/other-gpu-clients-*.txt`). The Codex
desktop app used 0.56 s of GPU over the 7.4-minute v2 run. `gpu_busy_percent` readings before each
probe are the decaying average of the previous probe, not another workload.

`probe.js` gained a macOS argument branch while v2 ran (between samples 004 and 005); the Linux
argument list is unchanged.

## Disposition

- `lm_head_argmax_q4k.wgsl` and its digest return to the measured arm A. `kernels:check` and
  `tests/gpu/lm-head-argmax-recorder.test.js` pass.
- The candidate stays in [candidate.patch](candidate.patch). Re-promotion needs a positive
  complete-request bound under this contract on the target platform. A Metal-only win would need an
  explicitly selected variant, not a shared default that is neutral-to-slower on AMD.

## Separate finding for numerical localization

On this prompt both arms' **profiled** request selects token 198 (`\n`) at step 2, where every
unprofiled request selects 13 (`.`); the continuations then diverge
([default-lane-profiled-B.json](amd-20261021-v2-precondition/default-lane-profiled-B.json)). Profiling
changes submission shape and LM-head route. Which route matches a full-precision oracle is not
established here.

## M3 half (mac-host)

The restated order puts M3 first. Because the AMD half was already measured, it ran first here,
which does not change the AMD outcome. Use [contract-v3-metal.json](contract-v3-metal.json):
same harness, same lifecycle checks, and the Metal product profile.

```sh
R=reports/qwen-lm-head-closure-20261021
git worktree add --detach ../lmhead-A HEAD            # baseline = current main
git worktree add --detach ../lmhead-B HEAD && git -C ../lmhead-B apply "$PWD/$R/candidate.patch"
ln -s "$PWD/node_modules" ../lmhead-A/node_modules; ln -s "$PWD/node_modules" ../lmhead-B/node_modules
node $R/kernel-probe.js ../lmhead-A <model> $R/contract-v3-metal.json kernel-A.json
node $R/kernel-probe.js ../lmhead-B <model> $R/contract-v3-metal.json kernel-B.json
node $R/compare.js ../lmhead-A ../lmhead-B <model> <newOutDir> 10 $R/contract-v3-metal.json
```

Before the timed run, confirm the precondition: phase-1 dispatches in `requests[1].pipelineCounts`. A Metal pass alone
can justify only an explicitly selected Metal variant. AMD shows no complete-request gain for a
shared default.

Component: `doppler.runtime-source.gpu.kernels`, `doppler` (evidence).
Intent: preserved.
Acceptance evidence: this directory's summaries (v1, v2, v3), `npm run kernels:check`, `node --test tests/gpu/lm-head-argmax-recorder.test.js`.
Boundary effects: none; no manifest pinned the candidate digest.
