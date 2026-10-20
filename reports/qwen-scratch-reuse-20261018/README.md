# Qwen 2B bounded decode scratch experiment — rejected

One session-owned intermediate removed allocations but did not establish a
complete-request gain. The runtime patch was removed. The patch, exact candidate
archive, failed preparation attempt, raw timings, and physical lifecycle evidence
remain reproducible. Bind-group caching stayed disabled. Reploid was untouched.

[Summary](summary.json), [frozen contract](contract.json),
[identities](identities.json), [candidate patch](candidate.patch), and
[Vulkan successor qualification](vulkan-qualification.json) own this result.
This is an internal optimization experiment, not a vendor comparison or a
promoted package. No general arena or new configuration surface was added.

## Frozen identities and workload

- Baseline source: `0e75e3c4f51c806c8af6d8bdf473b063fe62532f`.
- Baseline npm archive SHA-256:
  `2662a8b9923dbeb4a455556bf6536a5b1291595a4f5846800eff312f6c24c60f`.
- Candidate archive SHA-256:
  `addf3788637a68ed929c20445d7a540e6a5bd417811284e8a5a11737b6c4284d`.
  Its inventory differs in exactly `ffn/dense.js` and `ffn/dense-executor.js`.
- Qwen3.5-2B Q4_K successor manifest SHA-256:
  `c3364f7fecc01b70ce63b4de3347d8305a1a6437ac63bb943edece7fa279fc0e`.
  F16 embeddings/KV, F32 activation/compute lane, 2,048 context.
  Weight/tokenizer bytes are unchanged from the retained predecessor. This is an
  unsigned local qualification artifact, not a catalog release.
- M3, Chrome 155.0.8059.80, Metal, Apple driver 26.6.2 (Build 25G83).
- Eight opening pairs, alternating baseline/candidate then candidate/baseline;
  one first request plus five warm requests per opening. Prompt:
  `The capital of France is`; greedy 32 output tokens, 31 decode tokens,
  no chat template, seed 0, unchanged sampling and 16-token delivery grouping.
- Each arm opens a fresh browser and loads via HTTP. OS disk caches were not
  flushed. Opening is measured separately and is not an OS-cache-cold claim.
  The host was nonexclusive; no unrelated services were stopped.
- Primary timing uses existing runtime counters and the same 500 ms browser
  process CPU/RSS sampler in both arms. Allocation attribution uses a separate
  instrumented pass and is excluded from the primary timings.

## Lifetime trace and bounded change

| Resource | Allocation/owner | Final relevant use and safe reuse |
| --- | --- | --- |
| Gate/up input | Post-attention norm output; pool/recorder | Fused gate/up reads it; ordinary recorder cleanup waits for completion. Unchanged. |
| `ffn_down` input | `fused_ffn.executeFusedFFN` acquires `fused_ffn_output`; `dense.js` tracks it | Ordinary path reads it in down projection, then releases after recorded submission completion. Diagnostic completion tasks, captures, or LoRA can extend its lifetime. |
| `ffn_down` output | Matmul output resolver acquires a pooled output | Residual add, cast, fused next norm, or following layer consumes it. It can escape the FFN helper. Left unchanged. |
| Weights | Loader owns immutable gate/up/down buffers | Model unload after operation leases settle. Left unchanged. |
| Uniforms | Existing recorder/content-addressed uniform machinery | No content, ownership, or binding-cache changes in this experiment. |
| Candidate intermediate | Existing `DecodeBufferManager.ffnIntermediate`, one per session | Down read precedes the next gate/up write on the same ordered GPU queue. Session teardown destroys it only after active operations settle. |

Source owners: [dense FFN](../../src/inference/pipelines/text/ffn/dense.js),
[fused gate/up adapter](../../src/inference/pipelines/text/ffn/dense-executor.js),
[kernel allocation](../../src/gpu/kernels/fused_ffn.js),
[residual consumer](../../src/inference/pipelines/text/ffn/standard.js),
[decode-buffer owner](../../src/inference/decode-buffers.js),
[recorder cleanup](../../src/gpu/command-recorder.js), and
[operation/unload leases](../../src/inference/pipelines/shader-scoped-pipeline.js).
Links describe the unchanged control; the retained patch specifies the candidate.

The candidate forwarded that already allocated intermediate through the existing
kernel `outputBuffer` argument. It applied only to recorded single-token Q4_K
F32 decode with sufficient capacity, no paired representation, adapters,
training capture, probes, active operator diagnostics, TSIR capture, or deferred
logit-health tracing. The norm-statistics path returned before the new path.
Other paths retained the allocator. No shader, dtype, graph, submission cadence,
or CPU buffer write changed.

Queue order makes later GPU writes follow earlier GPU reads even while those
commands are pending. This does not authorize a CPU overwrite, relocation, or
destruction before completion. The raw session buffer is not pool-owned and is
not reclaimed by recorder temporary cleanup. Reset retains it; session unload or
device recreation ends its ownership. Separate sessions own separate slots.
Public overlapping operations remain serialized by the current compatibility
scope; these tests do not establish simultaneous multi-session GPU execution.

## Complete-request comparison

Values below are medians of eight opening-level warm medians, five requests each.
They are not forty independent opening samples. Raw observations, including
stalls and unfavorable pairs, are retained.

| Metric | Allocator control | Scratch candidate |
| --- | ---: | ---: |
| Opening, ms | 4,320.8 | 4,528.9 |
| First delivered output, ms | 113.85 | 114.75 |
| Complete request, ms | 863.35 | 888.00 |
| Decode, tokens/s | 41.62 | 40.59 |
| CPU decode recording, ms | 163.15 | 154.40 |
| New pool allocations/request | 7,202 | 6,458 |
| Peak pooled allocation, MiB | 2,052.28 | 2,040.28 |
| Peak sampled process RSS sum, MiB | 3,502.36 | 3,540.87 |

The primary estimator is paired complete-request percentage gain, not the ratio
of the table's marginal medians. Its median was **−0.14%**, with a fixed
20,000-resample opening-pair bootstrap 95% interval **[−5.43%, +10.24%]**.
Four pairs favored each arm. The predeclared requirement was at least 2% median
gain with a positive lower bound and no material first-output, decode, memory,
or lifecycle regression. The improvement requirement failed; no optional
extra sampling was used to seek a favorable result.

The separate control attribution pass observed 4,464 `fused_ffn_output` pool
acquisitions across six requests: 11.1 ms synchronous JavaScript call time.
Of these, 4,420 created new GPU buffers, consuming 5.0 ms inside `createBuffer`
(nested in acquisition, not additive). The candidate replaced all 4,464 with
one existing session buffer. Timer resolution and deferred driver/GPU work
limit attribution: these API times are not complete allocation costs.
The matched complete requests remain the decision evidence.

Peak pool allocation fell 12 MiB, while retained weights were unchanged. Pool
counters exclude raw session buffers, KV allocations outside the pool, and
opaque driver allocation. The reused slot was already allocated in both arms.
Process RSS sums can double-count shared pages and are not physical VRAM.

## Numerical and lifecycle evidence

- All 96 timed/first M3 generations matched the control's token IDs and text.
  Every opening also matched the full public-prefill-logit SHA-256.
- The candidate lifecycle journey matched short and 833-token long-prefill
  outputs and the public logit hash; cancellation followed by reuse passed.
- Two resident sessions used different prompts. Overlapping requests, isolated
  cancellation, unrelated progress, and subsequent reuse passed.
- Unload requested during a stream waited for that stream, preserved its exact
  answer, and did not change the other session's answer.
- Device destruction after submission rejected the operation. Cleanup,
  recreation, and identical regenerated output passed. This is deliberate
  destruction, not a claim about every spontaneous hardware-loss mode.
- Focused decode-buffer representation/ownership test, syntax checks, and source
  architecture check passed. The source-style check was run after reverting the
  candidate. No broad repository CI result or optimized release is claimed.

## Physical Vulkan successor closure

The unchanged baseline archive ran the same lifecycle journey on physical
Radeon 8060S, RADV 26.0.3, Vulkan 1.4.335. The source bytes were copied locally
on AMD, preserving historical files. All 33 files matched independently retained
SHA-256 identities and the frozen standard-BLAKE3 implementation. That BLAKE3
implementation had already matched independent Python BLAKE3 on M3; AMD did not
have a second independent BLAKE3 library installed.

Short/long tokens and text plus the full public logit hash exactly matched the
retained **same-device** predecessor receipt under the same configuration.
Short/long tokens also matched M3, but Metal/Vulkan public-logit byte hashes
were different. No cross-device bit-exact logit claim is made. Cancellation,
overlapping requests, in-flight unload, cleanup, and device recreation passed.
This closes this successor's physical Vulkan journey, not AMD scratch speed or
universal model correctness. Lavapipe evidence remains a separate class.

The first preparation attempt failed because the verifier compared a Uint8Array
hash with a hex string. No source artifact changed. The failed log and receipt
are retained; the corrected verifier used a fresh destination. No model weights
were downloaded or historical manifests rewritten.

## Next measured target

Close this bounded reuse campaign. Do not add bind-group caching: although the
FFN input identity stabilized, its output still changed, and no complete-request
benefit survived the fixed test. Do not infer that all scratch reuse is useless.

The control's decode wait was dominated by map/fence completion: 586.75 ms,
versus 4.30 ms cleanup and 0.10 ms copy. Submission wait (586.90 ms) and overall
readback wait (592.05 ms) overlap and must not be summed. Recording took
163.15 ms; decode recorded 6,293 dispatches in eight passes per request.
The next diagnostic should attribute actual GPU execution inside that fence to
operators before changing another kernel or buffer policy. A long fence is not
evidence that readback copying itself is slow. Preserve the rejected four-token
delivery experiment and unchanged precision/grouping.

## Evidence and reproduction

[Raw bundle](raw-evidence.tar.gz) retains the candidate npm archive, primary
receipts, attribution/lifecycle receipts, progress log, exact Metal config,
Vulkan config/receipts, preparation failures, and verification logs.
[SHA256SUMS](SHA256SUMS) pins the report and bundle. No weights are included.
The baseline archive and successor provisioning instructions remain in the
[preceding report](../qwen-bind-group-reuse-20261017/README.md#evidence-and-reproduction).

Extract the bundle into this report directory. Use installed Playwright/Chrome,
provisioned successor weights, and fresh extraction/output directories:

```sh
python3 reports/weight-representation-20261017/extract-runtime.py reports/qwen-scratch-reuse-20261018/doppler-gpu-0.6.32.tgz .cache/qwen-scratch-candidate-20261018
node reports/qwen-scratch-reuse-20261018/run.js .cache/qwen-scratch-candidate-20261018 .cache/qwen-2b-standard-blake3-20261017 reports/qwen-scratch-reuse-20261018/metal-config.json /tmp/new-scratch-lifecycle.json lifecycle
python3 reports/qwen-scratch-reuse-20261018/summarize.py
```

`run.js` modes are `measure`, `probe`, and `lifecycle`. It verifies served runtime
files against the extracted archive inventory. `compare.py` reproduces the fixed
sequence using the documented cache paths and refuses to overwrite existing
pair receipts; archive old receipts elsewhere before an intentional new campaign.
The reducer checks archive/config/browser/device/driver identities and exact
outputs, then evaluates the frozen acceptance without excluding slow samples.
Its last command reduces this retained campaign, not arbitrary new runs.

Component: `doppler.runtime-source.inference.pipelines.text` (candidate),
`doppler.docs` and verification (retained result). Intent preserved. Acceptance
evidence: frozen artifacts above. Boundary effects: none; runtime restored to
control, no Reploid change, no new public API or release promotion.
