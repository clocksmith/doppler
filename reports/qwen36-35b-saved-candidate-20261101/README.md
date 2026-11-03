# Qwen 3.6 35B: preserve the best saved request-time candidate

Status: retained performance candidate, with qualification still open. This
change saves the `s18-prefill-kernels-candidate` implementation that was already
measured on Jupiter. It does not establish a matched Pooled win or promote a
model artifact. The existing experimental profile remains the opt-in entry:
`profiles/qwen3-6-35b-a3b-q4_0-candidate`.

## Selection and measurements

`s18` has the lowest saved, uninstrumented Doppler mean request time for both
the short and long throughput prompts in `deco-evidence/doppler-35b-perf`.
Selection uses the whole request, rather than the fastest isolated decode
sample. The saved receipt has one warmup and two measured requests per prompt;
these are diagnostic point estimates, without a paired confidence interval.

| Prompt | First token | Prefill | Inter-token decode | 128-token request |
| --- | ---: | ---: | ---: | ---: |
| Short, 28 input tokens | 137.05 ms | 133.95 ms | 13.236 ms/token | 1818.70 ms |
| Long, 2287 input tokens | 4065.30 ms | 4060.10 ms | 14.371 ms/token | 5891.10 ms |

Each value is the arithmetic mean over non-warmup attempts. Inter-token decode
uses `(last delivery - first delivery) / 127`. Both requests stop at the
128-token budget; they do not measure naturally completed answers.

The preceding `s17` candidate has mean request times of 1853.20 ms short and
6076.05 ms long. `s18` reduces those point estimates by 1.86% and 3.04%,
respectively. Earlier candidates have lower long-context decode estimates;
`s18` is the selected request-time candidate, not a winner on every metric.
It was measured separately from `s17`, not in alternating baseline/candidate
pairs, so these deltas do not establish an improvement beyond run variance.

Scope: Qwen3.6-35B-A3B source GGUF blocks, F32 activations and KV, greedy
single-token decode, speculation disabled, 256-token prefill chunks, Chrome
WebGPU on Jupiter's AMD/RADV lane. The retained ready receipt exposes the
actual adapter, feature list, and observed weight materializations. The model
is already resident for these request measurements: `s18` reports 241.67
seconds to ready, including approximately 171.70 seconds of configured warm
generation. That startup cost is separate from first-token latency.

## Saved implementation

- Coalesced activation and packed-weight staging in the native GGUF tile64
  prefill matmul; ordered F32 accumulation and native block decoding remain.
- Coalesced F32 matmul loads with padded workgroup tiles for both weight layouts.
- Shared normalized q/k in GGUF register recurrent prefill, plus the registered
  prefix/parallel convolution mechanism. Short token sequences retain the
  prefix-only dispatch; immediate and recorded adapters use the same guard.
- Generated shader patch, kernel registry metadata, shader digest table, and
  existing conversion/profile digest references stay aligned. Conversion
  changes update content pins only; no artifact or catalog is promoted.

The checkout's 2976 source files matched the saved `s18` source snapshot before
this commit. `measurements.json` records the runtime-source tree hash excluding
the unrelated package-ignore file, individual changed-file hashes, raw timing
values, and saved-output comparisons. No benchmark was rerun for this retention
change.

## Correctness and development checks

Every saved `s18` attempt for records, missing, short, and long has identical
prompt IDs, generated IDs, and decoded text to the first saved `s17` attempt
for that task, with no reported GPU errors. The records/missing task assertions
pass. This is output-continuation evidence, not independent full-logit or
per-layer numerical qualification.

Focused checks run while preserving the candidate:

- `flock -n /run/media/x/models/doe-gpu-timing.lock node tests/gpu/q4_0-kernels-gpu.test.js`
  passes: tile64 output is bit-identical to tiled baseline for Q4_0, Q4_1,
  Q5_0, Q8_0, and Q6_K across both tested tail geometries. F32 weight layouts
  agree bitwise and match the CPU reference within the existing tolerance.
- `flock -n /run/media/x/models/doe-gpu-timing.lock node tests/gpu/gguf-linear-attention-register-decode-gpu.test.js`
  passes: seven-token prefill and three following decode steps preserve output,
  recurrent-state, and convolution-state bits versus the unfused path.
- Both tests use the Node `webgpu` provider and Mesa software adapter available
  to this session. They are shader correctness checks, not new physical GPU
  timing or browser qualification.
- `npm run kernels:check`, `npm run kernels:package-digests:check`, and
  `npm run config:single-source:check` pass. The kernel check includes uniforms,
  generated variants, registry reachability, digests, and language contracts.
- `npm run source:style:check` passes, including unused-local and source-boundary
  checks.

## Open comparison gates

The earlier `f5-paired-short` receipt alternates both products, but uses native
Pooled flash/F16 KV against Doppler F32 KV. It is not a matched-settings proof
and predates `s18`. The latest candidate has no fresh paired Pooled run.
Neither speculation nor complete-answer latency is established here.
The independent CPU-reference, router-kernel, and generated-position routing
gaps in the [numerical report](../qwen36-35b-pooled-numerics-20261031/README.md)
remain open. No tolerance or comparison gate is weakened by this commit.

Raw evidence remains on the external model volume; `retained-files.json`
binds the receipts, execution settings, ready records, and all attempts by
SHA-256. `measurements.json` is the compact derived summary.

Component: `doppler.runtime-source.gpu.kernels`, `doppler.runtime-source.config`,
`doppler.tests`. Intent: preserved. Acceptance evidence: commands and retained
receipts above. Boundary effects: none.
