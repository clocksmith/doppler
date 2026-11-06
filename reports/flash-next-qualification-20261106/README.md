# Flash-Next: retained gate, upstream drift and F32 product rounding

Normal long-input qualification remains failing. The owner selected tracing
upstream drift under the existing gate. No routing-aware acceptance policy,
tolerance change, CPU tensor substitution, dtype switch, recipe projection
promotion or performance claim is made here.

## Repaired arithmetic boundary

On the exact captured layer-5 inputs, the original HC branch outputs match a
fused multiply-add pattern and differ from the pinned source's separate-product
mean. The old `multiply_ordered` helper did not preserve the intended product
rounding. See [mixing-order analysis](hc-mixing-order.json).

The branch and injection shaders now exchange F32 products between adjacent
invocations through workgroup memory before addition. This establishes a real
rounding boundary using one workgroup-sized scratch array; it allocates no
additional model buffer and preserves stream order and dispatch geometry.
All invocations reach barriers, including partial workgroups. The canonical
shader digests and the source conversion recipe's references are regenerated.
The retained external artifact is not modified.

The same arithmetic-cancellation fixture fails with the original shaders and passes with
the repaired shaders on AMD. Its input makes separate-product arithmetic
produce zero where contraction produces a nonzero result; alternating columns
also require nonzero outputs. Tests cover odd widths 65/257, token and stream
boundaries, immediate execution and recorded execution. Shader and test hashes
are in the [before](hc-rounding-before-amd.log) and
[after](hc-rounding-after-amd.log) logs.

Physical Metal Node tests and Chrome 157/Metal pass too. The existing operator
and decoder-composition checks pass. Independent source traces confirm that
at token 2003, the repaired layer-0 and layer-5 branch outputs exactly match the source mean
on the GPU's own captured inputs. This does not imply equal inputs or model
outputs. See [layer 0](rounding-prefix-mixing-layer0.json),
[layer 5](rounding-prefix-mixing-layer5.json), and the browser/Metal logs.

## What the trace establishes

At token 2003, layer-0 HC input and norm are bit-exact against the independent
source. Its down projection is the first differing observed stage: maximum
absolute difference 0.000213623. The layer-5 router input differs by
0.00000691414 with the rounding fix alone. These are observed slices, not full
tensor claims. The source/GPU expert sets agree in layers 0–4; the first
remaining mismatch is layer 5, token 2003, experts 466/510. See
[stages](rounding-prefix-stage-comparison.json.gz) and
[actual GPU selection replay](rounding-route-boundary.json).

A physical projection control supplies the exact independent CPU router input.
The existing F32 router selects expert 510, as the source does. The compensated
router matches every checked nearest-F32 dot, yet rounds both boundary logits
to -5.250730037689209, allowing lower-ID expert 466 to win. The subgroup router
also selects 510 on this input. Greater exact-dot accuracy alone is therefore
not a guarantee of the source's F32 expert choice. See
[exact-source-input component](exact-source-input-router5-component.json.gz).

An independent CPU-only control changes only router batch geometry to one row.
That changes the source's layer-5 choice and its later logits: maximum
CPU-to-CPU difference 0.531882, 44,475 policy violations. This diagnoses
F32 routing sensitivity; it does not replace the established reference or
waive the GPU failure. See [geometry](cpu-router-geometry-layer5.json),
[control](cpu-batch1-control.json), and [comparison](cpu-geometry-comparison.json).

## Gate results and remaining work

The comparator remains `compare-learned-reference.py`, policy
`doppler.boundary-tolerance/quantized-v1`, absolute 0.05 plus relative 0.05.
All rows below compare all 248,320 last-position logits for the frozen 2,304-token
input against the same independent artifact/F16-KV reference. All first tokens
agree; all these numerical comparisons fail.

| Explicit candidate | Maximum error | Outside policy |
| --- | ---: | ---: |
| Baseline retained on November 5 | 0.774934 | 64,993 |
| Product rounding repair only | 0.601663 | 43,659 |
| Repair + subgroup HC projections + compensated router | 0.649533 | 43,560 |
| Repair + subgroup HC projections, existing router | 0.482180 | 30,619 |
| Repair + subgroup HC and router projections | 0.463951 | 36,656 |

Projection candidates are identified offline overlays, not recipe changes.
Their Q4K bindings, weights, F32 activations/math and F16 KV remain unchanged.
The more accurate existing HC kernels reduce component error on exact learned
inputs, but the complete model must still pass the original gate. The retained
run JSON and compressed observations expose every candidate and its source
graph; none is selected by normal runtime defaults.

## Other evidence and harness repairs

- Six long-input baseline lifecycle cases pass: queued abort, decode rollback,
  reset/reuse, paused abort, unload of a paused stream, and reload/reuse. This
  receipt predates the rounding repair; it is not a new post-repair lifecycle
  claim. See [receipt](baseline-long-lifecycle.json.gz).
- QSA receipt hashing previously truncated each U32 index to its low byte.
  Hashing now preserves the raw bytes, offsets and all index bits; new receipts
  retain indices with `raw-u32-bytes/v2`. Old receipts remain unchanged and the
  legacy hash fails the receipt contract. On a new physical capture, all 84
  checked score/selection rows pass their independent same-input F32 bounds.
  These are operator checks, not model qualification. See
  [legacy check](qsa-legacy-same-input-check.json.gz) and
  [corrected check](qsa-corrected-same-input-check.json.gz).
- Component ranking assertions now rank nearest-F32 reference dots, because
  distinct F64 dots can round to identical stored F32 logits. The nearest-F32
  equality check remains strict. The model comparator and tolerance policy
  have not changed.

## Identities and reproduction

Artifact: `qwen-3-8-flash-next-text-q4k-ehf16-af32`, manifest SHA-256
`4902a0b730a9310903b00bc0f8f587528fa1af9f37a722d2f9287bafceceec09`.
Source checkpoint: `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Independent reference SHA-256:
`5c22cd2b9e47a92c2988b0e461cad88e444d51919f5f0c7ab9c8cea20e377fd3`.
Frozen inputs SHA-256:
`06a819d529593b58abd6cb27566bb835a9b91aac13a77c0c7479b7e4e7b0c378`.
Unchanged policy SHA-256:
`f1c4629d593ec2517e3628ac68131dc7568c6aba5e0f03db627d26228c720413`.

Large tensor payloads and pinned source/control prefixes remain under the
external models volume's `qwen38-flash-next-de4b8e4d/qualify` directory, at the
paths and hashes recorded in the receipts. The model remains single-sequence
and unpadded. These tests do not qualify another backend, platform or serving
deployment.

The provisioned tests use `DOPPLER_FLASH_NEXT_PROBE`,
`DOPPLER_FLASH_NEXT_UPSTREAM_TRACE`, `DOPPLER_FLASH_NEXT_ROUTER_REPLAY`, and
`DOPPLER_FLASH_NEXT_ROUTER_PROBE` with their retained run JSON. Diagnostic
source instrumentation adds observations and intentionally stops before a
named MoE; it never replaces arithmetic or feeds captured data back.
Each Python/JS report script documents its arguments. The CPU traces use the
pinned unchanged source operators, independent weight decoding and frozen CPU
prefixes. `prepare-hc-candidate.js` asserts that only named F32 projections and
the two repaired shader digests differ from the artifact graph.

The last candidate's actual GPU expert sets match the independent source at
all 2,304 positions in every layer from 0 through 14. Its first mismatch is
layer 15, position 2135 (experts 115/147). This is a useful isolation boundary,
not a passing model result. Its QSA selection matches all 2,048 source tokens;
the source cutoff gap is 0.163565 versus maximum score drift 0.0000915527.
The resumed CPU suffix reproduces the original layer-14/15 router logits
bit-for-bit. See [QSA query](subgroup-layer15-qsa-query.json).

At this query, the attention HC input has maximum error 0.00000118371. After
attention/injection, the stream error is 0.00000786036; the MLP HC norm error
is 0.000943184 and its branch/router-input error is 0.000281334. These are
observed slices, not a demonstration of an incorrect normalization operator.
Attention/output-projection and HC input propagation remain the next numerical
boundary to isolate. See [stage trace](subgroup-layer15-stages.json).
The candidate is not promoted and the model gate remains failing.

Focused validation is retained in `focused-regressions.log`,
`kernel-checks.log`, and the physical GPU/browser logs. Dependency freshness,
source architecture, unused locals, package closure, JavaScript syntax and
Python compilation checks also pass. These checks do not establish full-model
acceptance or performance.

Component: `doppler.runtime-source.gpu.kernels`,
`doppler.runtime-source.config`, `doppler.tests`, `doppler.docs`.
Intent: preserved. Boundary effects: explicit product rounding and regenerated
shader/recipe identity; test-only observations and receipt corrections.
Normal long-context qualification, decoding under a passing configuration,
post-repair lifecycle qualification and performance measurements remain open.

Large JSON receipts and expanded run configurations are gzip-compressed without
changing their original bytes. Decompress a run JSON before supplying its path
to a provisioned test; the external models volume retains the original files.

## Scale-aware follow-up and identical-input controls

The layer-15 normalization jump is predominantly incoming error being rescaled.
The updated stage comparison tool records reference RMS, absolute errors,
RMS-normalized errors and individual stream measurements at every observed
stage. Zero reference scale is explicit, with undefined nonzero/zero ratios
represented as `null`, not a hidden epsilon. Four regression checks cover
scale invariance, unequal stream magnitudes, zero scale and invalid data.

The [plot](relative-prefix.png), [SVG](relative-prefix.svg) and
[CSV](relative-prefix.csv) cover token 2135 in all layers 0–15 under the same
identified subgroup-projection candidate. Each GPU router-input slice is
bit-exact to its earlier normal full-model observation; each CPU slice is
bit-exact to its independent frozen source prefix. See
[bindings](relative-prefix-binding.json) and the
[relative-stage receipt](relative-prefix-stage-comparison.json.gz).
These are single-token stage observations, not full-tensor parity.

At layer 15, the worst of four per-stream ratios is:

| Stage | Maximum error / source RMS | RMS error / source RMS |
| --- | ---: | ---: |
| Attention HC input | 1.124e-4 | 1.541e-5 |
| Attention HC norm | 1.629e-4 | 1.579e-5 |
| MLP HC input | 5.941e-4 | 8.956e-5 |
| MLP HC norm | 4.473e-4 | 8.798e-5 |

The unchanged source norm, including epsilon and learned weights, was replayed
on the exact GPU inputs at the original batch geometry. Replaying its original
CPU input reproduces the frozen output bit-for-bit. Local GPU/source norm
differences are at most 1.91e-6 (attention) and 3.81e-6 (MLP), versus total
differences 1.68e-4 and 9.43e-4. This supports the scale diagnosis at these
specific slices; it does not prove that every norm or earlier operator is exact.
See the [same-input norm replay](relative-layer15-norm-same-input.json.gz).

The earlier layer-0 down-projection difference at token 2003 is 2.136e-4
against source output RMS 24.8316: a maximum-error/RMS ratio of 8.60e-6.
For the subgroup candidate at token 2135, that ratio is 1.68e-6. Those are
different inputs and kernel paths, not a controlled speed or accuracy ratio.

The plot also shows growth during layer-15 attention. Identical-input controls
localize it further:

- The unchanged source attention on actual GPU query and stored F16 KV bytes
  differs from the GPU core by at most 5.84e-6 (RMS 2.55e-7), versus a total
  source/GPU difference of 4.14e-4 (RMS 1.81e-5). Its positive control exactly
  reproduces the original source output row. Source-only substitutions of GPU
  query, key or value inputs show that the value input dominates this row.
  These substitutions are CPU operator diagnostics, never model qualification
  or tensor feedback. See [attention control](attention-same-input-layer15.json).
- All 1,179,648 stored V values match CPU F16 rounding of the GPU's actual
  F32 projection outputs bit-for-bit. The V projection itself differs from the
  same-input CPU projection by at most 2.86e-6 (RMS 8.91e-8), versus total
  difference 1.69e-4 (RMS 4.68e-6). The source projection control reproduces
  every original F32 V value. Earlier input drift crosses F16 rounding
  boundaries; the cache writer is not introducing a separate encoding error.
  See [value control](value-projection-control.json).
- At layer 0, the full source linear-attention branch was replayed on the
  exact GPU input. The original CPU replay reproduces every source output;
  the checked GPU output row differs from the same-input source branch by
  at most 4.02e-7 (RMS 6.11e-8). This is a localized numerical difference,
  not evidence of an out-of-contract operator. See
  [linear branch control](linear-layer0-same-input.json).

Absolute jumps or a smooth relative trend alone cannot distinguish an operator
defect from accumulated rounding and narrow-cache sensitivity. These controls
do not establish irreducibility, and do not authorize a routing-aware gate.
The next unresolved numerical boundary is the small local difference inside
the layer-0 linear branch (projections, recurrent core and output). No further
runtime defect was demonstrated in this follow-up, so no runtime math, dtype,
artifact, kernel path default or tolerance was changed. The full model remains
unqualified under the same failing gate.

The new trace helper can capture complete branch inputs, raw attention outputs
and actual KV bytes before cancellation rolls state back. One initial probe
requested `post_input_norm`, which this HC path does not emit; its failed
receipt is retained. The replacement reads the actual branch input directly.
Successful diagnostic captures bind bit-for-bit to the normal candidate's
router input and preserve normal computation.

Component: `doppler.tests`, `doppler.docs`. Intent: preserved.
Acceptance evidence: the physical readback/control receipts, four scale-metric
regressions, JavaScript syntax and Python compilation checks, and rendered plot
inspection. Boundary effects: test-only observations and descriptive metrics;
production execution and the model acceptance policy are unchanged.

## Layer-0 linear branch decomposition

The staged capture preserves its complete branch input, checked branch-output
row and router-input row bit-for-bit against the previously bound normal
candidate. No kernel path or dtype changes were made. The source module's
original full-sequence replay is bit-exact before the CPU-only controls.

On identical captured inputs at token 2135:

| Boundary | Maximum difference | RMS difference |
| --- | ---: | ---: |
| QKV projection | 9.54e-6 | 4.91e-7 |
| Z projection | 3.34e-6 | 4.53e-7 |
| A projection | 4.05e-6 | 1.31e-6 |
| B projection | 5.72e-6 | 2.07e-6 |
| Recurrent/gated core after supplying exact GPU projection outputs to CPU | 6.85e-7 | 3.76e-8 |
| Output projection on exact GPU core | 3.87e-7 | 3.67e-8 |

These local differences do not demonstrate a violated operator contract. The
CPU projection substitutions are diagnostic controls and never feed tensors
back into GPU execution or replace the model reference. No full-model gate
rerun or readiness change follows from this decomposition. See
[stage controls](linear-layer0-stage-controls.json) and
[normal-run binding](linear-layer0-stage-binding.json).
