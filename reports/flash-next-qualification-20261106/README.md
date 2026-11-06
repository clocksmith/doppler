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
