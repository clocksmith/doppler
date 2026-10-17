# Cooperative packed W8A8 operator tuning — 2026-10-16

Component: `doppler.runtime-source.gpu.kernels`
Intent: preserved.
Boundary effects: additive registered matmul variant and W8A8 wrapper admission;
no model conversion, artifact promotion, public service, or default-path change.

## Result and scope

`matmul/w8a8_packed_cooperative` reduces each output across 16 lanes in a
64-lane workgroup. Adjacent lanes read adjacent packed words. Integer dot
products use `dot4I8Packed`; a packed byte check preserves rejection of -128,
and a last-word mask preserves zero-padding validation. The reduction retains
zero-row scale validation and sticky GPU error reporting. The original
`w8a8_packed` and `w8a8_portable` remain unchanged.

The independent rational-reference checks pass on physical Apple Metal and
AMD RDNA3 browser WebGPU. Complete synthetic projection batches improve on
Mac; AMD gains depend on shape and submission overhead. These are operator
tuning results, not a model qualification or an application benchmark claim.
No assertion that W8A8 beats F16, Q4K, llama.cpp, or Transformers.js follows.

## Frozen comparison

- Fixture: [w8a8-operator-profile.json](../../tests/fixtures/w8a8-operator-profile.json).
- Identical seeded inputs and packed weights within each alternating pair.
- Three warmup pairs, 12 measured pairs, eight identical projections per
  submission, one mandatory recorder-level status validation. Output bits
  compared after every run, outside the timing interval.
- Complete wall time includes recording, allocation from warmed pools,
  quantization, packed dot product, two exact rescale passes, submission,
  completion and error validation. It excludes model loading and output-copy
  observation. Reused weights/inputs make this a warm-cache microbenchmark.
- Three separately timestamp-instrumented runs per variant/shape; they do not
  enter the primary wall-time samples. All raw samples and stalls are retained.
- Shapes are synthetic. The 5,120-wide reductions cover a larger reduction
  width; their 1,024 output columns are not a full Qwen model qualification.
- Mac: Chrome 155.0.8059.80 / Apple Metal. AMD: Chrome 146.0.7680.177 / Vulkan,
  AMD RDNA3. Do not compare absolute speeds between these environments.
- Public contributors remained running. This was not an isolated-device
  throughput campaign. AMD ran in a separate source snapshot under an 8 GiB
  process-tree memory limit, no swap, and a 300-second runtime bound. Neither
  machine loaded model weights for this test.

Final median complete-batch time divided by eight, in milliseconds per
projection (not single-request latency):

| Device | M × N × K | Original packed | Cooperative | Ratio |
|---|---:|---:|---:|---:|
| Mac | 1 × 640 × 2048 | 0.2594 | 0.1584 | 1.64× |
| Mac | 16 × 640 × 2048 | 0.4397 | 0.3644 | 1.21× |
| Mac | 128 × 640 × 2048 | 2.2519 | 1.7872 | 1.26× |
| Mac | 1 × 1024 × 5120 | 0.5869 | 0.2153 | 2.73× |
| Mac | 16 × 1024 × 5120 | 1.2987 | 0.9091 | 1.43× |
| AMD | 1 × 640 × 2048 | 0.6494 | 0.6581 | 0.99× |
| AMD | 16 × 640 × 2048 | 0.6200 | 0.6437 | 0.96× |
| AMD | 128 × 640 × 2048 | 0.7444 | 0.7531 | 0.99× |
| AMD | 1 × 1024 × 5120 | 0.9141 | 0.6472 | 1.41× |
| AMD | 16 × 1024 × 5120 | 0.9019 | 0.7016 | 1.29× |

The timestamped dot stage for `1 × 1024 × 5120` improves from 0.4479 to
0.0654 ms on Mac and 0.3767 to 0.0356 ms on AMD. This larger kernel-only
improvement is not the complete projection speedup. AMD's smaller-shape
wall-time improvement in the first run did not reproduce; those shapes are
not claimed as wins. See [amd-initial.json](amd-initial.json).

## Correctness and ownership

Both [mac-final.json](mac-final.json) and [amd-final.json](amd-final.json)
pass 103 checks covering all three variants:

- Eight shapes, including partial columns, K tails, zero rows, rounding ties,
  and the maximum exact I32 reduction width of 133,144.
- Independent exact activation bytes, every I32 accumulator, and final F32
  bits in immediate and recorded execution.
- 7,168 byte-domain/tail cases per variant: all 256 byte values in each packed
  position over seven reduction widths. Invalid words must set status and
  contribute no dot product.
- 1,032 rescale cases, including 126 overflow cases and signed-zero flushing.
- Nonfinite/subnormal activation, scale underflow, -128, dirty padding,
  invalid scales, a zero row with non-unit scale, and output overflow.
- Existing shared-status, partial-submit, cancellation, abandonment and fresh
  reuse controls; no remaining active or deferred GPU buffers.

The performance harness also verifies exact agreement between the two variants
on every warmup, timed run and profiled run. Final pooled reusable allocation
is 12,456,448 bytes on both machines, with zero active/deferred bytes; the pool
and device are destroyed afterward. This is pool telemetry, not total process
or driver memory.

## Reproduction and provenance

From the Doppler repository, with Playwright and Chrome installed:

```sh
node tests/integration/w8a8-physical.test.js /tmp/w8a8-result.json tests/fixtures/w8a8-operator-profile.json
```

The runner executes a copied source snapshot and records hashes for every
served source file, browser/system identity, fixture bytes/config and launch
flags. The Mac harness selects Metal by leaving the backend unforced; the
Linux harness explicitly requests Vulkan. A failed correctness gate prevents
profiling. Failed profiles retain partial results and fail the command.

[amd-copied-source.json](amd-copied-source.json) identifies the Mac working tree
copied into AMD's isolated directory. The `commit` in the AMD receipt describes
its surrounding, older checkout, not the copied source. Served-file hashes
bind the actual executed bytes. Final shader digest:

`dc527069dc264bafc95cc1f971a79e755122564cc7b72289f98621c33424d90d`.

Investigation receipts retained:

- `mac-baseline.json`: original portable/packed comparison, one projection per
  submission; localized the GPU wall to the dot stage.
- `mac-cooperative.json` and `cooperative-v1.wgsl`: cooperative reduction before
  packed validation; some batch shapes were inconclusive or slower.
- `mac-cooperative-packed-checks.json`: packed validation added, still single
  projections. The final fixture batches eight and adds finer wall-clock
  resolution; only paired variants within a receipt are directly comparable.

Acceptance evidence: physical runs above; strict-source and declaration tsc;
unused-local/style gates; kernel registry, uniform generation, and source
kernel/conversion digest checks; focused matmul dispatch, registry loader,
uniform writer/generation, W8A8 reference and loader tests.

The combined `check-source-types.js` wrapper reports an already-stale debt
inventory entry for `src/config/schema/speculation-self.schema.js`. Direct
strict tsc passes; this change does not modify that unrelated inventory.

## Promotion boundary

No automatic routing rule or model graph selects this new variant. It is
available through the explicit W8A8 variant contract and registered kernel
metadata. Before promotion, measure the complete exact W8A8 model with this
variant under the same numerical and task-quality gates. The earlier Gemma
W8A8 model-quality rejection remains in force. Qwen's qualified Q4K bytes are
not interchangeable with W8A8 or GGUF Q8_0; this change does not convert or
qualify Qwen W8A8, add Q8_0 loading, or deploy Reploid.
