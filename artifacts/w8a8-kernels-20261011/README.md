# W8A8 GPU operator evidence

These are scoped physical operator checks on Linux AMD RDNA3 through Chrome
146.0.7680.177. They do not establish model quality, full-model speed, installed
package acceptance, or another device's support. Each invocation freezes its own
source copy and records served SHA-256 identities and the concurrent working-tree
status. The standard package and accepted model pins remain unchanged.

Commands (run from the Doppler repository):

```sh
node tests/integration/w8a8-physical.test.js artifacts/w8a8-kernels-20261011/initial-physical.json
node tests/integration/w8a8-physical.test.js artifacts/w8a8-kernels-20261011/rescale-boundaries-physical.json
node tests/integration/w8a8-physical.test.js artifacts/w8a8-kernels-20261011/lifecycle-physical.json
```

All three completed successfully. The final receipt adds to the earlier controls:

- Exact GPU activation bytes against an independent BigInt rational quantizer.
- Exact I32 accumulation and final F32 bits for portable and packed-dot variants,
  immediate and recorded execution, including K tails and the 133,144-wide I32 bound.
- Seven matrix shapes, including K=640 and K=2048. These are operator coverage,
  not a full Gemma workload benchmark.
- Integer-to-F32 tie cases and 1,032 multiplication boundary/random pairs, including
  126 overflow cases, signed-zero flushing, minimum-normal round-up and mantissa carry.
- Thirty-two invalid-input checks covering nonfinite/subnormal activations, scale
  underflow, forbidden -128, nonzero padding, invalid scales and output overflow.
- Nine preflight rejections covering dimensions, shape/metadata, missing variant,
  offsets, output aliasing and incompatible specialization constants.
- A single mandatory status readback across multiple projections, including partial
  submissions and deferred completion. An earlier failed projection cannot be erased
  by a later successful projection.
- Unsubmitted abandonment, pre-dispatch cancellation, subsequent valid execution,
  zero active allocations and zero deferred cleanup after settlement.

Only the four-byte status is read to the CPU by runtime execution; activation
packing, dot products, and both exact rescale stages execute in WGSL. Test-only
readbacks verify the declared arithmetic against the independent reference.

## Frozen representative projections

The follow-up invocation uses the unchanged qualification fixture and its seed:

```sh
node tests/integration/w8a8-physical.test.js artifacts/w8a8-kernels-20261011/frozen-projections-physical.json tests/fixtures/w8a8-qualification.json
```

It passed all fifteen combinations of M=1/16/128 and the declared projection
(N,K) pairs (1024,640), (256,640), (640,1024), (2048,640), and (640,2048).
Both variants match every independently quantized activation byte, every I32
accumulator, and every F32 result bit, in immediate and recorded execution. The
same run repeats the boundary/lifecycle controls and adds explicit all-zero rows.
No output sampling substitutes for complete matrices. The receipt binds the
fixture digest and served source identities at committed runtime `a7193ebe`.
The reference uses independent BigInt rational quantization; its integer products
and sums remain exactly representable in JavaScript Number under the fixed I32
bound. Active and deferred allocation counts both finish at zero. This extends
operator correctness coverage; it does not establish model accuracy or speed.
