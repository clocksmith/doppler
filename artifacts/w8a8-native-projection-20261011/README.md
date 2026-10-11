# One W8A8 projection for native provider evaluation

This handoff contains synthetic operands for the existing Gemma query shape
`M=1, N=1024, K=640`, seed `7331`. It does not contain Gemma model weights and
does not establish model accuracy, native Doe execution, or performance.

Download/extract [projection.tar.gz](projection.tar.gz). Its SHA-256 is
`128bbc084593483c2f4e4c532f8b118e2c25b68d7f7f1b0dd3ee73adec8911af`.
The [index](index.json) records all retained checks and binds the exact installed
Doppler archive `7c240dad4d1ae98e204e9727e687b9bc4379abd758b16af0fd857a99d7adac8d`.
All 17 payload hashes were independently rechecked after export. The focused
exporter controls, runtime closure, source style, source architecture and
generated dependency-map checks passed. These are offline checks, not another
physical qualification run.

The shape and operand construction come from
[`w8a8-qualification.json`](../../tests/fixtures/w8a8-qualification.json) and
[`w8a8-physical.js`](../../tests/kernels/w8a8-physical.js). The retained
[physical receipt](../w8a8-kernels-20261011/frozen-projections-physical.json)
verified exact packed bytes, signed accumulators and rescaled output for both
portable and packed-dot variants. It records browser AMD/Vulkan execution;
that evidence does not qualify a different native provider.

The Node-only [exporter](../../tools/export-w8a8-projection.js) reuses the
[independent reference](../../tests/kernels/reference/w8a8.js). It requires
the oracle, operand generator, fixture and packaged shader hashes to match that retained physical
verification. It copies unchanged shader bytes from the installed diagnostic
package and binds the archive digest through its installation receipt. It
generates no expected values from GPU observations.

Reproduce with an unused output directory in an explicit config:

```sh
node tools/export-w8a8-projection.js artifacts/w8a8-native-projection-20261011/export-config.json
node tests/tooling/export-w8a8-projection.test.js
```

The emitted manifest owns every payload digest, little-endian layout, buffer
size, initial state, binding, override constant, dispatch and expected output.
It includes original F32 inputs, packed activation and weight rows, exact I32
accumulators, intermediate F32 rescaling and final F32 output bits. The full
projection starts with F32 activations and executes packing plus dot plus two
rescale passes. Timing a prepacked dot alone would be a different component
scope. Read and require zero status before accepting output; await completion,
read back every declared result and release all owned resources.

## Doe handoff and attribution

At Doe `5fbfd7b8c`, the existing
`bench/executors/run-packed-integer-controls.mjs` exercises ordinary WebGPU on
explicit native Vulkan providers. Its retained packed-integer controls pass on
Doe and Dawn; Dawn's separate normalized-packing failures do not disprove this
integer-dot path. Doe's current dot lowering uses integer multiply/add;
dedicated integer-dot acceleration remains unqualified.

The smallest native follow-up is to load this one fixture through that existing
benchmark ownership, execute the declared stages on both native providers,
and compare all bytes against the independent reference before measuring.
That comparison changes both compiler and execution provider together.

A Tint/Doe compiler by Dawn/Doe executor matrix requires more explicit plumbing:
the current ordinary JavaScript shader-module surface accepts WGSL strings,
not imported SPIR-V. Doe's C ABI accepts `WGPUShaderSourceSPIRV`, but imported
modules initialize binding reflection empty and workgroup metadata to one;
explicit pipeline-layout and dispatch handling need qualification. Native Dawn
SPIR-V admission, its subsequent translation and the exact executed artifact
also need verification. Existing Tint/Doe compilation reports can produce
SPIR-V; they do not themselves establish mixed execution arms. Do not relabel
a WGSL-to-WGSL preprocessing arm as direct Doe-SPIR-V execution through Dawn.

Keep the W8A8-versus-F16/Q4 quantization evidence separate from same-W8A8
provider comparisons. Freeze native timing scope, initialization/reuse,
completion/readback, compiler options and actual provider/driver identities
before any speed claim. No native run accompanies this export.

Component: `doppler.repository-tooling`, `doppler.tests`.
Intent: preserved.
Boundary effects: offline fixture handoff only; no shipped runtime, package,
model, native Doe source or release change.
Acceptance evidence: [test.log](test.log), [closure.log](closure.log),
[style.log](style.log), [architecture.log](architecture.log), [map.log](map.log).
