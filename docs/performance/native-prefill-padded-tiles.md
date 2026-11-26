# Native packed prefill rectangular grid guard

Component: `doppler.runtime-source.gpu.kernels`
Intent: preserved. Boundary effects: generated matmul shader identity and digest.

`q40Dispatch` flattens logical tile counts into a device-bounded rectangle. With
axis limit 65535, M=2287, N=8192 and tile 16 require 73216 workgroups, but the
rectangle contains 131070. The old prefill shader ran its K loop, packed-weight
loads and barriers in 57854 padded groups, even though it discarded their outputs. The
GEMV entry already had an equivalent uniform row guard.

The prefill entry now returns when its workgroup-uniform tile index exceeds
the declared M/N tile count, before shared-memory work and barriers. This applies
to every format using this shader: Q4_0, Q4_1, Q5_0, Q8_0 and Q6_K. Arithmetic
for valid tiles, accumulation, storage and dispatch planning remain unchanged.
The canonical generated-patch owner and generated shader/digest are updated;
the common dequantization include is unchanged.

CPU geometry checks cover rectangular padding and complete valid-tile coverage.
On Apple M3 Metal Node 22.20 / webgpu 0.4.0, ten immediate/recorded cases across all
five formats with a forced 2D rectangle pass numerical and output-canary checks.
A bounded K=32 case with the actual M=2287, N=8192 shape and 131070-group rectangle
also passes. The full K=2048 model workload was not replayed. Numerical parity
alone does not measure avoided work: the skip follows from the uniform source
guard; there is no claimed kernel speedup or measured DRAM saving here.

This source-proven waste is a plausible contributor to the native LONG failure,
not an established watchdog cause. No last-submitted kernel receipt exists for
that failed request. Full-model numerical/quality and performance qualification
on amd-host Chrome remain pending. The independent 256-token chunk correction in
[native prefill token budget](native-prefill-token-budget.md) must remain
separately attributable.
