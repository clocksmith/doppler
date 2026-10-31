# Native packed prefill rectangular grid guard

Component: `doppler.runtime-source.gpu.kernels`
Intent: preserved. Boundary effects: generated matmul shader identity and digest.

`q40Dispatch` flattens logical tile counts into a device-bounded rectangle. With
axis limit65535, M2287/N8192/tile16 requires73216 workgroups, but the rectangle
contains131070. The old prefill shader ran its K loop, packed-weight loads and
barriers in57854 padded groups, even though it discarded their outputs. The
GEMV entry already had an equivalent uniform row guard.

The prefill entry now returns when its workgroup-uniform tile index exceeds
the declared M/N tile count, before shared-memory work and barriers. This applies
to every format using this shader: Q4_0, Q4_1, Q5_0, Q8_0 and Q6_K. Arithmetic
for valid tiles, accumulation, storage and dispatch planning remain unchanged.
The canonical generated-patch owner and generated shader/digest are updated;
the common dequantization include is unchanged.

CPU geometry checks cover rectangular padding and complete valid-tile coverage.
On Apple M3 Metal Node22.20/webgpu0.4.0, ten immediate/recorded cases across all
five formats with a forced2D rectangle pass numerical and output-canary checks.
A bounded-K32 case with the actual M2287/N8192 shape and131070-group rectangle
also passes. The full K2048 model workload was not replayed. Numerical parity
alone does not measure avoided work: the skip follows from the uniform source
guard; there is no claimed kernel speedup or measured DRAM saving here.

This source-proven waste is a plausible contributor to the native LONG failure,
not an established watchdog cause. No last-submitted kernel receipt exists for
that failed request. Full-model numerical/quality and performance qualification
on amd-host Chrome remain pending. The independent profile256 correction must
remain separately attributable.
