# F32 KV online decode attention

The historical 35B runs discussed here used **F32 activations**. F16 weight
storage and F16 KV storage are separate contracts and do not imply F16
activations. The native attribution run used F32 activations and F32 KV; its
prepared decode workspace was active (700 buffers, 660 bindings and 20,160
recorded hits). The compatibility attention selection therefore cannot be
explained by an F16 activation admission gate or an inactive workspace. Pure CPU
encode time has not been isolated; orchestration timers include awaits.

The execution compiler previously widened `attention_decode_online_f16kv.wgsl`
and its fixed-head-256 variant to `attention_streaming.wgsl`. The latter is a
compatibility baseline with one invocation per head. Selecting it did not mean
that the GPU was unavailable: it changed the attention algorithm's parallelism.
The retained native 35B diagnostic selected that baseline and attributed 27.499 ms
of an 82.360 ms median profiled decode step to ten attention dispatches.

The F32 widening map now selects `attention_decode_online.wgsl`. This registered
variant is generated from the existing online F32-Q/F16-KV source by changing only
KV storage element types and removing the F16 capability requirement. It retains
F32 accumulators, compensated summation, masks, and the existing subgroup/chunk
algorithm. It supports heads through 512 and does not inherit the separate
`decode_subgroup` kernel's 2048-key limit. It does not require the `subgroup_id`
WGSL language feature. Hardware without subgroups still uses the declared
non-subgroup fallback; F32 buffers are never bound to F16-KV shaders.

`tests/integration/attention-f32kv-online-policy.test.js` covers the widened
identity, precision metadata, digest invalidation, source immutability, feature
rejection and non-subgroup fallback. The existing capability and graph-transform
suites remain required. Kernel registry, generated uniforms, content digests,
style and declaration checks cover the added variant.

`tests/inference/attention-f32kv-online-gpu-regression.test.js` compares immediate
and recorded execution with an independent Float64 reference. The first Apple M3
Metal qualification passed 16 cases: 16 query heads / 2 KV heads / head dimension
256 with KV lengths 0, 1, 43, 257 and 2287; sliding-window plus softcap; and head
128/512 boundaries. Maximum absolute error was 1.192e-8 against the unchanged
1e-4 operator gate.

Instrumented component timings used five alternating samples per variant after
one warmup. At KV length 43 the streaming/online medians were 2.818048/0.065536 ms;
at KV length 2287 they were 153.878528/1.245184 ms. Timestamp resolution was
65.536 microseconds, including one zero-valued short sample. These component
measurements do not establish model-level speed, token parity, Radeon performance,
or comparison with another engine. Full-model numerical continuation, quality,
and balanced production-path timing remain separate admission gates.

Component: `doppler.runtime-source.gpu.kernels` and the existing execution-graph
compiler. Intent preserved. No public session field or execution adapter changed.
