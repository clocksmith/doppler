# Flash GGUF decode control GEMV (opt-in)

Use `profiles/flash-next-gguf-control-gemv` from a repository checkout with a
native Flash GGUF source manifest. It patches seven **decode-only** floating
projections to the existing `gemv_f32_subgroup` kernel: linear alpha/beta,
MoE router/shared router, attention/MLP hyperconnection injection, and QSA query/key.
It requires device `subgroups` and WGSL `subgroup_id`; it is not a blanket default
for unsupported devices or other models.

Select it with `--runtime-profile profiles/flash-next-gguf-control-gemv` on an
otherwise valid CLI inference request. If your existing run supplies custom
runtime settings, instead merge this profile's `runtime.inference.executionPatch`
into that run's runtime configuration before model loading. Use one runtime input
path per CLI command, not both `--runtime-profile` and `--runtime-config`.

The profile changes no prefill steps, packed-weight kernels, source quantization,
KV precision, cache budgets, routing, or sampling. It does not enable the unrelated
compact-slot or segmented-recording experiments. The post-layer mixer remains
unchanged because that graph section also serves prefill.

## Qualification boundary

On one AMD Jupiter device, controlled same-code ABBA against `d492842d` retained
all 47 generated token IDs in all eight frozen JavaScript-review responses.
Repeated decode: baseline 13.334/13.328 s, candidate 11.715/11.689 s (about 12.2%
less time); repeated total: 24.143/24.158 s versus 22.252/22.472 s (about 7.4% less).
All repeated decode physical reads were zero. Disk-bound fresh decode had a
smaller benefit. These are workload/device-specific measurements, not a general
speed guarantee or end-to-end advantage over peer engines. Broader quality and hardware qualification
remain open.

F32 storage and accumulation remain F32, but subgroup reduction changes rounding
order. The independent FP64 oracle covers N/K = 4/10240, 48/2560, 512/2560,
640/2560, and 1/2560, using absolute-plus-relative tolerance
`1e-6 + 1e-5 * abs(reference)`. Immediate and recorded results match, with observed
subgroup shader provenance. One exact model continuation does not prove universal
bitwise equivalence.

```sh
node tests/config/flash-control-gemv-profile.test.js
DOPPLER_REQUIRE_F32_GEMV_GPU=1 node tests/inference/flash-control-gemv-gpu.test.js
```
