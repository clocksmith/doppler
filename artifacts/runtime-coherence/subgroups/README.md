# Subgroup identity and execution requirements

The repaired `rmsnorm_stats_subgroups.wgsl` and
`attention_decode_subgroup.wgsl` use actual subgroup identity and count. Their
registry requirements agree with WGSL sources, flow through Forge into signed
TargetPlans, and reject unsupported devices before program preparation. Old
signed shader sources keep their own requirements and identities.

The RMSNorm stats wrapper also releases owned output allocations on preparation
failures and destroys direct uniforms after submission. Recorded uniforms retain
their recorder owner; borrowed input/output buffers are not destroyed.

The retained Node and Chrome 146 physical AMD RDNA 3 runs each passed 60 RMSNorm
and 54 attention cases. RMSNorm includes the separate portable reduction.
Tests cover dimensions below/above subgroup width and permuted data-index
geometry. They do not control hardware subgroup membership. The original shader
fails the instrumented case at workgroup size 128 and hidden size 17; this is not
a claim of failure under the adapter's ordinary native mapping.

The original wrapper also fails the allocation cleanup test. The repair passes
24 immediate/recorded, owned/borrowed, allocation/compilation/dispatch cases.
Sixteen focused test files plus four subsequent contract checks passed, as did
kernel generation, registry/source requirements, schema generation, source
style/architecture, package closure and runtime closure checks. The existing
declaration-oriented typecheck passed; strict implementation checking remains
separate work.

Standalone and the accepted Reploid library consumed the same installed archive:
SHA-256 `068b2bbfb3b0568bb54eface1fee61809a703a3ec8cf4d49b6ecb8879900d490`.
Those consumer tests use signed fixtures and injected logits. The physical runs
exercise operators directly; neither is new full-model qualification.

Reproduction from the recorded implementation:

```sh
node tests/kernels/subgroup-portability-physical.test.js
node tests/kernels/subgroup-portability-browser.test.js
node tests/gpu/rmsnorm-stats-ownership.test.js
node tools/run-node-tests.js tests/runtime/capsule-wgsl-language.test.js tests/config/kernel-language-contract.test.js tests/gpu/kernel-language-selection.test.js tests/tooling/wgsl-language-export.test.js
npm run kernels:check
TMPDIR=/dev/shm node tools/check-packed-package.js --retain /dev/shm/doppler-subgroup-consumer
```

Run Reploid's `tests/fixtures/doppler-installed-generation.js` at the revision in
`acceptance.json`, with `DOPPLER_TEST_CONSUMER` pointing to that retained
`consumer` directory. Archive reproduction also depends on the recorded npm,
Node and compression versions. The exact tested archive is retained locally;
this record does not claim publication or reproduction with another toolchain.

`acceptance.json` and `standalone-source-state.json` bind the source bytes;
`files.json` hashes this evidence. Failed baseline, browser setup, and schema
generation attempts are preserved. `remaining-subgroup-inventory.json` records
additional mapping patterns for ongoing algorithm-specific repair; this receipt
does not qualify those shaders.

Component: Doppler GPU kernels, Capsule runtime, config, Forge and packaging.
Intent: preserved.
Acceptance evidence: commands and retained logs above.
Boundary effects: WGSL language requirements cross the existing registry,
Forge and TargetPlan boundary. No Reploid source changes, publication or deployment.
