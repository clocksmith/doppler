# Registry-owned kernel interfaces

The registry now generates required parameter writers, checks their interfaces
against WGSL declarations, and supplies named bindings for sampling, matmul, and
RMS statistics. Matmul dispatch geometry is explicit metadata. The generic
wrapper validates resources before allocating uniforms. This changes no WGSL
source and does not rewrite historical signed model artifacts.

Component: `doppler.runtime-source.gpu.kernels`, `doppler.runtime-source.config`,
`doppler.repository-tooling`, `doppler.tests`.
Intent: preserved.
Boundary effects: registry schema, generated uniform writers, selected wrapper
bindings and dispatch geometry, development checks, package inventory and docs.
Reploid is an unchanged installed consumer.

## Exact installed bytes

The locally built 0.6.2 candidate has SHA-256
`852b858a1d1f5d42a951f1bed747b8e45d63c3bfad95f4e8c2e186a6d1affbd2`.
Two npm pack invocations reproduced it exactly under Node 22.22.1/npm 9.2.0:
1,796 files, 2,130,916 compressed bytes, 11,133,861 unpacked bytes.
See `archive-reproduction.json`, `standalone-npm-pack.json`, and
`standalone-receipt.json`. The archive is distinct from published 0.6.1 and
previous 0.6.2 candidates; this record does not claim publication or deployment.

Both physical consumer receipts name these exact Doppler bytes. Reploid's
unchanged archive has SHA-256
`b92d6ac8a88740aebe82745d7b2e67afd9eb6c8f676efc9bed675f1cad35a250`
from revision `19bc359ef1b7606fde3132ab2f37920f4891e6e5`.
Fixture receipts name base revision `5a176f9c` and its dirty state, not a pretend
commit for the new code. `source-files.json` binds the changed files by hash.

## Acceptance evidence

- Structural: 343 uniform-bearing variants across 284 shader modules, sharing
  124 layouts; seven variants have no uniform. All 2,894 declared fields receive
  distinctive values and independent reflected-byte assertions. Required,
  invalid, padding, immutable-layout and buffer-reuse cases pass. The deliberately
  changed-layout fixture updates generated code and rejects the old wrapper.
- Physical interface: all 124 original WGSL uniform structures return 976 fields
  per request, twice with different values and reused GPU buffers. The corrected
  vec3 alignment documentation example also compiles and returns correct fields.
- Physical operators: 17 wrapper/reference cases cover normalization, scale,
  paired normalization, split QKV and f32/f16-weight matmul; 540 existing physical
  sampling cases pass. Hardware: AMD Radeon 8060S, RDNA 3, Mesa 26.0.3.
- Installed models: Chrome 146.0.7680.177 on the physical AMD adapter passes
  standalone Qwen generation, embeddings and reranking; installed Reploid's
  public Doppler provider passes generation and display/completion agreement.
  Generation matches 16 frozen reference token IDs. Both consumers report no
  cleanup errors. These checks do not qualify other hardware or longer outputs.
- `compare-model-outputs.js` compares the standalone receipt to the retained
  accepted run with identical descriptors, requests, browser and GPU class.
  Generation output is exact; all 2,048 embedding values and the reranker numeric
  evidence are exact (zero error). Existing f32 tolerances were selected before
  comparison. This is regression evidence, not new source-model or reranking
  task-quality evidence.
- Installed contract checks pass package exports, browser types and injected
  execution, plus Reploid v1/v2 streaming, adapters, cancellation and ownership.
  Injected inference is separate from the physical checks above.
- 39 distinct focused GPU/config/tooling files pass across the final runs;
  176 inference files passed before the final binding/immutability refinements,
  which have focused and physical coverage. This is not a fresh full-suite pass.
  Kernel generation, source architecture/style, browser import, config,
  declarations, public boundary and runtime closure checks pass.

The initial focused failures (missing wrapper import and outdated geometry
fixtures), stale package inventory and stale current closure report are retained
alongside passing reruns. The initial package log refers to an intermediate
archive; only the explicitly identified final archive has model acceptance.

## Reproduction

Run from the repository root with a writable temporary directory. This machine
used `TMPDIR=/dev/shm/doppler-runtime-coherence` because `/tmp` exhausted quota.

```sh
npm run kernels:uniforms:check
node tests/tooling/kernel-uniform-generation.test.js
node tests/gpu/kernel-uniform-writers.test.js
node tests/gpu/kernel-binding-contract.test.js
node tests/gpu/matmul-dispatch-contract.test.js
node tests/gpu/unified-wrapper-cleanup.test.js
node tests/kernels/kernel-uniform-echo-physical.test.js
node tests/kernels/registry-wrapper-physical.test.js
node tests/kernels/sampling-distribution-physical.test.js
node tools/check-packed-package.js --retain /dev/shm/doppler-runtime-coherence/uniform-candidate-final
node tools/check-installed-capabilities.js artifacts/runtime-coherence/kernel-interfaces/physical-standalone-config.json
node tools/check-installed-capabilities.js artifacts/runtime-coherence/kernel-interfaces/physical-reploid-config.json
node artifacts/runtime-coherence/kernel-interfaces/compare-model-outputs.js
```

The physical configurations retain local model locations, public trust keys,
accepted TargetPlans, requests, browser arguments and expected token IDs.
Reproduction on another host requires provisioning those exact model artifacts,
the recorded Reploid archive and an appropriate physical WebGPU implementation.

## Remaining scope

Writers exist for all current uniform layouts, but other manual wrappers still
need migration. Whole-module binding differences can include unused resources;
the initial inventory is not a count of broken active pipelines. Structural
checks parse declarations, not shader numerical semantics. GPU sampling on the
public Capsule path, broader subgroup portability, ambient execution state and
strict JavaScript implementation checking remain separate work.
