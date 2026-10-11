# F32 tiled projection accumulation repair

The executed F16-weight/F32-activation prefill kernel now sums each 16-term K
tile before adding its subtotal to the output accumulator. This shortens a
1,024-term dependency chain to 16-term local sums and 64-term outer sums. It
keeps F32 arithmetic, operand formats, layouts, selection, and dispatch geometry.
There are 16 outer accumulators and 16 temporary subtotals per thread; workgroup
memory does not grow.

Physical AMD/RDNA3 execution of all 5,888 retained first-layer A/B projection
outputs confirms the mechanism:

| Projection | Previous maximum local error | Repaired maximum local error |
| --- | ---: | ---: |
| Recurrent time step A | 0.00002365203 | 0.000002133585 |
| Recurrent gate B | 0.000006454761 | 0.000001143483 |

The original kernel exactly reproduces the retained GPU outputs. Independent
F64 dot products use the actual captured F32 operands and exact deployed F16
weights. Eight synthetic GPU shape controls also pass: scalar and ragged shapes,
K tails, multiple workgroup tiles, negative/scaled outputs, long reductions,
large finite values, and small finite values. The tiled variant supports the
declared transposed row-major weight layout; its selection is unchanged.

## Complete-model consequence

A separate diagnostic archive changes exactly
`package/src/gpu/kernels/matmul_f16w_f32a_tiled.wgsl`. The qualified standard archive
remains SHA-256 `9280ead598b098f9f3ad49574a001e88c3ad4b1a433d58349e79ee0e9c01fccc`.
Weights, manifest, generation settings, and production precision remain unchanged.
The replay records the selected candidate shader SHA-256
`31b2f8d9623d77df223f12da61eb5a178209fa1ce43199fe94357d7791c2b3b5`, entry `main`,
and 36 selections. Reading a manifest filename alone would not prove this.

Against the previously frozen, independently evolved F64/F16-cache control:

| Selected prefix | Previous maximum error | Repaired maximum error | 0.001 |
| --- | ---: | ---: | --- |
| 0 | 0.000702658 | 0.000607192 | pass |
| 1 | 0.000940080 | 0.000889875 | pass |
| 2 | 0.000979554 | 0.000792388 | pass |
| 3 | 0.001528940 | 0.001206352 | **fail** |

All four selected tokens remain unchanged. Out-of-tolerance logits at prefix 3
fall from 148 to 11. This is a verified improvement, **not completed numerical
qualification**. Historical fixture comparisons also remain failed. No reference
was activated, no acceptance tolerance changed, and no 55-prefix campaign or
Mac replay was performed. Source-only generated kernel digests are synchronized;
existing local manifests, release archives, and browser delivery are untouched.

## Preserved failed interventions

1. A naive Kahan compensation expression compiled to exactly the original AMD
   outputs. `rejected-kahan.wgsl` and its failing physical receipt remain retained.
2. A pairwise change to the base matmul variant improved isolated operands but
   left complete-model logits identical. The model actually selects the
   register-tiled variant. The incorrect-owner intervention remains documented
   in `base-pairwise-not-executed.wgsl` and its model replay; the base production
   file was restored.
3. The first mixed-weight offline audit incorrectly applied a four-lane Q4
   simulation to the F16 gate projections. Those scope-incomplete results remain
   retained. Correct single-lane FMA simulation exactly matches their GPU output.
4. An odd-sized F16 synthetic weight upload exposed missing four-byte upload
   padding in the test harness. The failure is retained; subsequent controls
   upload padded bytes while retaining the declared tensor geometry.

Measured wall batches are retained but unisolated. Other GPU processes existed,
so these measurements establish neither a speedup nor a performance regression.

## Reproduction

The compressed actual operands and diagnostic archive are retained here. Unpack
`operands.json.gz` to an explicit local path, then run:

```sh
node tests/integration/f16-projection-accumulation-replay.js \
  /tmp/f16-operands.json \
  artifacts/accumulation-repair-20261011/baseline-tiled.wgsl \
  src/gpu/kernels/matmul_f16w_f32a_tiled.wgsl \
  /tmp/f16-projection-physical.json tiled

node tests/integration/frozen-reference-producer-replay.js \
  artifacts/accumulation-repair-20261011/tiled-candidate-diagnostic.tgz \
  models/local/qwen-3-5-0-8b-q4k-ehaf16 \
  /tmp/f16-prefixes.json current /tmp/f16-prefix-target.json
```

The prefix target is `{"index":2,"step":3,"prefixesOnly":true}`. This validated
option executes only the four uninstrumented prefixes, without another cache
capture. Full raw vectors, failed runs, bindings, and compact comparisons are
retained with hashes in [review.json](review.json).

Validation: three synthetic accumulation controls, frozen-prefix scope controls,
actual-operand and eight general physical shape controls, matmul dispatch and
configuration immutability tests, kernel package digests, uniform layouts,
registry reachability, Python/JavaScript syntax, and diff checks.

Component: `doppler.runtime-source.gpu.kernels`, `doppler.runtime-source.config`,
`doppler.tests`, `doppler.repository-tooling`.
Intent: preserved. Boundary effects: kernel arithmetic and generated digest
identities; no public interface, dtype, model semantics, or coordination change.
