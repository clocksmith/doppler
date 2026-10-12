# Paired final-logits controls

Component: `doppler.runtime-source.inference.pipelines.text`
Intent: preserved. The approved explicit paired-intermediate representation must
reach final normalization and projection without being reinterpreted as scalar
F32. Final public logits remain scalar F32.

The active plan now supplies precision, component count, and byte stride to the
logits paths. Last-token copying uses the declared element size. Immediate and
recorded normalization retain weight encoding and paired activation metadata;
the LM head explicitly requests scalar output. Unsupported scalar-only selected
logits and fused greedy selection reject paired inputs. Representation-aware
probes, kernel traces, and explicit
debug readback now decode logical values as F64 observations and retain the raw
F32 component data plus its digest for full captures. These observations never
feed values back into model execution. Scalar TSIR fixture output rejects this
representation explicitly.

`representation-before.txt` retains the original missing-plan-field failure.
`finalization-before.txt` retains a physical invalid-bias failure that left four
pool-owned buffers instead of the three borrowed inputs. Finalization now
releases or records its consumed input even when validation/dispatch throws.

Acceptance evidence:

- `node tests/inference/logits-paired-contract.test.js`: plan propagation,
  logical shape, malformed representation, capacity validation, and scalar
  compatibility.
- `node tests/inference/logits-paired-physical.test.js`: nine Mac Metal/Dawn
  controls. All rows, last-row copying, immediate/direct/recorded output and
  failure ownership. `mac-controls-with-observation.txt` adds a physical paired snapshot and retains exact operands, independent
  Number/F64 equations rounded to F32, outputs, tolerance and provider receipt.
- `mac-scalar-ownership.txt`: existing physical scalar ownership regression.
- Focused last-position, softcap, final-norm weight, utility cleanup,
  selected-logits diagnostic and output-transform tests also pass.

`source-identities.json` binds tested working-tree kernel and logits files. These
controls use two rows of 128 hidden elements and four vocabulary entries. They
do not establish whole-model accuracy, AMD logits acceptance, signed opening,
distributed execution, memory-ceiling acceptance, or a qualified archive. The
whole-model gate remains the approved F64 reference at 0.001 over all 55 prefixes
on both required GPUs.

Boundary effects: consumes the explicit paired Tensor/kernel contracts; no
Reploid runtime pin, model bytes, F16 cache policy, or deployment changes.

Observation regressions also exercise immediate and recorded capture, malformed
metadata, undersized logical buffers, failed mapping cleanup, and cancellation
before submission. Recorded diagnostic staging is now recorder-owned so abort
and failed submission release it without running a completion callback.
`tests/inference/probes-readback.test.js` and
`tests/integration/debug-tensor-contract.test.js` pass with synthetic devices.
Eleven focused suites and source type/style/architecture/component checks pass.
No new whole-model or release acceptance is claimed.
