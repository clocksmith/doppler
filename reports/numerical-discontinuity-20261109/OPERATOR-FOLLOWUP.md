# First-normalization witness and current Flash-Next router inputs

This extends the frozen diagnostic in `08c12ad6`, without modifying its files.
It incorporates the current F32-KV routing capture used by the layer-25–47
bisection retained in `03521e0d`. **Neither model is newly qualified.**

## Historical 0.8B: the earliest captured normalization difference is explained

The original Qwen 3.5 0.8B plain/proxy captures have identical embeddings.
An offline F32 evaluation of the archived `rmsnorm.wgsl` main entry reproduces
**188,416 of 188,416 physical GPU outputs**, across 184 positions, exactly.
Its maximum difference from an independent high-precision formula is
`5.032750927114193e-7`. That formula and the retained paired proxy differ by at
most `1.9539925233402755e-14`.

The witness reproduces the shader's four values per thread, 256-thread tree
reduction, epsilon addition, inverse-root refinement, and `(1 + weight)` scaling.
The inverse-root estimates are constrained before comparing outputs by WGSL's
[2-ULP accuracy allowance](https://www.w3.org/TR/2026/CRD-WGSL-20261106/#floating-point-accuracy).
Integer inequalities enclose the mathematical inverse root; exact integer FMA
and nearest-even F32 rounding avoid relying on host arithmetic for that step.
Every complete row must match one common permitted estimate. Estimates are
never selected independently for each element.

Eight arithmetic models were declared before analysis. All four models with
left-associated output multiplication explain every row; none of the four
right-associated models explains a complete row. The result does not distinguish
the fused-versus-separate refinement choices. It establishes a fixed admissible
evaluation for this operator, not compiler conformance or correctness of its
downstream consumers. JSON inputs containing zero are rejected because they
cannot preserve negative-zero identity.

Each model rejects a changed output and a wrong weight-offset operation on all
184 rows. These controls detect those faults; they do not demonstrate rejection
of every possible bug. The previous packet's deliberately altered values inside
a loose dot-product envelope remain an explicit warning against using interval
membership as acceptance.

## New physical Metal control

A frozen seed generated 32 new rows of 1,024 F16-valued inputs, over 16 magnitude
scales, before GPU execution. Chrome `157.0.8103.38` ran the archived shader on
Apple M3 / Metal. The same fixed arithmetic models reproduce **32,768 of 32,768
outputs**. Maximum error against the independent formula is `3.2338851019275694e-7`.
All 32 corrupted-output and all 32 wrong-offset controls are rejected. There
were no shader compilation or WebGPU errors.

These are new synthetic **operator inputs**, not held-out prompts or a complete
model run. The existing replay harness calls every fixture “captured”; here its
`matchesCaptured: false` compares the physical result to a rounded-F64 synthetic
baseline. It does not mean this control failed. The original receipt and values
are preserved in [heldout-metal.json.gz](heldout-metal.json.gz); the labeled
[summary](heldout-metal-summary.json) explains that inherited field.

The final analyzer adds fail-closed checks for unsupported contracts, nonfinite
values and ambiguous zero encoding. These are validation changes after the
first run, not changes to the frozen arithmetic models or an output tolerance.
Both datasets were reanalyzed with the final tool and retain identical counts.

## Current Flash-Next: both first routing changes persist with exact dots

The CPU-only analysis on Jupiter verifies the manifest, observation, reference
receipt, NPZ bytes, captured F32 bytes and F16 routing weights before arithmetic.
It evaluates all 512 experts at positions 1717, 2271 and 2303 in layer 25 using
exact integer dyadic sums, reported rounded to F64. This checks **raw-logit
ordering**, not a new physical softmax/top-k execution.

Margins below are source-selected expert minus GPU-selected expert:

| Position / pair | Exact dot on source input | Exact dot on GPU input | Change caused by different input |
| --- | ---: | ---: | ---: |
| 1717: 493 minus 330 | +1.514252473e-6 | −2.701793814e-6 | −4.216046287e-6 |
| 2271: 311 minus 411 | +1.133673765e-6 | −8.463160519e-7 | −1.979989817e-6 |

At both positions, each side's exact-dot top-10 set matches its stored-logit
top-10 set. Position 2303 keeps the same set on both sides and both evaluations.
The 1717 observation independently reproduces the existing November 7 boundary
report; 2271 is the extension. These were known failing positions, not a holdout.

Thus improving router accumulation alone cannot reconcile these two decisions
while preserving their actual inputs. Earlier arithmetic supplies enough input
drift to reverse both choices. **This does not establish that the input drift is
permitted rounding error.** An independent bound or constructive witness for
that upstream computation remains missing. Local dot envelopes admit all sampled
outputs but are deliberately not used as acceptance.

The new bisection's intermediate-head and grafted-state projections localize
propagation. They are counterfactual diagnostics, not full-model qualification
at each intermediate layer. The final 13 violations remain failures.

## Reproduction and integrity

Nine focused arithmetic tests pass. Both normalization and router analyzers
reject an incorrect input SHA before creating output. The original packet and
five comparator/reference files retain their previous hashes. No runtime,
tolerance, reference implementation, production service or deployment changed.
One new GPU invocation ran the 32-row operator fixture; Jupiter analysis was
CPU-only. The existing harness closes its browser and destroys its GPU resources.

Raw normalization evidence, shader, fixture and input bundles are retained under
`.cache/numerical-discontinuity-20261109/` on the analysis Mac and Jupiter. The
extension receipt lists their hashes. These include model-weight slices and are
operator-local dependencies, not silently assumed public data.

From the Doppler checkout, with those retained dependencies:

```sh
node --test tools/numerical-discontinuity-core.test.js tools/dyadic-diagnostic.test.js
node tools/analyze-input-normalization.js \
  reports/numerical-discontinuity-20261109/input-normalization-protocol.json \
  .cache/numerical-discontinuity-20261109/inputs.json.gz \
  .cache/numerical-discontinuity-20261109/input-norm-evidence.json /tmp/norm-result.json
node tools/prepare-normalization-control.js fixture \
  .cache/numerical-discontinuity-20261109/input-norm-evidence.json \
  reports/numerical-discontinuity-20261109/input-normalization-heldout-protocol.json \
  /tmp/norm-fixture.json
node tests/integration/rmsnorm-captured-replay.js /tmp/norm-fixture.json \
  .cache/numerical-discontinuity-20261109/frozen-rmsnorm.wgsl /tmp/norm-metal.json
node tools/prepare-normalization-control.js bundle \
  reports/numerical-discontinuity-20261109/input-normalization-heldout-protocol.json \
  /tmp/norm-fixture.json /tmp/norm-metal.json \
  reports/numerical-discontinuity-20261109/input-normalization-protocol.json \
  /tmp/norm-inputs.json.gz /tmp/norm-analysis.json
node tools/analyze-input-normalization.js /tmp/norm-analysis.json \
  /tmp/norm-inputs.json.gz .cache/numerical-discontinuity-20261109/input-norm-evidence.json \
  /tmp/norm-heldout-result.json
```

The fixture generator checks byte identity against the frozen fixture before
writing. The router tool takes the frozen protocol, artifact directory,
qualification directory, source-trace directory and output path; exact paths
and identities are in `operator-receipt.json`. No model execution is performed
by either analyzer.

Remaining work: justify the upstream Flash-Next input differences and the
0.8B computation from normalization to cache storage; then evaluate unseen
prompts and stronger fault controls. This packet does not propose a relaxed
correctness contract or claim an available speedup.

Component: `doppler.repository-tooling`. Intent: preserved. Acceptance evidence:
frozen operator witnesses, physical Metal control, arithmetic tests and identity
negative controls. Boundary effects: reads historical Reploid and Flash-Next
evidence; no runtime or qualification change.
