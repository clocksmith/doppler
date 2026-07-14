# WGSL Repair V13 Semantic Contract

V13 freezes the semantic evaluation requirements that follow V12. It does not
modify V12, select one external20 seed, or claim semantic WGSL correctness. The
machine-readable policy is
`tools/policies/wgsl-repair-v13-semantic-policy.json`; the current readiness
receipt is `wgsl-repair-v13-semantic-readiness-2026-07-13.json`.

## Frozen pass definition

Compilation in Chromium WebGPU is a blocking prerequisite. A semantic task
passes only when every declared shape and workgroup variant dispatches, matches
its hash-bound CPU oracle, preserves buffer canaries and read-only bytes,
satisfies two qualified metamorphic relations, and clears historical
regressions.

Numerical agreement uses:

```text
abs(actual - reference) <= absTolerance + relTolerance * abs(reference)
```

The frozen ceilings are exact comparison for integer and Boolean outputs,
`1e-5` absolute plus `1e-4` relative tolerance for float32, and `0.002`
absolute plus `0.02` relative tolerance for float16. Tasks may use narrower
tolerances. Wider tolerances require a new policy revision frozen before
candidate evaluation. Unexpected NaN or infinity fails.

Every task requires three shape classes—nominal, non-workgroup-multiple, and
boundary or tail—and two semantically valid workgroup variants. Prefix and
suffix canaries each contain 16 elements; read-only buffers and output padding
must remain byte-identical.

## Current decision

The readiness gate returns `blocked`. V12 adapters are verified locally but do
not have governed immutable URLs. Semantic populations, task and oracle
manifests, historical regressions, seed-level checkpoint selection, and
trainer-to-Doppler adapter parity are absent. No semantic dispatch receipt
exists.

Accordingly, `semanticEvaluationAllowed`, `semanticClaimAllowed`,
`wgslDoctorAllowed`, and `autonomousShaderAuthorAllowed` are all false. The
first product may be a verified replacement-only repair assistant only after
this contract passes; complete shader authorship remains a separate SAME-R
experiment.

## Required continuation

1. Define a governed immutable destination and publish the three exact V12
   adapter byte streams with revision-pinned or content-addressed URLs.
2. Freeze disjoint semantic calibration, checkpoint-selection,
   seed-confirmation, and one-use promotion populations.
3. Implement and seal CPU oracles, task inputs, tolerance tiers, metamorphic
   relations, buffer layouts, shape/workgroup variants, and historical cases.
4. Use calibration and checkpoint-selection evidence to select one external20
   seed; do not infer a deployable checkpoint from V12's lane selection.
5. Prove Transformers-to-Doppler adapter activation parity, then run semantic
   confirmation and promotion.
6. Productize WGSL Doctor only after the exact hosted adapter clears the
   semantic contract. Never auto-apply a candidate that fails compilation.
