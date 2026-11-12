# Flash-Next inventory: forced-routing control

**Question.** Does Doppler's GPU forward pass the unchanged
`doppler.boundary-tolerance/quantized-v1` gate on the 2,187-token `inventory`
prompt when the MoE top-k expert selection matches the F32 CPU reference?

**Answer.** Yes. With the CPU reference's expert sets at every layer and
token, the GPU stays well within the gate. Without forcing, it fails as
before. This is a counterfactual control. It does **not** qualify the model,
and the tolerance is unchanged.

| Run (last position, 248,320 logits) | Max absolute error | RMS error | Violations | Greedy token |
| --- | ---: | ---: | ---: | --- |
| Unforced (main `a542a64f`) | 0.467271 | 0.0737 | **30,664** | 271 / 271 |
| Indices forced (GPU weights for the CPU set) | 0.0000491 | 0.0000070 | 0 | 271 / 271 |
| Indices and weights forced (CPU weights) | 0.0000477 | 0.0000057 | 0 | 271 / 271 |

The unforced logits from main are byte-identical (`04c41923…`) to the
retained frozen-runtime observation in
[flash-next-new-prompts-20261110](../flash-next-new-prompts-20261110/README.md).
The comparator reproduces that report's 0.467271 / 30,664 result.

## Routing picture

These are live readbacks of the executed GPU index and weight buffers after
`softmax_topk_strided.wgsl`, not rankings inferred from logits.

- **Unforced:** the first executed difference is at Layer 24, token 112.
  There are 9,754 differing token-layer expert sets through Layer 47, which
  matches the earlier inferred count. Layers 0–23 have no differences. Router
  logits reach 10.6 absolute from the CPU values by Layer 37.
- **Forced (both modes):** 0 executed differences at all 48 layers. Forcing
  changed the GPU's own choice at very few token-layers:
  - indices mode: 3 (L23/t2126, L24/t112, L25/t1335)
  - indices-and-weights mode: 2 (L24/t112, L25/t358)

  Under forcing, the largest difference between GPU and CPU router logits
  across all layers is 0.0089 in indices mode and 0.0037 in
  indices-and-weights mode.
- **Indices-mode weights:** the GPU computes them from its own logits over the
  masked set, so they differ from the unforced path by F32 rounding. That is
  why its trajectory, and its near-tie flips, are not identical to the
  unforced run before Layer 24. The largest executed difference from the CPU
  weights is 2.7e-4.

The CPU reference has its own near-ties. The smallest rank-10/11 logit margin
is 4.8e-7. Margins fall below 1e-5 at 26 token-layers and below 1e-4 at 326,
out of 104,976.

## What this establishes

- The entire gate failure comes from discrete expert-selection flips at
  near-tied ranks. With routing matched, every other operator stays within the
  gate, and the residual of 4.9e-5 is about 10,000 times inside it.
- The gate cannot distinguish "wrong arithmetic" from "a different but
  equally rounded expert choice at a margin of about 1e-5". The earlier
  same-input controls found no shader defect in Layers 0–24.

## Owner decision (not made here)

1. **Keep failing.** The gate remains a strict trajectory match. Long
   prompts fail until the GPU's upstream rounding is close enough to the CPU's
   that no near-tie flips. The CPU's own margins (down to 4.8e-7) make that
   target prompt-dependent.
2. **Routing-aware acceptance.** Accept when the unforced run fails only
   because of selection flips, and every flip occurs where the CPU's rank-k
   margin is below a stated bound (for example 1e-4). The run forced to the
   CPU selection must also pass the unchanged gate. This needs a declared
   margin, a forced-run requirement and a policy version. Any
   higher-precision change would be a separate evaluation.
3. **A different reference for routed models.** For example, a
   distributional or task-level check instead of logit identity. This changes
   what the gate measures.

## Mechanism and reproduction

- `tests/helpers/reference-routing-selection-control.js` is a test-only port:
  - It patches `MoERouter.prototype.computeRouterLogitsGPU` to mask non-reference
    experts to `-FLT_MAX`, so the unchanged top-k shader selects the CPU set.
  - In `indices-weights` mode, it writes the CPU weights by expert ID after top-k.

  It is never imported by `src/` and does not touch runtime config. Its unit
  test is `tests/inference/reference-routing-selection-control.test.js`.
- The driver is `tools/run-flash-next-forced-routing.js`. It runs the
  provisioned `flash-next-learned-physical` test with the existing
  `moe-routing-observation-loader`, and stamps each observation with
  `counterfactual` and `normalModelQualified: false`. The standard comparator
  refuses such observations.
- `prepare-routing-fixture.py` converts the retained CPU F32 source router
  rows to the fixture. It checks the hashes against the source-trace receipt
  and confirms that the experts are the top-k of the CPU logits. It never reads
  GPU output.
- `compare-forced-routing.py` applies the comparator's arithmetic and policy.

Each run held a shared GPU lock and was niced. No service, tolerance,
artifact, profile or catalog state changed. Full per-layer rows and hashes are
in [comparison.json](comparison.json).

Component: `doppler.tests`, `doppler.repository-tooling`. Intent: preserved.
Boundary effects: none.
