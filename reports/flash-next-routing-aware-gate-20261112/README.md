# Flash-Next: routing-aware gate dry run

**Question.** What does the proposed
[routing-aware acceptance gate](../../docs/routing-aware-acceptance.md) say
about the retained Nov-10 and Nov-12 Flash-Next artifacts?

**Answer.** It returns `fail`, and Flash-Next stays **BLOCKED**. No GPU or CPU
model ran for this report. [dry-run.json](dry-run.json) was produced by
[dry-run.js](dry-run.js) from committed artifacts only, and records their
hashes.

## Gate reasons (fail-closed)

| Code | Why |
| --- | --- |
| `POLICY_NOT_APPROVED` | All limits are proposed. No run can qualify before owner approval. |
| `POLICY_HASH_MISMATCH`, `PROMPT_SET_HASH_MISMATCH` | The retained runs predate the policy and the pinned prompt set. |
| `TUPLE_NOT_QUALIFIABLE` | The receipts record "AMD RDNA3, Node webgpu 0.4.0" but no exact adapter string. |
| `CASE_NOT_PINNED`, `CASE_MISSING` | `inventory` is not a pinned fresh case. None of the 22 pinned cases has run. |
| `INPUT_MISSING` | The per-token CPU router logits are not committed, and there are no same-input checks on the tested tuple. |
| `TASK_QUALITY_MISSING` | No task-quality results exist. |

## Informational, not admissible

The attribution rule was applied to the divergences in
[routing-trace.json](../flash-next-new-prompts-20261110/routing-trace.json).
Those were inferred from captured GPU router logits. They are not executed
readbacks.

- **Divergences:** 9,754 token-layers. One root is at Layer 24, token 112. The
  other 9,753 are its cascade, and no divergence falls outside that root's
  cone. The executed readback in the
  [forced-routing control](../flash-next-forced-routing-20261112/README.md)
  agrees: its first executed divergence is also L24/t112, and 9,754 token-layers
  differ.
- **Root margin:** the root's CPU flip displacement (expert 60 to 225) is
  1.049e-5. Under the proposed multiplier, the Layer-24 near-tie bound is
  2 × 3.03e-4 = 6.07e-4, so the root would count as a near-tie.
- **Forced-indices control:** 0 quantized-v1 violations and 0 executed
  differences. The GPU's own sets at the three forcing interventions were not
  recorded, so the own-choice near-tie rule cannot be checked.
- **Same-input checks:** they cover Layers 0 to 24 at position 112 of
  `inventory_prefix113`. The gate requires all 48 layers, plus final norm and LM
  head, with quantized-v1 violation counts.

This is consistent with the gate passing a fresh run, but it does not show that
one would. Passing still needs fresh unforced runs on the pinned prompts,
executed readbacks, full same-input coverage and the task-quality limits.

Component: `doppler.repository-tooling`, `doppler.tests`, `doppler.docs`.
Intent: preserved. Boundary effects: none.
