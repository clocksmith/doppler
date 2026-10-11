# Original-weight contractor conversation: rejected

Two fresh CPU responses completed naturally within the frozen budgets. Both fail
the factual criteria. This closes this configured source-control experiment with
a rejection; it does not qualify a model, package, partition, or application.

## Executed control

Original `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
weights were represented exactly in F32, with eager source equations and stored
F16 K/V. This is not the model's default native BF16 implementation. The receipt
records the source files, packages, tokens, arithmetic, and sampler. It used the
previously successful thinking policy: temperature 0.6, top-p 0.95, top-k 20,
min-p 0, repetition penalty 1. Limits remained 4,096 generated tokens, 8,192 total
tokens, and 16,000 raw decoded UTF-16 characters.

The [suite](suite.json) freezes the actual retained Reploid messages and review
criteria before execution. The second case contains the historical incorrect
assistant response, not the first answer produced in this run. Histories are
passed as ordered messages, not flattened into one user prompt.

| Seed 17 case | Generated / total tokens | Raw characters | Completion | Factual review |
| --- | --- | --- | --- | --- |
| Initial comparison | 1,874 / 2,360 | 7,333 | Natural EOS; within budgets | Fail |
| Retained-history correction | 2,435 / 3,995 | 9,884 | Natural EOS; within budgets | Fail |

The comparison calculates Cedar's $10,550 subtotal but also calls Maple cheapest
and Cedar most expensive. It assigns Maple's $3,240 deposit to removal, although
removal is included. It confuses Cedar's May 20 start with completion and misses
the three-week deadline conflict.

The correction response repeats facts but ends by approving the historical
comparison. It misses the removal reversals, misattributes Cedar plumbing to
`D3:P1` instead of `D3:P2`, and says no further Harbor electrical work is mentioned
despite the approval requirement above the allowance. These failures concern
the substance of the answer, not just citation punctuation.

The [manual review](review.json) is agent review against the frozen criteria,
not a human playtest or an estimate of general model accuracy. The unmodified
machine report retains `passed: null` for prose awaiting review; the separate
review records the negative verdict. EOS never counts as factual acceptance.

## Retained evidence and stopping

- [result.json.gz](result.json.gz): complete inputs, tokens, reasoning, final
  answers, completion checks, source identities, and configuration.
- [run-binding.json](run-binding.json) and [executed-source.tar.gz](executed-source.tar.gz):
  exact executed entrypoint and evaluator. Their hashes match the report.
- [run.log.gz](run.log.gz) and [controlled-stop.json](controlled-stop.json):
  after both seed-17 cases completed, the owned process was interrupted to avoid
  repeating a configuration already rejected by the frozen criteria. Seed-53
  cases did not complete and are neither failures nor passes.
- [interrupted-snapshot.json.gz](interrupted-snapshot.json.gz): a failed attempt
  to copy the report while it was overwritten. This is a reporting defect, not
  a model failure. Subsequent tooling commit `4528e012` writes reports atomically;
  its failure-injection regression proves an interrupted write preserves the
  prior report. That repair did not change the evaluator already running here.

The CPU process is stopped. The existing public contributor was left running.
No GPU package, shader, precision, acceptance reference, dependency, or deployed
application was changed. This result shows these errors can arise in the
configured original-weight control before Doppler conversion. It does not prove
all policies or implementations of Qwen3-1.7B fail, nor explain away Doppler's
separate numerical discrepancy.

Component: source-model quality diagnostic. Intent: preserved. Acceptance:
ordered-message and atomic-write regressions pass; this source candidate fails
the frozen factual criteria. Boundary effects: diagnostic reporting only.
