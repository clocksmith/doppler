# Explicit citation-policy comparison

Completed all twelve original-source Qwen3-1.7B task/seed pairs. **Rejected:5/12 complete passes.** All answers reached EOS within unchanged budgets. On the eight short-document pairs, factual conclusions dropped from8/8 to6/8; complete passes remained5/8. Both retained subtotal corrections now repeat160 instead of the required185. Three of four contractor answers invent dates; the fourth confuses the start with completion. All four miss required totals/remainders, exact supporting passages, or unresolved exclusions.

The [input audit](input-audit.json) established that the baseline evaluator supplied the exact ordered messages, official template and token IDs, including citation requests. No evaluator/input defect justified a runtime patch. The sole candidate change was an explicit system citation instruction; it did not correct the observed quality problem.

[Review](review.json), [run identity](run-binding.json), [raw report](source.json.gz), and [generation log](source.log.gz) retain every final answer and failed criterion. Existing factual/citation criteria, source weights, seeds17/53 and4096-output/8192-context/16000-UTF16 budgets remain unchanged. This was an exposed paired diagnostic, not independent new-task qualification. No converted-model run, package promotion or application policy change follows this failure.

The separately frozen [official nonthinking comparison](../qwen3-nonthinking-documents-20261012/README.md) tests the pinned checkpoint’s documented mode and sampling recommendation without this added citation instruction.
