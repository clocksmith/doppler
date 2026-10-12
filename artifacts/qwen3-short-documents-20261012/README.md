# Fresh short-document source screen

This screen tests the already-authorized Qwen3-1.7B original source with the thinking policy that previously passed 14 source tasks. It asks four smaller document questions under both retained seeds, including an ordered correction. It does not change the rejected multi-contractor suite or turn its failures into passes.

The [suite](suite.json) freezes every input, factual criterion, generation setting and limit. The [freeze](freeze.json) was written before execution. The unchanged CPU evaluator executes the original model, retains full thinking and final output, and requires manual factual review. [Execution identity](run-binding.json) binds the host, command outputs and tools. The CPU run completed with eight natural answers. All eight factual conclusions are correct, but three required supporting citations are absent. The full screen therefore rejects this configuration; no quality or deployment pass is claimed.

All eight answers must naturally finish with a single thinking boundary, preserve the cited facts, and remain within 4,096 generated tokens, 8,192 total tokens and 16,000 raw UTF-16 units. Natural prose is appropriate to these questions; earlier literal-output requirements remain intact for their own tasks. No seed selection, longer allowance, runtime change or new model is introduced.

A source pass would establish only this narrower task scope. Converted weights, installed runtime, physical memory, numerical accuracy and distributed execution remain separate requirements.

## Completed result

The [manual review](review.json) records **5/8 complete passes**, **8/8 correct factual conclusions** and **8/8 natural completions**. Seed 17 omits R7 in the deadline answer and R9 in the corrected-subtotal answer; seed 53 omits R5 in the conditional-dispatch answer. The [raw report](source.json.gz) retains every input, token, thinking segment, final answer and package identity; the [execution log](source.log.gz) retains progress. No omitted citation is silently supplied by the reviewer.

Maximum observed use was 612 generated tokens, 694 total tokens and 2,020 raw UTF-16 units. These are CPU source-task measurements, not GPU allocated-memory or latency claims. The complete eight-case result provides evidence of bounded useful content while leaving dependable citation coverage unresolved. It does not justify promoting the converted artifact or weakening the earlier contractor and literal-output requirements. No follow-on converted run was started after this source screen failed.

Component: `doppler.repository-tooling`. Intent: preserved. Acceptance evidence: hash-verified eight-case source report and final-answer manual review, with three failures retained. Boundary effects: none; runtime and Reploid configuration unchanged.
