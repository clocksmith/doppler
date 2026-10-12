# Fresh short-document source screen

This screen tests the already-authorized Qwen3-1.7B original source with the thinking policy that previously passed 14 source tasks. It asks four smaller document questions under both retained seeds, including an ordered correction. It does not change the rejected multi-contractor suite or turn its failures into passes.

The [suite](suite.json) freezes every input, factual criterion, generation setting and limit. The [freeze](freeze.json) was written before execution. The unchanged CPU evaluator executes the original model, retains full thinking and final output, and requires manual factual review. [Execution identity](run-binding.json) binds the host, command outputs and tools. The run is in progress; no quality or deployment pass is claimed.

All eight answers must naturally finish with a single thinking boundary, preserve the cited facts, and remain within 4,096 generated tokens, 8,192 total tokens and 16,000 raw UTF-16 units. Natural prose is appropriate to these questions; earlier literal-output requirements remain intact for their own tasks. No seed selection, longer allowance, runtime change or new model is introduced.

A source pass would establish only this narrower task scope. Converted weights, installed runtime, physical memory, numerical accuracy and distributed execution remain separate requirements.
