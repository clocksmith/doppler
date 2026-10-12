# Original 2B nonthinking document control

The preceding thinking configuration failed all four frozen document cases; three exhausted 4,096 generated tokens. This control changes only the official template’s `enable_thinking` value to false. It preserves the original authorized checkpoint, ordered histories, greedy sampling, repetition penalty 1.1, factual and citation criteria, and 4,096 generated / 8,192 total token / 16,000 UTF-16 character budgets.

The [freeze](freeze.json) binds the complete [suite](suite.json) and two disjoint batches. Each ordered conversation stays in one process. The original template supplies an empty, closed thinking block in the prompt; acceptance still requires natural completion and every factual criterion. No answer text, new instruction, seed selection, or production model change is introduced.

Original source quality only. A passing source configuration would still require independent converted-model, runtime, lifecycle, and application acceptance. All four source cases completed naturally within the fixed budgets, but all four fail their full factual/citation contracts. See the [review](review.json), [input and output audit](output-audit.json), and compressed raw reports/logs. No converted-model run is justified by this result.


## Frozen repetition-penalty ablation

The [next control](neutral-freeze.json) changes only repetition penalty1.1 to1.0. Both the source evaluator and Doppler apply that penalty to the complete prompt and generated history, so repeated source facts and citation identifiers receive it. The retained official Qwen3.5 family guidance recommends1.0; other recommended sampling settings remain unchanged here. This gives a specific mechanism to test, not a promise of improvement.

The exact four ordered cases, official nonthinking template, original weights, numerical precision, factual/citation criteria and budgets remain fixed. No production policy changes. If it passes, fresh held-out cases and an actual closed-loop follow-up remain required before converted-model qualification.
