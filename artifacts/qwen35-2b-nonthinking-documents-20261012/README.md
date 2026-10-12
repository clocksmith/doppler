# Original 2B nonthinking document control

The preceding thinking configuration failed all four frozen document cases; three exhausted 4,096 generated tokens. This control changes only the official template’s `enable_thinking` value to false. It preserves the original authorized checkpoint, ordered histories, greedy sampling, repetition penalty 1.1, factual and citation criteria, and 4,096 generated / 8,192 total token / 16,000 UTF-16 character budgets.

The [freeze](freeze.json) binds the complete [suite](suite.json) and two disjoint batches. Each ordered conversation stays in one process. The original template supplies an empty, closed thinking block in the prompt; acceptance still requires natural completion and every factual criterion. No answer text, new instruction, seed selection, or production model change is introduced.

Original source quality only. A passing source configuration would still require independent converted-model, runtime, lifecycle, and application acceptance. No execution results yet.
