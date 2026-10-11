# Repetition policy control: rejected

Original Qwen3.5-2B stops at 120 on the unchanged request to count through 200, both with full-history repetition penalty 1.1 and with the multiplier disabled at 1.0. All 372 generated tokens, including EOS, match the retained baseline. This rejects the proposed penalty explanation for this source failure; it does not explain the separate 0.8B concurrent answer.

The prompt IDs, original weights, greedy sampling, F32 equations, F16 KV, 4,096-token output allowance and 8,192-token context allowance were held fixed. Hypothesis and acceptance were frozen before execution. Raw result/log and exact candidate control are retained; no runtime or application setting changed.

Component: source-generation diagnostic. Intent preserved. Boundaries unchanged.
