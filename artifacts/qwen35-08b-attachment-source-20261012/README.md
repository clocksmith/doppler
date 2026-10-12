# Original 0.8B control for the actual attachment failure

Component: doppler.tools. Intent: preserved. Boundary effects: test-only source evaluator and exact Reploid input evidence; no runtime, shader, package, or application change.

The distributed application answered the three-ticket question and its four-ticket follow-up with the same unit-price restatement, omitting both totals and the requested citation. This suite freezes those exact ordered messages, including the retained incorrect assistant response, from Reploid `1db09d20`. It preserves greedy decoding, repetition penalty 1.1, and 4,096 generated / 8,192 context limits. The existing source safety cap is 16,000 raw UTF-16 units. No additional instruction or seed is selected.

The original Qwen3.5-0.8B checkpoint at `2fc06364715b967f1860aea9cf38778875588b17` was obtained with only weights, tokenizer, configuration, index, template, and README. `download-receipt.json` binds every local file to its upstream LFS SHA-256 or Git blob identity. Original and deployed tokenizer vocabularies match across 248,070 entries.

`template-comparison.json` establishes that the first official-template input has exactly the same 127 token IDs as the actual accepted-runtime diagnostic. This excludes template/tokenization mismatch for that input. The runtime rejected its prefill allocation before generation; its receipt is not a model-quality result. At freeze time, the second runtime input-token comparison was unmeasured; the completed physical comparison below now covers both.

The original-weight tool now accepts an explicitly frozen ordered-message suite, reusing the existing task evaluator and F32 source equations with F16 KV storage. Original source weights and tied output head remain independent of deployed quantization. Pure identity regressions and the existing source-task messages test pass on Linux. Source generation completed both cases with natural EOS: $36 in 32 tokens and $48 in 40 tokens. Both original-weight arithmetic answers are correct. Citation formatting fails: the first names D1:P1 without the exact bracket marker, and the second omits the source identity. These results do not qualify the application profile. The deployed runtime previously repeated only the unit price; the next attribution control uses the exact deployed Q4 weights with independent source equations and identical input IDs/policy.

## Deployed-weight attribution

The independent CPU control verified the pinned model pieces and decoded exact deployed Q4/F16 tensors. Both inputs naturally produce the same 22-token unit-price restatement as physical distributed execution, byte-for-byte, rather than the correct original-weight quantities. This rules out the mesh or stale conversation state as necessary causes of these two failures. It does not yet identify which converted tensor causes the loss.

The conversion explicitly requests F16 embeddings with a separately quantized Q4 head despite source tying. That is an intentional policy to investigate, not established unintended converter behavior. Any tied-head sensitivity control is a separate diagnostic configuration and does not change approved numerical target weights or production precision. The generation replay's historical `dopplerToken` comparison field denotes the original-source baseline in these explicitly labeled control fixtures.

## Tied-head diagnostic precondition

The first tied-head diagnostic stopped before inference because it incorrectly required every original BF16 embedding value to survive F16 representation exactly. The full comparison found 54,169 differences among 254,279,680 values, maximum 5.94e-8, localized to subnormal rounding; the original and retained conversion origin both pin revision `2fc06364715b967f1860aea9cf38778875588b17`. Normal values agree. This is not evidence of different checkpoint identity. The March artifact predates the separately retained September F16 rounding repair.

The rejected precondition script/log are preserved. The revised diagnostic pins the exact deployed embedding byte hash and changes only the output-head reference to that same resident embedding; it does not repair embedding bytes or other tensors. Original equality is reported as false rather than assumed. This remains a separately identified CPU sensitivity test, not a production precision change or replacement numerical target.

The completed tied-head control still answers the first question with only the $12 unit price (35 tokens). It computes $48 on the follow-up (34 tokens), but omits citation markers in both. Output-head quantization affects behavior; restoring tying alone is insufficient. No precision/model change is promoted. The intervened embedding remains the exact previously deployed byte hash.

`physical-token-comparison.json` now binds the successful AMD standalone control using the accepted archive plus separately identified allocation repairs. Both runtime input sequences (127 / 166 tokens) exactly equal official original-template sequences; both runtime output sequences (22 tokens each, including EOS) exactly equal the independent deployed-weight CPU outputs. The preserved memory-limit rejections are not used as generation evidence.
