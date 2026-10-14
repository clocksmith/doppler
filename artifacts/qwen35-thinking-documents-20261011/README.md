# Original 2B thinking document control: rejected

All four frozen ordered-message cases completed on `amd-host`. **None qualifies.** The source processes have exited; no GPU execution occurred.

| Case | Generated tokens | Stop | Result |
|---|---:|---|---|
| Fresh bathroom comparison | 4,096 | Token limit | Reverses included plumbing/electrical terms; incomplete. |
| Fresh bathroom correction | 4,096 | Token limit | Claims September 12 plus three weeks is September 23; misstates included tax; incomplete. |
| Contractor comparison | 3,973 | EOS | Claims all contractors meet June 1 and invents a plumbing allowance; natural completion does not establish factual quality. |
| Retained contractor correction | 4,096 | Token limit | Delivers only an introductory final-answer fragment. |

The [complete review](suite-review.json) binds every raw result, reports all budgets, and checks each unchanged factual criterion. The earlier [first-case review](review.json) remains a historical checkpoint; its then-pending cases are resolved by the complete review. The [remaining-run receipt](remaining-run-binding.json) records the completed commands and output hashes. Raw compressed results retain both thinking and final answers. Factual review considers the delivered answer after the closing thinking boundary; internal thinking is not a substitute for it.

The [frozen suite](suite.json) uses the original Qwen3.5-2B thinking template, unchanged greedy generation policy, 4,096 generated tokens, 8,192 total tokens and 16,000 UTF-16 code units. No rejected system instruction, seed selection, longer budget, reference change or runtime correction is introduced. Every result stays within those limits, but three do not naturally finish. The preceding [counting result](../qwen35-thinking-policy-20261011/README.md) remains a passing single source case, not document qualification. The converted-model reasoning holdout remains rejected.

Each `document-*.json` freezes messages and exact original-template token IDs. `suite.json` owns the criteria. The empty token comparator in the existing generation replay supplies no expected GPU answer; its first-token divergence field is inapplicable. These CPU source-weight controls use F32 equations and F16 KV storage. They do not qualify converted weights, an installed package, physical GPUs, or distribution, and do not establish that every generation task fails.

Component: `doppler.repository-tooling` / retained model-quality evidence. Intent: preserved. Acceptance evidence: four hash-verified results and `suite-review.json`; all four rejected. Boundary effects: none. Runtime and application settings remain unchanged.
