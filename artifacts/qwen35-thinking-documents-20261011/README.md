# Original 2B thinking document control

The frozen suite applies the original Qwen3.5-2B thinking template and unchanged greedy generation policy to four ordered-message document cases. It preserves the reviewed factual criteria, 4,096 generated tokens, 8,192 total tokens and 16,000 UTF-16 code units. No rejected system instruction is added.

The first source comparison is running on `x@128.tail995236.ts.net` in `/var/tmp/doppler-repetition-source-20261011`. The other three cases are frozen, not executed. The preceding [counting result](../qwen35-thinking-policy-20261011/README.md) passes one source case; the existing converted-model reasoning holdout remains rejected. This is source-quality investigation, not installed-package or distribution acceptance.

Each `document-*.json` freezes messages and exact original-template token IDs. `suite.json` owns the criteria. An empty token comparator in the existing generation replay deliberately supplies no expected GPU answer; its first-token divergence field is inapplicable. Only natural completion, budgets, the thinking boundary and factual review determine this experiment. No runtime or application configuration changes.
