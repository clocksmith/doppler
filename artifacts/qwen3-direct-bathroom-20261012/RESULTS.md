# Direct original-source result: rejected

All four predeclared pairs completed naturally within their unchanged budgets.
**Zero of four pass the complete factual/citation requirements.** Removing the
structured system instruction did not qualify this source configuration.

| Case | Seed | Generated tokens | Prompt + generated | Completion | Full factual review |
|---|---:|---:|---:|---|---|
| fresh-bathroom-comparison | 17 | 2016 | 2498 | Natural EOS | Reject |
| fresh-bathroom-correction | 17 | 1924 | 2521 | Natural EOS | Reject |
| fresh-bathroom-comparison | 53 | 2199 | 2681 | Natural EOS | Reject |
| fresh-bathroom-correction | 53 | 2629 | 3226 | Natural EOS | Reject |

The answers sometimes calculate the stated subtotal or deposit correctly, but
omit required headroom, invent exclusions or conditions, and confuse dates.
Comparison seed53 adds deposits as extra costs. Correction seed53 calls
September20 five days after September25 and changes September12 plus three
weeks to September25; it then recommends the late option as timely. Correct
individual facts do not cancel those failures. Full criterion decisions and
exact answer hashes are in `review.json`.

The retained run generated 8768 tokens and charged 10926 prompt-plus-output
tokens across four separately bounded pairs. Elapsed process time was
7320.784 seconds, including loading and concurrent correctness workloads;
this is execution accounting, not a throughput benchmark. Original weights,
ordered messages, both seeds, official template and all raw thinking/answers are
bound by `freeze.json`, `executing-host-identity.json`, `result-binding.json` and
the compressed original report/log. No outputs were repaired or selected away.

This is an original Qwen3-1.7B source result. It neither diagnoses a distributed
runtime defect nor qualifies deployed Qwen0.8B, converted weights or an archive.
The separately frozen source-ledger application experiment is a different
primary treatment; it must meet the same complete task requirements.

Component: source task-quality evidence. Intent: preserved.
Acceptance evidence: retained report, original criteria and manual review.
Boundary effects: none; no production model, policy, runtime or pin changed.
