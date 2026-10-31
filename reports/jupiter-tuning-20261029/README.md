# amd-host Qwen3.5 0.8B tuning checkpoint

This is Doppler work, not Doe work. The requested target is not fully met:
complete-request latency improved, but the retained comparison's short-prompt
first-token guardrail failed. No default, converted model, signed artifact,
publication claim, or other model/GPU lane is promoted here.

## Implementation

The opt-in JSON profile is
`profiles/qwen-3-5-0-8b-vulkan-subgroup8`. It extends the previously retained
Vulkan throughput profile. It uses eight-lane Q4_K projections and fused FFN,
a sixteen-lane materialized vocabulary head, partitioned head-256 prefill
attention, register-resident recurrent prefill/decode state, parallel prefill
convolution, session-owned decode scratch and exact binding reuse. Sequential
batch size 2/readback interval 32 remains explicit. Speculation is disabled.
Weights, tokenizer, F32 activations/accumulation, F16 KV and the 0.001 logit
tolerance are unchanged. Larger bounded caches are opt-in profile settings.

The greedy resolver now consistently handles zero temperature/threshold and
`topK=1`; head-fusion eligibility still guards masks, penalties and diagnostics.
The profile retains the materialized head because the tested fused head was
slower. Explicit fused FFN geometry no longer accidentally inherits an
incompatible separate-projection geometry. Device loss, borrowed-buffer
ownership and unsupported paired/MoE use remain visible failures.

## Measurements and their limits

`product-register-summary.json` describes 20 balanced pairs: four prompts,
five pairs each, 64 generated tokens, one same-prompt warmup then one timed
request in each fresh browser, warm OPFS, physical Radeon 8060S and the same
Chrome executable, prompt token IDs and context budget. No outliers were
removed. Both engines reached 64 tokens without an earlier EOS. Doppler's
outputs matched its frozen teachers; Pooled's outputs differ because weights
differ.

The configured register candidate's equally weighted paired complete-request
ratio was **0.93005**, with bootstrap 95% interval **0.90717–0.95237**. Those are
scoped product-format measurements, not an equal-weight engine comparison or
an application-quality result. Median request times, Doppler/Pooled:

| Prompt | Complete request ms | First token ms |
| --- | ---: | ---: |
| Code | 317.8 / 352.1 | 39.5 / 37.5 |
| Explanation | 326.0 / 353.5 | 40.1 / 43.5 |
| Arithmetic | 315.9 / 343.5 | 43.6 / 38.7 |
| Long context | 550.5 / 566.7 | 253.0 / 254.7 |

The full gate is **false**: code first-token latency regressed 5.33%, and
arithmetic regressed 12.66%, exceeding 5%. Doppler also delivers the remaining
63 tokens as a burst in this workload; Pooled delivers tokens individually.
Raw delivery timestamps are retained. Initialization/load/compilation are
excluded from these warm request measurements. An existing desktop Chrome
GPU process remained open; no other tuning process was intentionally run
concurrently.

The prior configured candidate, before register decode, is retained in
`product-summary.json`: ratio 1.00729, 95% interval 0.99470–1.01959, gate false.
The first screen and all intermediate/rejected attempts remain in the raw
archives. Overlapping attempts and the disk-full attempt are excluded from
performance conclusions with explicit receipts.

The timed register candidate is pinned by `source-register-identity.json`.
The delivery source is pinned by `source-delivery-identity.json`: unused
compensated-state code was removed from the scalar-only generated shader,
unsupported-lane guards were added, identical boolean validation was shared,
and static gate checks now follow the declared companion pass. These changes
were checked numerically, but **the cleaned delivery source has not received a
new repeated timing campaign**. Do not substitute its source identity into the
earlier timing receipt.

## Correctness and development checks

`full-logit-delivery-checks.json` checks the final configured batched route:
all 64 token IDs per prompt match, and all 248,320 materialized vocabulary
values at positions 0, 1, 2, 4, 8, 16, 32 and 63 pass unchanged absolute
0.001/relative 0 tolerance. Maximum absolute differences:

- Code: 0.00097990036.
- Explanation: 0.00081324577.
- Arithmetic: 0.00072312355.
- Long context: 0.00060051680.

Logit collection copies actual GPU-batched outputs into separate staging
buffers. Those instrumented runs are **untimed correctness evidence only**.
The initial naive-SiLU FFN candidate failed at 0.0012708902. Matching the
existing refined activation fixed that regression; failed arrays and
ablations remain retained.

Physical amd-host reference checks cover 36 projection cases, 12 refined FFN
cases, six recurrent prefill/history cases, six attention prefill cases and
12 immediate/recorded sequential register-decode cases. The final scalar-only
decode shader was rerun against the old GPU path and independent CPU state,
output and convolution-history references. Browser full-model checks are
separate from those Node/WebGPU operator tests.

`development-checks-initial.json` retains initial development failures.
`development-checks-final.json` records successful affected checks after fixes.
Earlier successful sampling, ownership, cleanup and identity checks remain in
`development-checks.json`. The routing audit is current; its 245 historical
integrity failures are unchanged, not resolved by this work. This is focused
development verification, not the full repository suite, a new release, other
hardware coverage, or independent application-quality qualification.

## Reproduction and remaining work

All source snapshots are deltas over `efd9549a`; each identity file records
its file hashes and the concurrent checkout HEAD. Apply deltas to a separate
copy, never in-place over hardlinked snapshot files. The source tree's own
CLI accepts the retained JSON configs. `screen.js`, the paired runners,
frozen contracts, hook, scripts and raw outputs are retained in the archives.
The paired runner uses the separately provisioned Pooled checkout/model/cache;
it does not redistribute that runtime or the weights.

Next work is to measure the short-prompt first-token gap, preserve finite-logit
and sampling semantics while removing demonstrated overhead, then freeze a
fresh delivery-source comparison. Inspection found that ordinary generation
currently reads the materialized first-token vocabulary back for CPU selection;
the existing GPU token-selection helper is a candidate to investigate, not a
completed optimization. Do not raise tolerance, drop prompts, change weights,
or collect extra samples of an unchanged candidate until a gate happens to pass.

Component: `doppler.runtime-source.gpu.kernels`, text inference, config and rules.
Intent: preserved.
Acceptance evidence: the focused checks, physical GPU logs, full-logit checks,
source identities and raw receipts retained here.
Boundary effects: new opt-in JSON settings and registered WGSL variants;
no Doe, model-artifact, default-profile, catalogue or application changes.
