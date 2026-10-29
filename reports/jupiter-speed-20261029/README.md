# Jupiter speed reconciliation and scoped configuration improvement

The opt-in Vulkan configuration passed the frozen complete-request gate on
Jupiter: **11.18% lower latency**, bootstrap 95% interval **8.60–13.62% lower**.
Forty uninstrumented requests cover four prompts and five alternating pairs per
prompt. Every scored request produced the same 64 token IDs as its counterpart.
All 32 sampled full-vocabulary vector comparisons passed absolute tolerance
`0.001`, relative tolerance zero; maximum observed difference was `0.000834942`.

This validates the complete configuration and workload in the retained receipts,
not all uses of the profile. The profile remains experimental and opt-in. No
manifest, signed model, catalog entry, default configuration, kernel or foreign
engine was replaced. This is not an equal-quality Pooled comparison.

## Authoritative baseline and reconciliation

[baseline-identity.json](baseline-identity.json) binds 3,056 executed package files,
20 model files, environment output, and model byte sizes. All tracked regular
source files in the executed package match Doppler
`229cbf05e0d72595d0d4c1edf029d6077964c082`. The only extra candidate asset is the
explicit profile. The packaged compatibility directories are expanded aliases;
source verification compares their canonical tracked regular files.

Hardware: Jupiter, Ryzen AI MAX+ 395, Radeon 8060S / RADV STRIX_HALO; Chrome
146.0.7680.177, Node 22.22.1, Linux 7.0.0-38-generic, Mesa Vulkan 26.0.3-1ubuntu1.
The raw requests bind browser arguments, physical adapter, features, prompt IDs,
RDRR Q4_K_M model/tokenizer hashes, F16 embedding/KV and F32 activation paths,
2,048-token allocated KV capacity, greedy sampling and no speculation. Warm OPFS
loading, one warmup and one timed request per browser opening; model loading and
full harness wall are retained separately. No OS-cache flush or unrelated service
shutdown was performed. Source identity capture ran during part of the timed
population; all samples remain included.

The newer evidence lives under
`/home/x/deco/hatch/pooled-doppler-components-20261029/deep-profile/`, outside the
Doppler reports searched in the earlier review. [prior-evidence.json](prior-evidence.json)
verifies the original causal checkpoint's bound hashes, including its full-logit,
prepared-core and topology-preserving timestamp receipts. The original
`JUPITER-DECISION-REPORT.md` predates those later files and must not be used alone.

Reconciled findings:

- Intended Vulkan features with original cadence: 101.29 versus 120.39 tok/s in
  the prior controlled population; sampled complete logits passed unchanged
  tolerance. The new multi-prompt experiment below independently tests that lead.
- Prepared core: 116.99 versus 119.59 tok/s medians, mixed paired outcomes; no
  established gain. Its performance runs had zero binding hits because upstream
  resource identities changed. No prepared-core patch is promoted.
- Topology-preserving timestamps: GPU pass intervals occupy 79–88% of observed
  decode windows. Recording overlaps GPU work, and pass elapsed time can include
  scheduling/preemption. Observer perturbation reached 8.12%; neither active GPU
  utilization nor host-only recoverable time follows from these aggregates.
- Corrected optimized attention, rather than the accidentally selected generic
  path, still trails Pooled in the retained same-input synthetic shapes.
- Product-format Pooled runs use different quantized weights and produce different
  sequences. They do not establish an equal-quality engine speed ranking.

All prior failed attempts remain in their original campaign directories. The
prior archive/inventory is not overwritten. The new timed and numerical populations
have zero failed requests. Two evidence-script failures are recorded in
[attempts.json](attempts.json); neither changed numerical tolerances or selected
which performance samples survived.

## Measured configuration patch

[contract.json](contract.json) was written before the new runs. Baseline is the
manifest's original execution. Candidate uses
`profiles/qwen-3-5-0-8b-vulkan-throughput`, which extends the existing Vulkan
feature package while explicitly retaining sequential **4 x 8** readback cadence.
Both runtime batching and session decode-loop settings agree. The existing 4 x 4
parity profile and all defaults remain unchanged.

Medians below each represent five complete requests, not tail guarantees:

| Prompt | Input tokens | Baseline complete ms | Candidate complete ms | Baseline decode tok/s | Candidate decode tok/s |
|---|---:|---:|---:|---:|---:|
| Python two sum | 28 | 732.5 | 643.0 | 98.38 | 112.58 |
| Hash-map explanation | 31 | 783.8 | 676.6 | 90.94 | 113.02 |
| Arithmetic explanation | 34 | 688.7 | 658.0 | 104.58 | 113.49 |
| Cache workload context | 655 | 1381.0 | 1263.2 | 92.13 | 114.13 |

Model-initialization medians are measured separately from the warm request:

| Prompt | Baseline initialization ms | Candidate initialization ms |
|---|---:|---:|
| Python two sum | 1627.5 | 1605.1 |
| Hash-map explanation | 1651.3 | 1615.1 |
| Arithmetic explanation | 1612.6 | 1527.4 |
| Cache workload context | 1410.2 | 1340.2 |

These are fresh pipeline openings against warm local OPFS, not cold network or
cold-disk results. Whole harness elapsed time is retained for every opening;
Chrome startup and the warmup are excluded from the primary request estimator.
The campaign does not isolate configuration setup as the cause of any loading
difference or assert a separate loading improvement.

The primary estimator equally weights each prompt's mean paired log complete-
request ratio. It is not a ratio of pooled marginal medians. The candidate ratio
is `0.888235`, bootstrap 95% interval `[0.863812, 0.913966]`. It passes the frozen
2% minimum gain, positive confidence bound, per-prompt 3% complete-request and 5%
first-token regression limits. Arithmetic first-token medians were 94.8 versus
96.6 ms; cache-context medians were 708.2 versus 712.1 ms. Those small increases
are disclosed rather than hidden behind throughput.

[summary.json](summary.json) contains every timing sample, min/max, paired gains,
per-prompt setup costs and gate results. The experiment does not establish cold
network/disk loading, long decode, maximum context or application quality.

## Configuration and execution audit

[config-audit.json](config-audit.json) keeps requested settings, resolved command
settings, observed execution plans/cadence, fallback transitions/guard reasons,
and allocation counters separate. Resolved command settings precede manifest
compilation; they are not mislabeled as the final numeric session.
[dispatch-audit.json](dispatch-audit.json) records actual compiled module labels,
entrypoints, constants and dispatches in a separate untimed harness. Both ordinary
route execution and teacher-logit replay are included; counts are not per-token
GPU cost estimates. There are no timestamp probes or extra compute-pass splits.

The audit catches an important dormant option: **greedy LM-head fusion was requested
but did not execute**. The workload explicitly sets `temperature=0` and
`greedyThreshold=0`; `shouldUseGreedyLmHeadArgmaxFusion()` requires
`temperature < greedyThreshold`, so eligibility fails. Both lanes execute the
ordinary argmax path. This campaign therefore establishes no fused-output-head
speed or numerical claim. Using a positive greedy threshold would be a new
execution configuration needing its own frozen test, not an automatic promotion.

The candidate does execute the fused recurrent decode core, reduced projection
path and `attention_decode_online_head256_f16kv_output_gate/main`; the baseline
uses `attention_decode_online_f16kv/main`. Dispatch receipts establish selection,
not how much each feature contributed to the combined request gain.

## Ranked bottleneck experiments

These ranks identify measured targets and gaps, not a fabricated exclusive
partition of the critical path. JSON parsing has no demonstrated bottleneck.

| Rank | Target | Evidence | Disposition |
|---|---|---|---|
| 1 | Select existing Vulkan features without changing cadence | New 40-request population passes the fixed request and numerical gates | Retain opt-in profile for this exact tested scope |
| 2 | Ordinary attention at longer contexts | Corrected same-input optimized decode: 0.3377 vs 0.0734 ms at 4K, 1.3369 vs 0.2066 ms at 16K, Doppler vs Pooled; CPU oracle passed | Target split/work decomposition with independent references; synthetic ratios do not predict full-model gains |
| 3 | Quantized projections / FFN | Equal decoded synthetic weights and packed byte counts: 4096x4096 took 5.505 vs 2.2675 ms; 16384x4096 took 21.790 vs 13.1125 ms, per 64 projections | Investigate projection geometry and FFN separately with matched weights; no foreign-engine transplant selected |
| 4 | Output-head eligibility and recurrent costs | Head fusion is ineligible in this workload; recurrent fusion is observed, but no topology-preserving per-operator critical-path allocation exists | Freeze separate eligibility/geometry ablations; preserve ordinary production route and full logits |
| 5 | Broader prepared execution / allocation reuse | Prior prepared-core and bounded scratch attempts fail improvement requirements | No caching patch retained; require stable lifetimes plus a new complete-request win |

The new candidate does not change shaders. FFN, recurrent attention, ordinary
attention and output-head GPU times are not invented from dispatch counts or CPU
profiles. An operator attribution experiment remains open: timestamps inside or
around production passes need coverage and perturbation checks before ranking
exclusive costs. Current pass timestamps only establish the enclosing GPU window.

## Diagnostic correctness patch

`decodeBottleneck.schemaVersion: 2` retains observed intervals but leaves
`bottleneckClass` null and causal attribution unresolved. It no longer subtracts
GPU timestamps from completion waits, sums overlapping aggregates into a residual,
or subtracts independently aggregated readback subcomponents. Those derived fields
are null. `dominant` means largest observed aggregate interval; its shares overlap
and are not additive. Historical version 1 receipts stay unchanged.

The compare adapter delegates to the same calculation as the browser benchmark;
CLI and compare summaries explicitly label the interval without claiming a causal
bottleneck. This fixes an evidence-validity bug; it is not itself a speedup.
[main-smoke-verification.json](main-smoke-verification.json) confirms a physical
patched-main run preserves all 64 expected tokens and emits version 2 diagnostics.
That smoke is excluded from the performance estimator.

## Broader model evidence: open gates

The requested next dense overlap is **Qwen3.8 27B**: both repositories declare its
execution family and Jupiter has a local Doppler artifact. A fresh matched-byte
inventory, supported Pooled GGUF, numerical gate and resource budget are required
before its short/long prefill, decode, cold opening and memory campaign. No 27B
performance is inferred from the 0.8B result.

Pooled declares **Qwen3.6-35B-A3B MoE**. This Doppler revision has no corresponding
qualified conversion/runtime model lane or provisioned matching artifact. Doppler's
Gemma MoE recipe is a different architecture and is not a supported overlap with
this Pooled revision. The MoE comparison is blocked on a genuinely supported
common workload, not just spare Jupiter RAM. No model was silently downloaded,
new family onboarded, or MoE result manufactured. Larger-model and MoE measurements
remain unfinished.

Live/peak API allocation counters are retained in the raw results. They are not
physical GPU residency or total-process memory; the APU shares system RAM and
explicit-destroy accounting can miss garbage-collected resources. Cold loading and
resident memory remain separate measurement gates.

## Checks and reproduction

Passed focused overlap/CLI diagnostics, browser batching metrics, compare-engine
contract, runtime-profile discovery and onboarding checks; source style/unused and
source/declaration type checks. Jupiter also ran the focused diagnostics, batching
and comparison tests. The source architecture check fails on unchanged
`src/gpu/command-recorder.js` (776 lines vs its 750-line limit); no budget was raised.
No full suite or release/benchmark promotion campaign is claimed.

[raw-evidence.tar.xz](raw-evidence.tar.xz) contains all scored configs/results,
untimed logit vectors, dispatch hooks, logs and physical smoke. The runtime archive
contains the exact baseline package plus candidate profile, excluding dependencies
and model weights. Baseline hashes, source hashes and model hashes remain explicit.
`SHA256SUMS` pins the delivered bundle.

Reproduce in a fresh directory on Jupiter; never overwrite the retained runs:

1. Copy this report's scripts and `contract.json` to a new directory. Extract
   `runtime-snapshot.tar.gz` into its `runtime/` subdirectory and link the same
   installed Doppler `node_modules` there. The contract's local model and prior
   config paths must exist; relocations need a separately identified contract.
2. Run `node run-profile.js`; it refuses an existing `runs/` directory. No probes
   or profile timestamps enter the scored population.
3. Run `python3 prepare-gates.py`, then `node run-gates.js`. This builds an isolated
   diagnostic runtime and refuses an existing destination.
4. Run `python3 summarize.py`. It applies the unchanged frozen gates and retains
   every sample; numerical audit timing is excluded from performance.

Component: `doppler.runtime-source.config`, `doppler.runtime-source.inference`,
`doppler.runtime-source.cli`, `doppler.repository-tooling`, `doppler.tests`.
Intent: preserved. Acceptance evidence: frozen contract, 40 scored requests,
32 numerical vector comparisons, dispatch audit, patched-main smoke and focused
checks above. Boundary effects: explicit opt-in profile and versioned observation
semantics; no arithmetic, resource-lifetime, default, signed-artifact or application
policy change.
