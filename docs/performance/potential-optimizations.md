# Potential Doppler optimizations

Move work from repeated execution into validated preparation, while keeping model
identity, target-specific execution plans, and private cached state separate.
This is the single maintained inventory of potential optimizations. Implement
**1 → 2 → 3 → 4 → 5**, selecting a measured bottleneck and qualifying each change
independently. Extend the existing prepared-execution machinery. Published
speedups motivate experiments; they are not Doppler performance promises.

## Prepared execution ownership

| Boundary | Responsibility |
| --- | --- |
| Portable manifest | Model and tokenizer identities, tensor semantics, quantization identity, architecture, and compatible artifact references. |
| TargetPlan | Device-specific representation loading, packed layouts, kernel variants/constants, binding recipes, shape-specific schedules, scratch lifetimes, allocation, target requirements, and invalidation rules. |
| Capsule | Signed implementations and immutable prepared assets bound to their model and TargetPlan: packed layouts, qualified indices, and continuation automata. Only explicitly distributable checkpoint assets belong here. |
| Runtime cache | Device-bound objects and permission-scoped tokenization, embeddings, prefix checkpoints, and conversation state. Private prompts and state never become part of a publicly distributed model package. |

Signing establishes provenance, not correctness or confidentiality. Device
objects remain local instantiations of a validated plan. Prepared immutable
parameters may survive conversation resets; dynamic sequence state must be
cleared or restored from an authorized exact checkpoint. Unload and device loss
invalidate owned resources. Do not add another artifact format or planner where
an existing contract can be extended.

Reduce local repeated work before adding WAN partitioning. Prefer a whole-model
participant when it fits the exact model and context. Partitioning is principally
justified by capacity; treat a latency benefit as a separate hypothesis requiring
transfer, coordination, and complete-answer measurements.

## 1 Per tensor representation reachability

Compile every tensor and alias as `packed-only`, `dense-only`, or `both`, including
all reachable fallback kernels. Load no representation the selected plan cannot
consume. A packed consumer elsewhere in the graph cannot justify dropping a
dense representation required by another reachable consumer; tied weights and
aliases must be analyzed together. Remove dense shadows only after proving that
every reachable consumer accepts the retained representation.

The text loader now compiles these requirements in
[`weight-representation-plan.js`](../../src/loader/weight-representation-plan.js)
before reading weights, using shared loader bindings and the runtime's exact
layer/phase variant resolver. The Q4 materializer retains independent F16/F32
copies when required. Explicit fallback plans, composite FFN consumers, physical
aliases, tied endpoints and stored-F16 adaptive fallbacks participate in the
union. Unknown consumers remain conservative; legacy callers without resolved
paths retain their existing loading policy. This is not universal optimization
of every quantization format or implicit sharing of alias allocations.

The [frozen comparison and open gates](../../reports/weight-representation-20261017/README.md)
own qualification status. A physical mixed-consumer fixture demonstrates one
removed F16 shadow while preserving fallback copies. The final 2B comparison on
M3/AMD and 27B comparison on AMD retain identical model and peak pool bytes;
these profiles were already packed. Their timings establish no speedup. M3 27B
and the separate hosted CI gates remain open in that report.

Expected benefit: less startup conversion and GPU residency. Removing an F16
shadow saves roughly `2N` bytes for `N` elements; F32 saves `4N`, before alignment
and allocator effects. This does not inherently reduce downloaded weight bytes
or model arithmetic. Count packed and dense lifetimes separately.

## 2 Static dispatch and memory plans

Extend [execution-v1 preparation](../../src/inference/pipelines/text/execution-v1/plan.js)
and [resident pipeline preparation](../../src/inference/pipelines/text/resident-pipeline-preparation.js).
TargetPlan should declare binding recipes, uniform-ring ownership, kernel
constants, tensor lifetimes, scratch requirements, alias-compatible arena
offsets, and invalidation rules.

Prepare compatible bind groups once buffers stabilize, starting with the
[matmul wrapper](../../src/gpu/kernels/matmul.js) and
[dispatch owner](../../src/gpu/kernels/dispatch.js). Include pipeline layout,
buffer identity, offsets, sizes, and device identity in compatibility checks.
Rebuild after relocation, device loss, or incompatible session state. Cached bind
groups must never observe another request's uniforms or reclaimed scratch.

Derive scratch slots and safe reuse from the selected graph's actual lifetimes;
extend [decode buffers](../../src/inference/decode-buffers.js), the
[buffer pool](../../src/memory/buffer-pool.js), and
[command recorder](../../src/gpu/command-recorder.js). Keep concurrent requests
and pending GPU operations isolated. Use shape-specific arenas where justified,
including context and concurrency bounds, instead of assuming one static
allocation fits all requests.

Also retain validated immutable parameter preparation across fresh sequences.
The [linear-attention owner](../../src/inference/pipelines/text/linear-attention.js)
separates prepared parameters from dynamic recurrent state. Qualify reuse against
changed weights, precision, reset failures, cancellation, and device loss; never
reuse the previous conversation's recurrent state implicitly.

The [2026-10-17 frozen-runtime delivery experiment](../../reports/qwen-delivery-attribution-20261017/README.md)
attributes 16-token bursts to grouped readback before generator yield. Reducing
readback grouping to four tokens improves cadence but regresses 0.8B decode by
roughly 14% on M3 and 33% on AMD against restored controls; that candidate is
rejected. Readback wait includes GPU execution, not just transfer. Preserve queued
work and explicit resource ownership when evaluating progressive delivery; do not
trade away measured throughput merely to reduce the group size.

Expected benefit: less CPU orchestration, repeated conversion/readback, and
allocation. Graph-level layout, constant, fusion, and buffer optimization have
precedent in [oneDNN Graph Compiler](https://arxiv.org/abs/2301.01333);
its [official implementation documentation](https://uxlfoundation.github.io/oneDNN/v3.5/dev_guide_graph_compiler.html)
describes graph compilation and flexible fusion. That precedent supplies no
Doppler speedup estimate.

## 3 Exact prefix state checkpoints

Extend existing [KV state copying](../../src/inference/kv-cache/copy-state.js) and
[recurrent snapshots](../../src/inference/pipelines/text/linear-attention/state-clone.js)
into bounded, permission-scoped caches for identical system prompts, schemas,
document prefixes, and conversation roots. Cache tokenization and embeddings
where their own input/representation contracts match; cache all causal state
needed to resume, including KV pages and recurrent state for hybrid models.

Bind a checkpoint to the exact model, quantization, adapter, tokenizer, token
sequence, positions, attention/RoPE policy, state layout, runtime representation,
and kernel/execution identity. Cross-plan reuse needs explicit numerical
compatibility evidence. No similar-hidden-state reuse. A document after a
different preceding context generally cannot reuse its previous contextual KV
state. Semantic-search embeddings form a separate cache.

Measure lookup and restoration against recomputation. Bound retention and count
disk storage, resident memory, eviction, misses, transfers, and copy-on-write.
Test distinct prefixes, adapters, permission scopes, concurrent branches, and
cancellation. Cached preparation must not be disguised as uncached prefill in a
comparison.

[Prompt Cache](https://arxiv.org/html/2311.04934v2) reports 8× GPU and 60× CPU TTFT
improvements in its abstract. Its modular attention scheme restricts attention
to individual modules, an approximation described in section 3.1. It does not
prove arbitrary module reuse is exact relative to ordinary full-context causal
attention.

## 4 Verified continuation automata

Use tokenizer-bound suffix tries or compact n-gram structures derived from
accepted corpora and permitted prior outputs to draft multiple likely tokens.
Verify drafts with the target model; correct and resume target execution at the
first rejection. Reuse the existing
[speculative runtime](../../src/inference/pipelines/text/generator/speculative-runtime.js),
[sampling support](../../src/inference/speculative.js), and rollback machinery.
Avoid a second speculative execution path.

Acceptance and correction must be correct for the sampling policy. Greedy
verification and probabilistic verification need separate gates. Preserving the
sampling distribution does not necessarily reproduce the same seeded token
sequence. Include rejected drafts, rollback, verification work, datastore
residency, construction cost, and invalidation after tokenizer/model/adapter
changes. A draft's likelihood is never authority to emit an unverified token.

[CREST](https://arxiv.org/abs/2408.04678) reports matching REST's accepted-token
length with 10.6–13.5× less datastore storage on its tested workloads. This is
datastore compression, not model-weight compression or a guaranteed decode
speedup. Learned draft heads such as ReDrafter/EAGLE remain separate candidates
requiring training/loading and the same acceptance gates; prefer the compact
retrieval experiment first.

## 5 Conditional sparse and compressed assets

Keep these experiments separate and activate only for a selected bottleneck or
memory-constrained workload:

- **Indexed output head:** investigate only when vocabulary projection dominates.
  Retrieve candidates and score them precisely, but exact candidate scores do
  not prove omitted tokens irrelevant. The
  [indexed output-head paper](https://arxiv.org/abs/2608.27460) uses approximate
  HNSW retrieval and reports up to 82% greater batch-one CPU throughput in its
  tested scope. It does not establish exact argmax or full-softmax equivalence.
  Certified bounds plus full-projection fallback would be additional
  engineering, initially scoped to greedy selection. Count index storage,
  fallback frequency, and quality independently.
- **Expert placement and prefetch:** prepare hot/cold placement and transfer
  schedules for MoE workloads that need lower GPU residency. Moving exact
  weights without changing routing can preserve model semantics; dropping or
  approximating experts cannot. Count transfers, host memory, missed prefetches,
  latency, and cancellation. Lower GPU residency is not a universal speedup.
- **Immutable block deduplication:** share only byte-identical immutable blocks
  with correct ownership, aliasing, and lifetime checks. Count physical storage
  and transfer savings separately from logical model size.
- **Shared bases plus low-rank residuals:** these alter model representation and
  require fresh quality qualification and an explicit model identity. They are
  not exact caches. [DeltaLLM](https://arxiv.org/abs/2501.18596) is a post-training
  model compression method of this kind. Keep the original representation as an
  explicitly selected fallback where required; fallback alone does not establish
  quality of the compressed path.

## Kernel opportunities merged from earlier investigations

These are conditional implementation choices within the preparation order,
not a competing roadmap. Reprofile the selected model before reviving them.

| Opportunity | Qualification condition |
| --- | --- |
| Packed Q4_K projections and FFN fusion | Preserve numerical gates and packed residency; local operator gains must survive full-model timing. Resolve the recorded Gemma/Metal numerical failure before removing a guard. |
| F16 activation paths and fused gate/up activation | Bind a separate numerical execution plan; qualify prefill and decode, PLE, output softcap, and quality. Historical speedup estimates are not acceptance evidence. |
| RMSNorm, QKV, output-head and dispatch fusion | Target measured launch or memory work; count register pressure, grid expansion, scratch, and complete latency. |
| Attention layout, flash prefill and quantized KV | Revisit when attention or long-context memory dominates; short-context neutral results do not justify global selection. |
| PLE preparation and prefetch | Move repeated immutable preparation off the request path where profiling supports it; keep token-dependent work and ownership exact. |
| Persistent dispatch or megakernels | Require a measured dispatch bottleneck and a portable target contract; do not add architecture merely because another runtime reports gains. |
| Decode submission and readback overlap | Reduce actual token-delivery stalls while preserving CPU/GPU overlap. Measure streamed iterator delivery, cancellation, and complete decode, not just callback timestamps. |

Historical observations and failed trials remain in the
[Gemma INT4 PLE investigation](../perf-investigations/gemma-4-int4ple-loop.md),
[corrected Gemma browser diagnosis](gemma4-e2b-browser-decode-bottleneck.md), and
[kernel optimization guide](../developer-guides/16-kernel-performance-optimization.md).
Those receipts retain their original model, hardware, and date scope.

## Evidence required for each change

Use the existing [evidence loop](../evidence-loop.md) and
[benchmark methodology](../benchmark-methodology.md). Freeze exact model and
artifact identity, precision, target/browser/provider, prompts/tokens, sampling,
context, output budget, concurrency, warmup, and cache policy. Interleave paired
runs, retain failures and all declared samples, and report uncertainty.

Report independently:

- Cold opening/loading, preparation, and download time; identify which disk,
  browser, shader, OS, and model caches were actually empty.
- Warm opening, warm first-token latency, prefill, decode, token-delivery stall
  percentiles, waiting, and complete-answer latency.
- CPU work, submissions/readbacks/transfers, downloaded bytes, disk footprint,
  host RAM, peak and resident GPU memory, retained caches, and eviction costs.
- Numerical/reference gates, workload quality, token agreement in its tested
  scope, cancellation, recovery, device loss, and concurrent-request isolation.

Do not double-count overlapping GPU waits or describe host wall-time counters as
GPU kernel time. Keep profiling separate from comparison runs. Show wins,
regressions, and unmeasured metrics; a universal superiority claim requires
matched evidence for every claimed metric and GPU. Do not close earlier CI
failures while their required checks remain skipped.
