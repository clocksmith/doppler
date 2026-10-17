# Qwen recurrent execution and resident-state optimization

Apple M3, macOS 26.6.2, Node 22.20.0/native WebGPU and Chrome 155.0.8059.80.
Base source: `ed69d0ce`. Local, already-provisioned Qwen3.5 Q4K models only.
These are scoped tuning results, not competitor comparisons, model-quality
qualification, or release promotion. No dtype, sampling, or acceptance threshold
was changed. Signed model artifacts were not rewritten.

## Retained changes

| Change and measured scope | Baseline | Candidate | Evidence |
| --- | ---: | ---: | --- |
| 0.8B input-partition recurrent clone, median wall | 6.587 ms | 1.926 ms (3.42x) | [Physical receipt](mac-recurrent-clone.json.gz) |
| 2B input-partition recurrent clone, median wall | 6.450 ms | 1.822 ms (3.54x) | [Physical receipt](mac-recurrent-clone-2b.json.gz) |
| 30,000 fresh-wrapper uniform preparations, median CPU wall | 386.605 ms | 12.685 ms (30.48x) | [Paired receipt](mac-uniform-validation.json) |
| Recurrent WGSL, f32 inputs, 128 tokens, median GPU time | 10.420 ms | 5.767 ms | [Paired operator receipt](mac-recurrent-state-reuse.json.gz) |
| Recurrent WGSL, f16 inputs, 128 tokens, median GPU time | 10.945 ms | 6.128 ms | [Paired operator receipt](mac-recurrent-state-reuse.json.gz) |

The shader's median **paired** speedups at 128 tokens were 1.64x and 1.68x;
candidate GPU time was lower in all 12 pairs for each dtype. At 22 tokens the
paired speedups were 1.58x and 1.61x. Single-token timings were too coarse/noisy
to establish the magnitude of a decode-operator improvement. Raw samples and
min/max ranges are retained in [summary.json](summary.json).

The shader consumes each decayed state element while accumulating K·state, then
consumes each updated element while accumulating Q·state. This replaces four
storage-reading loops with two, preserving increasing-K accumulation order,
rounding operations, compensated arithmetic, bindings, and workgroup geometry.
The f32/f16 and silu/sigmoid recurrent files are kept aligned. Fused decode
shaders are unchanged. Registry digests and the two affected Qwen3.8 source
recipes were regenerated; that does not qualify those 27B models.

Uniform validation now caches by deeply immutable layout plus kernel identity.
Dispatch wrappers sharing that layout no longer repeatedly canonicalize it.
Mutable nested fields and mismatched kernel/layout pairs still fail before GPU
allocation. The measured improvement is uniform preparation, not generation
throughput or all command recording.

Resident partition clones copy GPU-authoritative recurrent buffers directly,
instead of reading them to CPU and uploading them again. Each child owns its
buffers independently. CPU-only states still upload explicitly; unknown queue
completion quarantines allocations. CPU export continues to synchronize its
mirrors. KV cloning itself is unchanged. Both physical model checks cover nine
recurrent layers and 10,321,920 dynamic-state bytes, not a whole-model/network
state transfer. Each closes the parent and checks 128 forced-token continuation
activation outputs exactly, including the production composite KV/recurrent owner.

## Method and correctness

- Clone: 2 warmup pairs, 12 measured pairs, alternating order; every static and
  dynamic GPU buffer compared byte-for-byte outside the timed interval.
- Uniform preparation: 3 warmup pairs, 12 measured pairs, alternating order;
  10,000 iterations across three configured kernel layouts. Uniform bytes and
  byte counts match exactly. This is a CPU microbenchmark without GPU allocation.
- Shader: 3 warmup pairs, 12 measured pairs, alternating order, 16 heads,
  128-wide K/V, 128 threads, scalar f32 state. Separate fixed initial state for
  each lane; 1, 22, and 128 tokens with f32/f16 inputs. GPU timestamps exclude
  compilation, uploads, and readbacks. Every state/output byte matches after
  every pair. Compilation errors and nonfinite values fail the run.
- Existing physical CPU-reference regression passes for small/full-workgroup
  shapes, multistep execution, both gates, both input dtypes, and scalar/two-f32
  state. Existing tolerances were not changed.
- All saved 0.8B generation runs, including final Chrome and Node runs, produce
  the same 128 token IDs. This is one deterministic prompt, not general quality
  evidence. Final runs used the combined retained changes.

The source snapshots are file-to-SHA256 maps:
[initial clone/baseline tree](source-snapshot.json.gz),
[uniform-only candidate](uniform-source-snapshot.json.gz), and
[final candidate](optimized-source-snapshot.json.gz). The initial snapshot has
the new clone implementation but the original uniform writer and shaders;
ordinary generation does not call the partition-clone path. Clone receipts
also bind the raw local manifest and every model shard/tokenizer hash. Operator
receipts bind each compiled shader source independently.

## Whole-model diagnosis and inconclusive results

The original Chrome run used warm OPFS storage: 48.42 tokens/s, 170.6 ms prefill,
172.0 ms first token, 1102.6 ms model initialization. The final Chrome run was
49.91 tokens/s, 174.7 ms prefill, 176.2 ms first token, 689.8 ms initialization.
These separated observations **do not establish an end-to-end speedup**.

The uniform-only Node sequence was baseline/candidate/candidate/baseline/baseline/
candidate, with respective rates 42.57/39.26/24.29/19.78/21.12/24.45 tokens/s.
The host was shared with active public contributor processes; substantial
time-dependent drift prevents attributing those differences to the change.
The final combined Node run reached 47.84 tokens/s, also not a controlled gain.
All runs, including stalls, are retained; no favorable subset was selected.

The [instrumented Chrome receipt](mac-generation-profile.json.gz) resolves
f32 activations, f16 contiguous KV, and batched GPU decode. GPU decode timestamps
were approximately 519 ms versus 2784 ms decode wall. The noninstrumented
baseline recorded 39,751 operations over 127 decode steps (313/token), with
about 756 ms recording and 1906 ms effective submit/readback wait. Those waits
overlap and must not be added together. GPU-only speedups cannot remove that
entire wall. The [CPU profile](baseline-node.cpuprofile.gz) localized repeated
uniform-layout canonicalization as well as buffer/bind-group creation costs.

Combining Q/K norm reductions passed exact operator comparison but gave only
approximately 0.99–1.02x median paired ratios in the longer cases. It was
reverted. The [receipt](mac-recurrent-reduction.json.gz) and
[rejected patch](rejected-joint-reduction.patch.gz) remain available. An initial
probe used nonexistent registry variant `f16` and stopped; the complete run
corrected this to `default_f16` without changing the candidate. The existing
global negative-result schema requires generated-text/token receipts and decode
metrics, so this operator-only rejection is retained here without inventing
those fields. Retry only with a changed implementation, backend, or shape.

## Remaining work and next commands

| Area | Evidence/gap | Next action |
| --- | --- | --- |
| Prefill | Recurrent operator improved; no whole-prefill gain established | Repeat `bench` with the same generation config and observer enabled in alternating source trees, then longer prompts |
| Decode | Host recording and submit/readback dominate; fused recurrent path unchanged | Profile bind-group/buffer creation and dependency-safe submission/readback batching; test state reuse separately in the fused shader |
| Warm start | Warm OPFS load phases retained; no controlled startup change | Repeat fresh pipeline creation with fixed OPFS and explicit tokenizer/pipeline cache state |
| Cold start | Empty-storage/network/compiler-cold startup not measured | Prepare an isolated browser profile, retain byte/load/compile counters, then compare the same artifact and route; do not clear shared caches |
| KV/prefill transfer | Same-device composite clone continuation passes; KV copy and activation transport unchanged | Measure live-prefix bytes versus reserved capacity; preserve sliding/ring/paged semantics before any prefix-only copy change |
| Cross-device transfer | No network handoff/codec measurement | Bind source/recipient artifact identity, capture bytes and transfer/restore time, then require exact continuation before tuning overlap/compression |
| Other models/devices | 2B input-partition clone checked; no 2B full-generation or AMD/27B qualification | Run the named lane's baseline and parity workload before promoting these local findings |

Reproduction from the repository root (model paths are explicit in the configs):

```sh
node src/cli/doppler-cli.js bench --surface browser --config reports/qwen-optimization-20261017/generation-config.json --json
node src/cli/doppler-cli.js bench --surface node --config reports/qwen-optimization-20261017/generation-config.json --json
node tests/integration/recurrent-clone-profile.js reports/qwen-optimization-20261017/clone-config.json
node tests/integration/recurrent-clone-profile.js reports/qwen-optimization-20261017/clone-2b-config.json
node tests/integration/uniform-validation-profile.js reports/qwen-optimization-20261017/uniform-config.json /tmp/uniform-profile.json
node tests/integration/recurrent-kernel-profile.js reports/qwen-optimization-20261017/state-reuse-config.json /tmp/recurrent-profile.json
```

The paired configs name immutable `.tmp` source snapshots. On another checkout,
materialize the baseline from `ed69d0ce` and the retained candidate from this
commit into separate directories, update those explicit config paths, and
verify the relevant recorded source hashes before comparison. The rejected
candidate is reconstructed by applying its saved patch to the baseline. These
opt-in profiling scripts are not automatic unit tests. Generation uses two
warmups and three timed runs; exact resolved settings are in the raw envelopes.
For Node, use the top-level envelope surface/provider identity: the legacy
nested reference transcript labels itself `browser-webgpu` even on Node.

The [routing audit](qwen-routing-audit.json) also records pre-existing local
manifest gather/matmul digest drift. These development runs do not establish
signed-Capsule freshness or release eligibility.

## Acceptance and component handoff

Passed: strict source, declaration and project type checks; source style/unused
checks; recurrent clone ownership/failure tests; composite partition clone,
resident target/canonical execution, recurrent layout/upload cleanup tests;
408 uniform variants/3544 fields; uniform generator tests; physical recurrent
CPU references; gate closure, registry reachability, kernel/source-recipe digest
checks, and compensated-helper freshness.

The architecture check still fails on unchanged
`src/gpu/kernels/attention/executors.js`: 971 lines versus its existing reviewed
960-line ceiling. This was present at `ed69d0ce`; no ceiling was increased.
Full suites, multi-device qualification and vendor fairness gates were not run.

Component: `doppler.runtime-source.gpu.kernels`,
`doppler.runtime-source.inference.pipelines.text`,
`doppler.runtime-source.config`, `doppler.tests`.
Intent: preserved. Acceptance evidence: the commands and receipts above,
[summary](summary.json), and [artifact hashes](artifacts.json).
Boundary effects: no authority/API/dtype changes; partition attempts now own
GPU-resident recurrent clones, and config recipes pin the updated WGSL digests.
