# Operator attribution: bounded diagnostic, no optimization selected

This extends the existing command recorder. Its timestamp intervals now retain
operation labels, compilation-owned shader identity, constants, dispatch shape,
invocation count, and measured/unmeasured pass counts. Pipelines compiled outside
the cache retain unknown provenance; labels do not substitute for shader hashes.
Existing aggregate timing callers remain compatible. Unprofiled dispatch batching
is unchanged. No kernel, precision, allocation, or model policy changed here.

The [contract](contract.json) freezes the source snapshot, provisioned Qwen3.5-2B
Q4_K successor manifest, prior scratch-control configuration and expected output.
The M3 Metal probe used Chrome 155.0.8059.80, a five-token prompt, 32 generated
tokens, and three requests: ordinary, profiled, ordinary repeat. All three
preserved retained control token IDs and text; reset/reuse/unload completed.
This is not a new lifecycle qualification or an answer-quality result.

| Phase | Timestamped / recorded passes | Unknown shader among timestamped passes | Sum of measured intervals |
| --- | ---: | ---: | ---: |
| Prefill | 337 / 337 | 36 | 209.80 ms |
| Decode | 4,096 / 6,293 | 364 | 807.48 ms |

Decode exhausted the existing recorder query capacity. Its 2,197 unmeasured
passes remain unaccounted for. Indirect dispatch dimensions are unknown, and
raw passes cannot establish invocation counts. Copies, other queues, and gaps
outside compute passes are excluded. Full timestamp coverage does not imply
complete shader attribution.

The [summary](summary.json) groups intervals by phase, pipeline identity and
dispatch shape, retaining contributing operation labels and counts. In measured
decode work, fused FFN accounts for 242.32 ms across 484 invocations; one fused
matmul shape accounts for 188.93 ms across 1,088; output-head argmax phase 1
accounts for 162.12 ms across 20. Prefill's two dominant wide-tile matmul shapes
account for 82.36 and 80.07 ms. These totals identify substantial work, not waste.

**No optimization selected.** The probe does not establish avoidable work or
explain the earlier 587 ms completion wait quantitatively. Profiling splits
compute passes, and cold/warm effects and shared-machine interference differ.
Opening was 3,210.6 ms; the three requests took 2,216.1, 2,114.0 and 1,317.7 ms.
These are diagnostic observations, not a matched speed comparison. Completion
and readback waits overlap and must not be added. No performance candidate or
new acceptance campaign follows from these observations alone.

The [raw receipt](receipt.json.gz) retains every interval, ordinary execution
statistics, source hashes, model/config identities and cleanup. The frozen
source snapshot prevented concurrent workspace edits from changing this run.
Reproduction requires those exact source bytes and the already provisioned
model; the probe verifies their hashes and refuses an existing destination:

```sh
node reports/qwen-operator-attribution-20261018/probe.js reports/qwen-operator-attribution-20261018/contract.json /path/to/new-receipt.json
node reports/qwen-operator-attribution-20261018/summarize.js
```

The contract contains local snapshot/model paths. Relocation requires a separately
identified contract preserving source/model hashes, not overwriting this receipt.
No downloads, public service changes or Doe work were needed.

Focused evidence: command-recorder cleanup/coverage tests; nine pipeline identity
and hot-path tests; source style, architecture and dependency checks; declaration
type checking; this three-request physical diagnostic. No full CI or release
promotion is claimed.

Component: GPU command recording and verification. Intent preserved. Boundary
effects: observation metadata only; Reploid and model computation are unchanged.
