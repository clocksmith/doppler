# Ordered Qwen partition component checks

These are physical AMD Radeon 8060S / RDNA 3 checks using the Node WebGPU provider
on Vulkan (Mesa 26.0.3, Node 22.22.1). Each run unloads the whole-model control
before opening the split stages, serializes the activation boundaries, and
compares all 248,320 logits after prefill and two incremental decode steps.
This is one GPU, not multiple physical computers or an independent source oracle.

| Artifact | Cuts | Stages | Logit rows | Maximum difference |
| --- | --- | --- | --- | --- |
| Qwen 3.8 27B Q4K | 32 | 2 | 3 | 0 |
| Qwen 3.6 35B A3B source Q4_0 | 10, 20, 30 | 4 | 3 | 0 |
| Qwen 3.8 Flash-Next Q4K | 16, 32 | 3 | 3 | 0 |

Each successful result records exact manifest identity, runtime configuration,
generation configuration, plan identity, selected tokens, full-logit hashes,
adapter/provider identity, and stage descriptors. All three have no reported
uncaptured GPU errors or cleanup failures. The comparison tolerance stayed
`1e-4`; all rows were byte-identical, rather than merely within that tolerance.
The prompt was `1, 2, 3, 4, 5, 6, 7, 8,` (23 tokens). No throughput result is claimed.

The explicit Flash-Next candidate uses F32 KV and routed experts with a 2 GiB
cache per prepared stage. Its boundary carries all four residual streams. The
35B control disables whole-token replay, uses one-token decoding, and explicitly
disables whole-model warm generation for both sides. Each run retains its complete
resolved overlay; these are experimental component configurations, not new recipes.

## Defects exposed and repaired

- Flash-Next's final stage omitted the phase when constructing the logits
  configuration. The root mixer rejected execution. The retained
  `stage-logits-phase-failure.json` has no completed comparison rows; it is not a
  passing run. Stage execution now passes `prefill` or `decode` explicitly.
- 35B's whole-model warm generation tried to execute unloaded layers while opening
  the first stage. `qwen35b-warm-generation-failure.json` retains that failure.
  Partition admission now rejects this policy before weight loading and names
  the required explicit `null`. The successful configuration disables it.

Earlier harness preparation attempts are not counted: one omitted a required
sampling field, one used an incorrect logits callback, and another redundantly
rehash-verified complete shards. The first interrupted GPU process did not prove
cleanup settlement. The subsequent cancellation did settle and unload; none of
those attempts is qualification evidence.

## Identity and reproduction

`receipt.json` hashes the retained files. `post-run-source-inventory.json` inventories
the isolated uncommitted source snapshot after the runs, rather than pretending
it was a clean release commit. The 27B and Flash-Next runs predate the final
warm-generation rejection guard; their otherwise identical program had no
configured warm generation. The snapshot archive hash and size are retained in
the receipt. The harness and runtime repairs are committed with this report.

To repeat with operator-owned weights, make a JSON configuration with schema
`doppler.ordered-split-physical/v1`, absolute `modelDirectory` and `outputPath`,
the matching result's `modelIdentity`, `runtimeConfig`, `generation`, and
`plan.splitLayers`/`activationDtype`/`activationPrecision` under `partition`.
Set `steps: 3`, the prompt above, and explicit limits, for example:

```json
{
  "maxTokens": 4,
  "maxPromptTokens": 48,
  "maxActivationBytes": 8388608,
  "maxOutputCharacters": 1024,
  "maxAttempts": 4,
  "maxConcurrentAttempts": 2,
  "maxPreparationMs": 120000
}
```

```sh
DOPPLER_ORDERED_SPLIT_CONFIG=/absolute/control.json \
  node tests/integration/ordered-split-physical.test.js --require-model
```

The component comparisons reused existing operator-verified artifacts and did
not perform another whole-shard rehash. That policy is explicit in each result;
the manifest bytes are checked before GPU allocation. They do not establish a
new artifact-integrity receipt.

## Remaining boundaries

Flash-Next's independent long-context numerical gate remains open. These results
do not qualify Metal, mixed-device numerics, peer transport, signed N-stage
Capsules, public availability, active-state migration, LoRA, or speculative
execution. The existing whole-model provider remained active with its original
process identity, and the original GPU memory limit was unchanged.

Component: `doppler.runtime-source.inference.pipelines.text`,
`doppler.runtime-source.config`, `doppler.tests`.
Intent: preserved. Boundary effects: ordered plans can carry source-declared
hyper-connection streams and execute scalar MoE stages; no catalog promotion.
