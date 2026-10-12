# Compact hybrid KV: physical partition controls

The allocation-only archive `7449618b…` passed the existing short physical
partition and resident controls on x128, Node 22.22.1 and webgpu 0.4.0/Vulkan.
The exact archive, source-test hashes, model identity, provider observations,
reports and log hashes are in `receipt.json`. No accepted dependency changed.

Three full-vocabulary steps matched unsplit Doppler bit for bit (`maxDiff=0`).
Partition ranges were `[0,11]` and `[12,23]`; each allocated three KV layers
(786,432 bytes at context 128). The full cache allocated six. Seven validation,
failure and cancellation controls passed. The existing resident test passed
two interleaved attempts, full-logit/text parity, isolated recurrent cleanup,
short completion, retired-attempt rejection and cancellation. Both passing
processes exited zero after their pipeline/device/provider cleanup.

This is same-device allocation/indexing and lifecycle evidence. It is not an
independent accuracy reference, useful-answer qualification, two-machine run,
signed Capsule opening, or capacity test under per-executor memory limits.

## Retained test failures and repairs

`partition-layers.log` shows the original harness reusing recurrent state after
an injected recording failure. The required restart guard correctly rejected
it before the next injected failure. The canonical test now resets each failed
attempt and includes its retained recurrent state in cleanup accounting.
`partition-layers-fixed.log` and `partition-layers-settled.log` retain intermediate
test failures: fresh prefill creates retained state and submits initialization
work. The final one-submit cancellation control explicitly prepares a valid
incremental state first; its assertion remains exactly one submission.

The current resident test cannot run unchanged against the accepted-base API:
`resident-partition.log` records its newer sampling helper import failure.
An older test at `dd17c76d…` then reached a newer typed-error expectation,
retained in `resident-partition-archive-era.log`. The unchanged test from
`4fba376633a570b27ba29cd2abc675219c082918`, matching the installed API, passed.
The newer output-error-code assertion remains unqualified for this archive;
the passing test did not exercise that later-added check.

## Reproduction

Install the bound archive and `webgpu@0.4.0` in an isolated consumer. In a
separate test directory, link `src` to that consumer's `doppler-gpu/src` and
`node_modules` to the consumer's dependencies. Copy the canonical partition
test, `tests/helpers/gpu-probe.js`, and a package.json with `type: module`.
Extract the exact resident test with:

```sh
git show 4fba376633a570b27ba29cd2abc675219c082918:tests/integration/resident-partition-physical.test.js > tests/integration/resident-partition-accepted-api.test.js
```

Run each test separately with:

```sh
DOPPLER_NODE_WEBGPU_MODULE=webgpu DOPPLER_NODE_WEBGPU_PROVIDER_ARGS=backend=vulkan DOPPLER_PARTITION_MODEL_DIR=/home/x/deco/doppler/models/local/qwen-3-5-0-8b-q4k-ehaf16 node tests/integration/partition-layers-physical.test.js
DOPPLER_NODE_WEBGPU_MODULE=webgpu DOPPLER_NODE_WEBGPU_PROVIDER_ARGS=backend=vulkan DOPPLER_PARTITION_MODEL_DIR=/home/x/deco/doppler/models/local/qwen-3-5-0-8b-q4k-ehaf16 node tests/integration/resident-partition-accepted-api.test.js
```

Focused CPU controls also passed: `partition-kv-cache.test.js` and
`resident-partition-physical-config.test.js`. The test change has no runtime
effect. Earlier failures remain beside the passing logs.
