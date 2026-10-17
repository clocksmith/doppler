# LoRA partition execution on mac-host

The actual Qwen 3.5 0.8B Q4K model and pinned NER adapter ran on Apple M3 / native WebGPU. Whole-model and 12+12 layer partitions produced identical logits at three steps (248,320 logits each; unchanged tolerance 0.0001). Removing the adapter changed logits by a maximum 1.037746191, confirming actual adaptation. Cancellation, allocation-failure and cleanup checks passed.

[Raw result](mac-same-host.json) includes exact artifact hashes, prompt, layer ranges, GPU provider and comparisons. Base-weight allocations exclude adapter buffers; these are not complete residency estimates.

Reproduce with `DOPPLER_PARTITION_MODEL_DIR`, `DOPPLER_PARTITION_ADAPTER` (Reploid catalog adapter JSON) and `DOPPLER_PARTITION_ADAPTER_WEIGHTS` set, then `node tests/integration/partition-layers-physical.test.js`.

The earlier base manifest `sha256:edeb69dd65cbb26971abf773a95bf6dc219a82942c0951d65e1893d442b607c9` and current `sha256:18fe30e2c65c47bfdaa1f6714ced9fa90ed8c482a8d57b95cf0cdd4490e842e9` have identical tensor descriptors and shard identities. Only inference, metadata and artifactIdentity differ. This run qualifies the current manifest for this diagnostic; compatibility is not inferred from the model name.

Scope: same-device scalar partitions with one adapter. Every stage verifies the full adapter before retaining its own layer pairs and the global identity. This does not establish cross-host performance, browser support, paired arithmetic, adapter stacking, or selective adapter-byte downloads. Signed Capsule qualification remains separately enforced.

Component: text execution and client adapter loading. Intent: preserved. Boundary effect: adapter identity is part of the partition plan digest.
