# CATSCAN: Text Pipeline

Component: `doppler.runtime-source.inference.pipelines.text`

Parent: [Pipeline Registry](../CATSCAN.md)

## Target

Execute manifest-declared transformer generation, embedding, reranking, and multimodal decoding.

## Authority

- Owns config resolution, loading coordination, prefill/decode, layers, sampling, and lifecycle.
- Does not own artifact facts, application prompt policy, or undeclared modalities.

## Scope

- State, generators, layers, attention, FFN, logits, and declared encoder bridges.
- Partitions preserve model indices, precision, and cache policy; exclude peer transport/grants.

## Contracts

- Input: [Facade](../text.js), resolved session, weights, tokenizer, execution graph.
- Output: Tokens, text, embeddings, rankings, caches, phase statistics.

## Invariants

- Research channel-gain/rank-two rotation share isolated dense split-f32 SwiGLU state; reset clears requests. Frozen opaque handles own copied GPU buffers/immutable parameters. Rotation requires exact template-free complements. Base-cache attaches after base prefill; first logits stay base. Re-prefill transforms its entire prefill. Neither implies quality. Final-layer replay retains ordinary f32 post-attention/residual tensors, reruns standard final FFN/output head without attention/cache writes, requires attached capture/original model; reset/detach invalidates it. Reject sandwich norms, per-layer input, scaling, MoE, hyperconnections.

- Explicit config precedence/attribution; resolve prefill/decode before adapters.
- Release resources on load/reset/unload/abort/failure.
- Allocate assigned caches only; reject unsupported partition dependencies before materialization. Local agreement never qualifies resident/distributed inference.

## Acceptance

- Evidence: [parity/dtype/attention/resource/workload tests](../../../../tests/inference).
- Partitions: [physical comparison](../../../../tests/integration/partition-layers-physical.test.js), [retained diagnostic](../../../../reports/resident-partitions/20260927/README.md).

## Non-goals

- Exported methods imply no qualification.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
