# CATSCAN: Text Pipeline

Component: `doppler.runtime-source.inference.pipelines.text`

Parent: [Pipeline Registry](../CATSCAN.md)

## Target

Execute manifest-declared transformer generation, embedding, reranking, and declared multimodal decoder workloads.

## Authority

- Owns text-pipeline config resolution, model loading coordination, prefill/decode plans, layer execution, sampling, and pipeline lifecycle.
- Does not own source artifact facts, application prompt policy, or undeclared modality support.

## Scope

- Transformer pipeline state, generators, layers, attention, FFN, logits, and declared encoder bridges.
- Bounded layer partitions preserve original model indices, precision, and cache-layout policy; partition execution owns no peer transport or grant authority.

## Contracts

- Input: [Pipeline facade](../text.js), resolved runtime session, loaded weights, tokenizer, and execution graph.
- Output: Tokens, text, embeddings, rankings, cache state, and phase-specific runtime statistics.

## Invariants

- Config resolution order and source attribution remain explicit.
- Prefill and decode choices are resolved before adapter execution.
- Load, reset, unload, abort, and failure paths release owned resources.
- Partial pipelines allocate only assigned layer caches. Unsupported cross-partition dependencies fail before weight materialization. Local numerical agreement does not qualify a resident session or distributed inference.

## Acceptance

- Text generation parity, dtype, attention, resource, and workload tests pass.
- Evidence: [text inference tests](../../../../tests/inference).
- Partition evidence: [physical layer comparison](../../../../tests/integration/partition-layers-physical.test.js) and [retained local diagnostic](../../../../reports/resident-partitions/20260927/README.md).

## Non-goals

- Silently treating an exported method as a qualified product workload.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
