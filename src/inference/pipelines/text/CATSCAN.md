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

## Contracts

- Input: [Pipeline facade](../text.js), resolved runtime session, loaded weights, tokenizer, and execution graph.
- Output: Tokens, text, embeddings, rankings, cache state, and phase-specific runtime statistics.

## Invariants

- Config resolution order and source attribution remain explicit.
- Prefill and decode choices are resolved before adapter execution.
- Load, reset, unload, abort, and failure paths release owned resources.

## Acceptance

- Text generation parity, dtype, attention, resource, and workload tests pass.
- Evidence: [text inference tests](../../../../tests/inference).

## Non-goals

- Silently treating an exported method as a qualified product workload.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
