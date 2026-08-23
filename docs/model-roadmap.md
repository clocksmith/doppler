# Model Roadmap

This roadmap is the editorial model priority list for Doppler. It is separate
from the generated support matrix and release matrix. Doppler chooses the best
artifact and runtime implementation for each model from committed verification
and benchmark evidence.

For exact evidence, use:

- [Model support matrix](./model-support-matrix.md): runtime verification status.
- [Model support inventory](./model-support-inventory.md): gate status and next evidence gaps.
- [Release matrix](./release-matrix.md): benchmark and competitor evidence.

## Status Key

- `publish-story`: working model story with remaining release-evidence cleanup.
- `verified`: runtime receipts exist on at least one public surface.
- `benchmark-needed`: runtime support exists, but fair competitor evidence is incomplete.
- `runtime-needed`: catalog or conversion work exists, but runtime verification is not green.
- `target-needed`: no concrete catalog target is ready to claim.

## Tier 1: Ambient Models (Small & Ubiquitous Generation)

Goal: instant, private generation, JSON extraction, and UI intelligence on standard laptops and browser tabs.

| Model | Status | Current state |
| --- | --- | --- |
| Qwen 3.5 0.8B | publish-story | Browser and Node runtime verification exist. The hosted browser/Vulkan p512 comparison is release-claimable with exact output and promotable throughput cadence. |
| Qwen 3.5 2B | publish-story | Browser and Node runtime verification exist. Local comparisons cover p064, p256, and p512 with exact output match and promotable throughput cadence. |
| Gemma 3 270M | verified | Browser, Node, and Bun local compare receipts exist with exact output match across p064, p256, and p512. Standard lightweight ambient baseline. |
| Gemma 3 1B | verified | Browser and Node runtime verification exist. Clean baseline for 1B-class text generation and assistant tasks. |
| Gemma 4 E2B | publish-story | Node runtime verification exists. INT4-PLE has browser/Vulkan parity-section evidence under the product-format output policy. |

## Tier 2: Workstation Agents (The Flagship Frontier Pair)

Goal: autonomous coding, tool use, document analysis, and multi-step reasoning on 16GB–128GB unified APUs and discrete GPUs.

| Model | Status | Current state |
| --- | --- | --- |
| Qwen 3.8 27B | benchmark-needed | The lineage-acceleration campaign produced a development-signed text Pack with 128/128 exact tokens on physical AMD Node WebGPU, complete artifact verification, prefill identity binding, and immutable TargetPlan identity. Browser, application, comparative benchmark, and production-signing gates remain open. |
| [Meta Muse Glimmer 30B](programs/glimmer-architectural-generalization.md) | source-truth-only | Text, vision, and projector topology plus pinned reference formulas are represented with zero unresolved text operational facts. Generic Runtime mechanisms now cover explicit query scaling, per-layer NoPE, and a separate attention-gate projection, but Forge has not admitted their bindings and still rejects local attention, component norms, postnorm, embeddings, and final logits; no execution support is claimed. |
| Gemma 4 12B / 31B | benchmark-needed | Node runtime verification exists for 12B/31B lanes. Awaiting workstation pack qualification. |

## Tier 3: Retrieval Specialists (Embeddings & Reranking)

Goal: high-throughput dense embeddings and cross-encoder semantic reranking for local RAG pipelines.

| Model | Status | Current state |
| --- | --- | --- |
| Google EmbeddingGemma 300M | verified | Browser and Node runtime verification exist with complete RDRR manifest and quickstart catalog exposure. |
| Qwen 3 Embedding 0.6B | publish-story | Browser and Node runtime verification exist. Fresh hosted browser/Vulkan evidence is release-claimable and leads steady-state latency. |
| Qwen 3 Reranker 0.6B | publish-story | Browser and Node runtime verification exist. Fresh 15-run hosted browser/Vulkan evidence is release-claimable and leads rerank latency and accuracy. |

## Tier 4: Biological Sequence Specialists (First-Class for Reploid)

Goal: first-class protein and nucleotide sequence modeling for in-browser scientific research, mutation screening, and zero-egress discovery in Reploid.

| Model | Status | Current state |
| --- | --- | --- |
| ESM-2 35M (`esm2-t12-35m-ur50d`) | verified | Hosted Node/WebGPU sequence parity passed. Provides zero-egress residue and sequence embeddings in browser/Node sandboxes. |
| ESMC 300M (`esmc-300m`) | verified | Hosted Node/WebGPU sequence parity passed. High-capacity protein representation model for biological decision memory. |
| Nucleotide Transformer 50M (`nucleotide-transformer-v2-50m`) | verified | Multi-species DNA/RNA genomic sequence encoder verified under WebGPU qualification receipts. |

## Experimental & Translation Tier

Goal: narrow, evidence-backed translation specialists and research architecture candidates.

| Model | Status | Current state |
| --- | --- | --- |
| TranslateGemma 1B EN/ES NativeKD2 student | verified | Deterministic on 128-row WMT13 receipt at 31.9149 BLEU / 58.2124 chrF, 67.6% smaller than 4B teacher in browser/WebGPU. |
| DiffusionGemma 26B A4B | runtime-needed | Quarantined research lane. Runtime verification is the next gate before benchmark claims. |
| Gemma 4 MoE | target-needed | Conversion-only research work. |
