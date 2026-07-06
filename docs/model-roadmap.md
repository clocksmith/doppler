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

## Tier 1: Qwen Publish Story

Goal: one clean story across small generation, retrieval embeddings, and rerank.

| Model | Status | Current state |
| --- | --- | --- |
| Qwen 3.5 0.8B | publish-story | Browser and Node runtime verification exist. Fair local generation comparisons exist; release-grade benchmark promotion is still the open claim gap. |
| Qwen 3 Embedding 0.6B | publish-story | Browser and Node runtime verification exist. Local comparable embedding evidence exists; release-claim promotion remains open. |
| Qwen 3 Reranker 0.6B | publish-story | Browser and Node runtime verification exist. External reranker competitor comparison is still missing. |

## Tier 2: Next Practical Wins

Goal: next small and medium models that can become clean benchmark stories.

| Model | Status | Current state |
| --- | --- | --- |
| Qwen 3.5 2B | benchmark-needed | Browser and Node runtime verification exist. The current compare lane is capability-only until a correctness-clean benchmark lane is promoted. |
| Gemma 4 E2B | benchmark-needed | Node runtime verification exists. Doppler will keep only the best-performing E2B implementation in the public story; browser and benchmark receipts remain open. |

## Tier 3A: Large Dense

Goal: scale the same evidence model to larger dense checkpoints with real catalog
targets.

| Model | Status | Current state |
| --- | --- | --- |
| Qwen 3.6 27B | benchmark-needed | Cataloged runtime evidence exists across the active 27B lanes, with browser support for the current web-demo lane. Fair competitor comparison is not yet promoted. |
| Gemma 4 12B | benchmark-needed | Node runtime verification exists. Hosted promotion, browser evidence, and benchmark receipts remain open. |

## Tier 3B: Experimental / Architecture Risk

Goal: keep unusual architectures separate from normal benchmark work so they do
not blur release claims.

| Model | Status | Current state |
| --- | --- | --- |
| DiffusionGemma 26B A4B | runtime-needed | Cataloged as experimental. Runtime verification is the next gate before benchmark claims. |
| Gemma 4 MoE | target-needed | Conversion-only work exists, but there is no concrete catalog target to claim yet. It should stay out of publish tiers until a catalog target exists. |

## Stretch

Goal: opportunistic larger targets after the core Qwen and Gemma stories are
green.

| Model | Status | Current state |
| --- | --- | --- |
| Gemma 4 31B | benchmark-needed | Node runtime verification exists. Browser/release evidence and current benchmark receipts remain open. |
| Larger Qwen 3.6/3.7-class dense model | target-needed | No concrete catalog/HF target is currently selected. Add it only after the target exists in catalog and has an evidence path. |
