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
| Qwen 3.5 0.8B | publish-story | Browser and Node runtime verification exist. The hosted browser/Vulkan p512 comparison is release-claimable with exact output and promotable throughput cadence. |
| Qwen 3 Embedding 0.6B | publish-story | Browser and Node runtime verification exist. Fresh hosted browser/Vulkan evidence is release-claimable and Doppler leads steady-state embedding latency and throughput. |
| Qwen 3 Reranker 0.6B | publish-story | Browser and Node runtime verification exist. Fresh 15-run hosted browser/Vulkan evidence is release-claimable and Doppler leads rerank latency, throughput, and semantic pair accuracy. |

## Tier 2: Next Practical Wins

Goal: next small and medium models that can become clean benchmark stories.

| Model | Status | Current state |
| --- | --- | --- |
| Qwen 3.5 2B | publish-story | Browser and Node runtime verification exist. Fresh browser/Vulkan local comparisons cover p064, p256, and p512 with exact output match and promotable throughput cadence; hosted release-grade promotion remains open. |
| Gemma 4 E2B | publish-story | Node runtime verification exists. INT4-PLE has fresh browser/Vulkan p064 parity-section evidence that is local-comparable under the explicit product-format output policy; exact token parity is not claimed. The throughput-cadence section is still tuning evidence, and the plain Q4K local artifact needs a refreshed manifest before fair compare work resumes. |

## P4 Supported Legacy Tier

Goal: preserve older supported models with strong receipts without letting them
compete with the current Tier 1 and Tier 2 publish work.

| Model | Status | Current state |
| --- | --- | --- |
| Gemma 3 270M | verified | Browser, Node, and Bun local compare receipts exist with exact output match across p064, p256, and p512. Keep this as supported legacy evidence unless it becomes part of a new publish story. |

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

## Experimental Tier: Translation Specialists

Goal: surface narrow, evidence-backed language specialists without implying
Tier 1 multilingual support or general-purpose generation quality.

| Model | Status | Current state |
| --- | --- | --- |
| TranslateGemma 1B EN/ES NativeKD2 student | verified | The Q4K artifact is deterministic on the 128-row WMT13 receipt at 31.9149 BLEU / 58.2124 chrF, is 67.6% smaller than the cataloged 4B teacher, and passed directly from its pinned hosted revision in browser/WebGPU. Demo and hosted visibility carry explicit experimental, EN/ES-only, cross-runtime, and Radeon/Vulkan evidence boundaries. |

## Stretch

Goal: opportunistic larger targets after the core Qwen and Gemma stories are
green.

| Model | Status | Current state |
| --- | --- | --- |
| Gemma 4 31B | benchmark-needed | Node runtime verification exists. Browser/release evidence and current benchmark receipts remain open. |
| Larger Qwen 3.6/3.7-class dense model | target-needed | No concrete catalog/HF target is currently selected. Add it only after the target exists in catalog and has an evidence path. |
