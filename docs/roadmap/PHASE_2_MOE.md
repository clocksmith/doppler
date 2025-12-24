# Phase 2: MoE Efficiency

**Status:** Advanced (infrastructure complete, model validation needed)
**Prerequisites:** Phase 1 (buffer reuse, async pipeline)
**Goal:** Run Mixture-of-Experts models efficiently with expert paging.

---

## Milestones

- [x] GPU-native MoE routing working ✅
- [x] GPT-OSS 20B experimental ⏳ Partial
- [ ] Mixtral 8x7B E2E with expert paging (P0)
- [ ] Expert cache hit rate > 80% (P1)

---

## Work Items

### 2.1 Core MoE Infrastructure

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| GPU-native routing (softmax+topk) | P0 | ✅ Done | Custom WGSL |
| Expert FFN execution | P0 | ✅ Done | Per-expert matmul |
| Scatter-add combination | P0 | ✅ Done | Custom WGSL kernel |
| MoE router with load balancing | P0 | ✅ Done | `inference/moe-router.ts` |

### 2.2 Expert Paging (OPFS)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Expert-level shard granularity | P0 | ✅ Done | `expertShardMap` in manifest |
| `loadExpert(layerIdx, expertIdx)` API | P0 | ✅ Done | `loader/doppler-loader.ts` |
| Expert LRU cache in VRAM | P0 | ✅ Done | `loader/expert-cache.ts` |
| Prefetch next-layer experts | P1 | ✅ Done | `prefetchExperts()` method |
| Expert hit rate tracking | P1 | ✅ Done | `CacheStats` interface |
| Cache auto-tuning | P1 | ✅ Done | `autoTune()` detects VRAM |
| Smart eviction (in-use protection) | P1 | ✅ Done | `markInUse()` / `markNotInUse()` |
| Shared expert pinning | P1 | ✅ Done | `pinSharedExperts()` for DeepSeek |

### 2.2b Storage Optimization (Quick Wins)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Expand shard cache (2 → 8) | P0 | ✅ Done | Dynamic based on MoE config |
| Dynamic cache size based on model | P1 | ✅ Done | `numExpertsPerToken * 2 + 1`, capped at 16 |

### 2.3 Model Validation

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| GPT-OSS 20B E2E | P0 | ⏳ Partial | Router works, experts loading |
| Convert Mixtral 8x7B to RDRR | P0 | ⬜ TODO | |
| Mixtral 8x7B E2E | P0 | ⬜ TODO | With expert swapping |
| Benchmark MoE decode throughput | P0 | ⬜ TODO | |
| Benchmark vs WebLLM Mixtral | P1 | ⬜ TODO | tok/s, TTFT, VRAM |

---

## Architecture

### Expert Paging Strategy

```
Local VRAM:   Router + active experts (top 25%)
OPFS cache:   Recently used experts (next 50%)
P2P swarm:    Rare experts (bottom 25%) → Phase 4
```

### MoE Sparsity

| Model | Total Experts | Active/Token | Total Size | Active Size |
|-------|---------------|--------------|------------|-------------|
| Phi-mini-MoE | 16 | 2 | ~15GB | ~2.5GB |
| Mixtral 8x7B | 8 | 2 | ~90GB | ~24GB |
| GPT-OSS 20B | 32 | 4 | ~40GB | ~8GB |
| Qwen3-30B-A3B | 128 | 8 | ~60GB | ~9GB |
| Qwen3-235B-A22B | 128 | 8 | ~470GB | ~51GB |
| Kimi K2 | 385 (384+1 shared) | 8 | ~2TB | ~82GB |

**Key insight:** Only ~6-25% of experts active per token. Page the rest.

---

## Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Mixtral 8x7B E2E | Working | - | ⬜ |
| Expert page latency (OPFS) | < 100ms | - | ⬜ |
| Expert cache hit rate | > 80% | - | ⬜ |
| VRAM usage (2 experts) | < 8GB | - | ⬜ |

---

## Key Files

| File | Purpose |
|------|---------|
| `inference/moe-router.ts` | Router implementation |
| `gpu/kernels/moe_gather.wgsl` | Expert gathering |
| `gpu/kernels/scatter_add.wgsl` | Output combination |
| `loader/doppler-loader.ts` | Expert loading API, prefetching |
| `loader/expert-cache.ts` | LRU cache with smart eviction |
| `storage/rdrr-format.ts` | `MoEConfig` with expert mapping |
| `tools/rdrr-writer.ts` | Expert tensor detection during conversion |

---

## Dependencies

- **Phase 1:** Buffer reuse (reduces memory pressure during expert swapping)
- **Phase 1:** Async pipeline (enables expert prefetch overlap)

---

## Next Phase

[Phase 3: Scale](PHASE_3_SCALE.md) - Extends expert paging to unified memory and larger models.

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

