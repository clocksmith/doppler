# MoE (Mixture of Experts) Internals

Technical deep-dive on MoE routing, expert paging, and sparsity exploitation.

---

## MoE Sparsity

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

## Expert Paging Strategy

```
Local VRAM:   Router + active experts (top 25%)
OPFS cache:   Recently used experts (next 50%)
P2P swarm:    Rare experts (bottom 25%) → Phase 4
```

### Expert Cache Implementation

| Feature | Status |
|---------|--------|
| Expert LRU cache in VRAM | Done |
| Expert hit rate tracking | Done (`CacheStats` interface) |
| Cache auto-tuning | Done (`autoTune()` detects VRAM) |
| Smart eviction (in-use protection) | Done (`markInUse()` / `markNotInUse()`) |
| Shared expert pinning | Done (`pinSharedExperts()` for DeepSeek) |
| Prefetch next-layer experts | Done (`prefetchExperts()` method) |

### Dynamic Cache Size

```typescript
// Based on MoE config
cacheSize = numExpertsPerToken * 2 + 1  // Capped at 16
```

---

## Core MoE Infrastructure

| Component | Status |
|-----------|--------|
| GPU-native routing (softmax+topk) | Done (Custom WGSL) |
| Expert FFN execution | Done (Per-expert matmul) |
| Scatter-add combination | Done (Custom WGSL kernel) |
| MoE router with load balancing | Done (`inference/moe-router.js`) |

---

## Expert-Aligned Storage

### Layout Comparison

| Layout | Reads/Expert | Bytes/Expert | Use Case |
|--------|--------------|--------------|----------|
| Current (interleaved) | 2-4 shards | ~192MB | Dense models |
| Expert-aligned | 1 shard | ~80MB | MoE models |
| Column-major | 1 slice | ~20MB | Tensor parallel |

### Expert-Aligned Manifest Example

```json
{
  "shardingStrategy": "expert",
  "shards": [
    { "index": 0, "type": "dense", "size": 134217728 },
    { "index": 1, "type": "expert", "expertKey": "0_0", "size": 83886080 },
    { "index": 2, "type": "expert", "expertKey": "0_1", "size": 83886080 }
  ]
}
```

---

## Hierarchical Routing (Design)

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input                                │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Tier-1 Gatekeeper (2B Dense)                   │
│  Semantic intent → Cluster selection → Prefetch trigger     │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Systems  │    │   Web    │    │  Math    │
    │ Cluster  │    │ Cluster  │    │ Cluster  │
    └──────────┘    └──────────┘    └──────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│            Tier-2 Granular Router (per-token)               │
│  Within-cluster top-k │ All candidates in RAM               │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Files

| File | Purpose |
|------|---------|
| `inference/moe-router.js` | Router implementation |
| `gpu/kernels/moe_gather.wgsl` | Expert gathering |
| `gpu/kernels/scatter_add.wgsl` | Output combination |
| `loader/doppler-loader.js` | Expert loading API, prefetching |
| `loader/expert-cache.js` | LRU cache with smart eviction |
| `src/formats/rdrr/types.js` | `MoEConfig` with expert mapping |
| `src/converter/writer.js` | Expert tensor detection during conversion |
