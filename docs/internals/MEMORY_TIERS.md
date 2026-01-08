# Memory Tiers Internals

Technical deep-dive on tiered memory architecture, unified memory, and storage layouts.

---

## Tiered Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GPU VRAM (8-24 GB)                   │
│  Active layers │ Hot experts (top 25%) │ KV cache (active)  │
└─────────────────────────────────────────────────────────────┘
                              ↑↓ ~1ms
┌─────────────────────────────────────────────────────────────┐
│                    Unified Memory (32-128 GB)               │
│  Warm experts (next 50%) │ KV overflow │ Prefetch buffer    │
└─────────────────────────────────────────────────────────────┘
                              ↑↓ ~50ms
┌─────────────────────────────────────────────────────────────┐
│                        OPFS Cache (10-50 GB)                │
│  Cold experts (rare 25%) │ Model shards │ LoRA adapters     │
└─────────────────────────────────────────────────────────────┘
                              ↑↓ ~200ms
┌─────────────────────────────────────────────────────────────┐
│                        P2P Swarm (Unlimited)                │
│  Rare shards │ New models │ Community adapters              │
└─────────────────────────────────────────────────────────────┘
```

---

## WebLLM Limits vs DOPPLER

| Constraint | WebLLM Limit | DOPPLER Target |
|------------|--------------|----------------|
| Model size (unified mem) | ~31GB | 60GB+ |
| Model size (with paging) | ~31GB | 100GB+ (MoE) |
| Dynamic expert loading | No | Yes |
| Cross-session persistence | No | Yes (OPFS) |
| Context length | Fixed | Dynamic (KV overflow) |

---

## Target Models

| Model | Total Size | Active Size | Strategy |
|-------|------------|-------------|----------|
| Llama 3.1 8B | ~8GB | 8GB | Unified memory |
| Llama 3.1 70B | ~35GB | 35GB | Unified memory (64GB Mac) |
| Phi-mini-MoE | ~15GB | ~2.5GB | Expert paging (small) |
| Mixtral 8x7B | ~90GB | ~24GB | Expert paging |
| GPT-OSS 20B | ~40GB | ~8GB | Expert paging |
| Qwen3-30B-A3B | ~60GB | ~9GB | Expert paging |
| Qwen3-235B-A22B | ~470GB | ~51GB | Expert paging + P2P |
| Kimi K2 (1T) | ~2TB | ~82GB | P2P distributed (stretch) |

---

## Column-Major Storage for Tensor Parallelism

For tensor-parallel inference, store weights column-wise to enable partial loading:

```
Standard (row-major):  W[out, in] stored row-by-row
  Shard contains: rows 0-N of full weight matrix
  Loading: Must load entire tensor

Column-major:          W[out, in] stored column-by-column
  Shard 0: W[:, 0:K/4]      // First quarter of columns
  Shard 1: W[:, K/4:K/2]    // Second quarter
  Shard 2: W[:, K/2:3K/4]   // Third quarter
  Shard 3: W[:, 3K/4:K]     // Fourth quarter
  Loading: Can load partial tensor for TP rank
```

### Tensor Parallelism Use Cases

| TP Rank | Loads | Output | Notes |
|---------|-------|--------|-------|
| 0 (of 4) | W[:, 0:K/4] | Y[:, 0:N/4] | First quarter |
| 1 (of 4) | W[:, K/4:K/2] | Y[:, N/4:N/2] | Second quarter |
| 2 (of 4) | W[:, K/2:3K/4] | Y[:, N/2:3N/4] | Third quarter |
| 3 (of 4) | W[:, 3K/4:K] | Y[:, 3N/4:N] | Fourth quarter |

Each rank only loads 25% of weight → 4x memory reduction per device.

### Column-Major Manifest Example

```json
{
  "defaultWeightLayout": "column",
  "tensors": {
    "layers.0.self_attn.q_proj.weight": {
      "shard": 0,
      "offset": 0,
      "size": 8388608,
      "shape": [4096, 4096],
      "dtype": "f16",
      "layout": "column",
      "originalShape": [4096, 4096],
      "sliceDim": 1,
      "sliceIdx": 0,
      "sliceCount": 4
    }
  }
}
```

---

## P2P Mesh Architecture

```
Agent A                    Agent B                    Agent C
   │                          │                          │
   │◄─── shard request ───────│                          │
   │──── verified shard ─────►│                          │
   │                          │◄─── shard request ───────│
   │                          │──── verified shard ─────►│
   │                          │                          │
   └──────────── mesh gossip: who has what ─────────────┘
```

---

## KV Cache Overflow

For long contexts, KV cache can spill to unified memory:

| Task | Purpose |
|------|---------|
| KV cache spill to unified memory | For long contexts |
| Sliding window + spill hybrid | Keep recent in VRAM |
| KV cache compression | Reduce memory footprint |

---

## Key Files

| File | Purpose |
|------|---------|
| `memory/capability.ts` | Unified memory detection |
| `loader/doppler-loader.ts` | `loadExpert()` API, shard cache, partial tensor loading |
| `loader/expert-cache.ts` | LRU expert cache |
| `storage/shard-manager.ts` | Expert-level granularity |
| `src/formats/rdrr/types.ts` | Manifest types, sharding strategy, layout fields |
| `src/converter/writer.ts` | Expert-aligned ordering, column-major transpose, weight fusion |
| `gpu/kernels/matmul.ts` | Layout-aware kernel selection |
| `inference/pipeline.ts` | Expert prefetch scheduling |
| `inference/kv-cache.ts` | Overflow to unified memory |
