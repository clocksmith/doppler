# Phase 3: Scale Beyond WebLLM

**Status:** Planned
**Prerequisites:** Phase 1-2 (performance parity, MoE efficiency)
**Goal:** Run models larger than WebLLM's ~31GB limit using tiered memory.

---

## Milestones

- [x] Unified memory detection ✅
- [ ] 16GB model on 16GB unified memory (P0)
- [ ] 40GB+ model with expert paging (P0)
- [ ] 128K context via KV overflow (P1)

---

## Work Items

### 3.1 Unified Memory Support

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Detect unified memory architecture | P0 | ✅ Done | `memory/capability.ts` |
| Test 8GB model load | P0 | ⬜ TODO | Llama 3.1 8B |
| Test 16GB model load | P0 | ⬜ TODO | Needs larger model |
| Benchmark GPU↔unified latency | P1 | ⬜ TODO | Measure actual overhead |
| Document memory tier selection | P1 | ⬜ TODO | Auto-detection logic |

### 3.2 Expert Paging from P2P

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Expert shard request protocol | P0 | ⬜ TODO | Extend P2P spec |
| Expert availability announcements | P0 | ⬜ TODO | Peer inventory |
| Parallel expert fetch | P1 | ⬜ TODO | Multiple peers |
| Expert verification | P0 | ⬜ TODO | Hash check before use |

### 3.3 KV Cache Overflow

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| KV cache spill to unified memory | P1 | ⬜ TODO | For long contexts |
| Sliding window + spill hybrid | P2 | ⬜ TODO | Keep recent in VRAM |
| KV cache compression | P2 | ⬜ TODO | Reduce memory footprint |

### 3.4 Storage Layout Optimization

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Expert-aligned tensor ordering | P0 | ⬜ TODO | Group expert tensors in writer |
| Expert-per-shard sharding | P0 | ⬜ TODO | 1 shard = 1 expert |
| Variable shard sizes | P1 | ⬜ TODO | Match shard to expert size |
| `shardingStrategy` manifest field | P1 | ⬜ TODO | `fixed` / `expert` / `layer` |
| Shard type metadata | P1 | ⬜ TODO | `dense` vs `expert` shards |
| Column-major storage (TP) | P2 | ⬜ TODO | For tensor parallelism |
| Partial tensor loading | P2 | ⬜ TODO | Load slices for distributed |

#### Storage Layout Comparison

| Layout | Reads/Expert | Bytes/Expert | Use Case |
|--------|--------------|--------------|----------|
| Current (interleaved) | 2-4 shards | ~192MB | Dense models |
| Expert-aligned | 1 shard | ~80MB | MoE models |
| Column-major | 1 slice | ~20MB | Tensor parallel |

#### Expert-Aligned Manifest Example

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

### 3.5 Column-Major Storage for Tensor Parallelism

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Add `layout` field to TensorLocation | P2 | ⬜ TODO | `'row' \| 'column'` |
| Add `sliceDim`, `sliceIdx`, `sliceCount` fields | P2 | ⬜ TODO | For partial tensor loading |
| Implement column-wise tensor splitting in writer | P2 | ⬜ TODO | Split along output dimension |
| Update loader for partial tensor reads | P2 | ⬜ TODO | Load specific slices |
| Add transposed matmul kernels | P2 | ⬜ TODO | For column-major weights |

#### Column-Major Storage Layout

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

#### Column-Major Manifest Example

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

#### Tensor Parallelism Use Cases

| TP Rank | Loads | Output | Notes |
|---------|-------|--------|-------|
| 0 (of 4) | W[:, 0:K/4] | Y[:, 0:N/4] | First quarter |
| 1 (of 4) | W[:, K/4:K/2] | Y[:, N/4:N/2] | Second quarter |
| 2 (of 4) | W[:, K/2:3K/4] | Y[:, N/2:3N/4] | Third quarter |
| 3 (of 4) | W[:, 3K/4:K] | Y[:, 3N/4:N] | Fourth quarter |

Each rank only loads 25% of weight → 4x memory reduction per device.

---

## Architecture

### Tiered Memory

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

### WebLLM Limits vs DOPPLER

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

## Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| 16GB model on 16GB unified | No OOM | - | ⬜ |
| Expert page latency (OPFS) | < 100ms | - | ⬜ |
| Expert page latency (P2P) | < 500ms | - | ⬜ |
| Expert cache hit rate | > 80% | - | ⬜ |
| No regression on small models | Same tok/s | - | ⬜ |

---

## Key Files

| File | Purpose |
|------|---------|
| `memory/capability.ts` | Unified memory detection |
| `loader/doppler-loader.ts` | `loadExpert()` API, shard cache, partial tensor loading |
| `loader/expert-cache.ts` | LRU expert cache |
| `storage/shard-manager.ts` | Expert-level granularity |
| `storage/rdrr-format.ts` | Manifest types, sharding strategy, layout fields |
| `tools/rdrr-writer.ts` | Expert-aligned ordering, column-major transpose, weight fusion |
| `gpu/kernels/matmul.ts` | Layout-aware kernel selection |
| `inference/pipeline.ts` | Expert prefetch scheduling |
| `inference/kv-cache.ts` | Overflow to unified memory |

---

## Dependencies

- **Phase 1:** Buffer reuse, async pipeline
- **Phase 2:** MoE routing, expert paging infrastructure

---

## Next Phase

[Phase 4: P2P](PHASE_4_P2P.md) - Distributed shard distribution and remote inference.

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

