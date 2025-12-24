# Phase 4: P2P Distribution

**Status:** Design
**Prerequisites:** Phase 1-3 (performance, MoE, scale)
**Goal:** Distributed model distribution and optional remote inference via P2P mesh.

---

## Milestones

- [ ] MVP: Swarm shard cache working (P0)
- [ ] Expert paging from P2P swarm (P0)
- [ ] Hierarchical routing with cluster prefetch (P1)
- [ ] 10-peer swarm self-heals on dropout (P1)
- [ ] Remote inference offload (P2)

---

## Work Items

### 4.1 MVP: Swarm Shard Cache

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| WebTorrent tracker integration | P0 | ⬜ TODO | `wss://tracker` signaling |
| WebRTC data channel setup | P0 | ⬜ TODO | Control + bulk channels |
| Shard availability index | P0 | ⬜ TODO | Peers advertise hash inventory |
| Shard transfer protocol | P0 | ⬜ TODO | Request, stream, ack, retry |
| Shard integrity verification | P0 | ⬜ TODO | SHA256 check before use |
| Local OPFS caching | P0 | ⬜ TODO | Store verified shards |
| `setCustomShardLoader()` integration | P0 | ⬜ TODO | Plug into existing API |

### 4.2 P2P Expert Paging

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Expert shard registry in manifest | P0 | ⬜ TODO | Expert → shard mapping |
| Expert prefetch scheduler | P1 | ⬜ TODO | Prioritize critical path |
| Expert LRU in OPFS | P0 | ⬜ TODO | Hot set pinning |
| Peer "have" announcements | P0 | ⬜ TODO | Inventory gossip |
| Chunked expert transfer | P0 | ⬜ TODO | 128KB chunks |

### 4.3 Hierarchical Routing (Design Proposal)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Static cluster definitions | P1 | ⬜ TODO | Manual expert grouping |
| Tier-1 Gatekeeper (rule-based) | P1 | ⬜ TODO | Keyword matching |
| Tier-1 Gatekeeper (ML classifier) | P2 | ⬜ TODO | Semantic intent detection |
| Tier-2 cluster prefetch | P1 | ⬜ TODO | Prefetch entire cluster |
| Latency masking via prediction | P1 | ⬜ TODO | Prefetch before user finishes typing |
| Adaptive cluster boundaries | P2 | ⬜ TODO | Based on usage patterns |

### 4.4 Remote Inference (Optional)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Inference request/response protocol | P2 | ⬜ TODO | Prompt + options → tokens |
| Peer capability discovery | P2 | ⬜ TODO | GPU info, loaded models |
| Prefill offload | P2 | ⬜ TODO | Offload to fast peer |
| Result verification | P2 | ⬜ TODO | Hash-based output validation |

### 4.5 Infrastructure

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| ICE configuration (STUN/TURN) | P0 | ⬜ TODO | NAT traversal |
| TURN cost monitoring | P1 | ⬜ TODO | Disconnect if over budget |
| Peer scoring algorithm | P1 | ⬜ TODO | Throughput, latency, reliability |
| Connection keepalive | P0 | ⬜ TODO | Re-announce every 30s |

---

## Architecture

### Hierarchical Routing

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

### P2P Mesh

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

## Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Shard fetch from peer | < 500ms | - | ⬜ |
| Cluster prefetch hit rate | > 90% | - | ⬜ |
| Swarm self-healing (peer dropout) | < 5s recovery | - | ⬜ |
| Remote inference latency | < 2s for prefill offload | - | ⬜ |

---

## Key Files

| File | Purpose |
|------|---------|
| `p2p/tracker.ts` | WebTorrent tracker integration |
| `p2p/peer-connection.ts` | WebRTC data channels |
| `p2p/shard-transfer.ts` | Chunked shard protocol |
| `inference/hierarchical-router.ts` | Tier-1/Tier-2 routing |
| `storage/shard-manager.ts` | P2P shard loader integration |

---

## Dependencies

- **Phase 1-2:** Performance optimizations (efficient local inference)
- **Phase 3:** Expert paging infrastructure (OPFS caching)

---

## Next Phase

[Phase 5: Evolution](PHASE_5_EVOLUTION.md) - LoRA adapters and kernel evolution.

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

