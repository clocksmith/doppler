# Phase 5: Evolution

**Status:** In Progress (Tier 1 Infrastructure Complete)
**Prerequisites:** Phase 4 (P2P distribution)
**Goal:** Enable model evolution via LoRA adapters and kernel improvements without retraining.

---

## Milestones

- [x] Static adapter loading from local files (P0) - COMPLETE
- [x] Adapter registry with OPFS/IndexedDB persistence (P0) - COMPLETE
- [ ] Adapter registry with P2P distribution (P0)
- [ ] User profile with automatic adapter selection (P1)
- [ ] Micro-LoRA training on corrections (P2)
- [ ] Kernel variant distribution via swarm (P2)

---

## Work Items

### 5.1 Adapter Infrastructure (Tier 1 - COMPLETE)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| LoRA weight loading | P0 | ✅ DONE | `loadLoRAWeights()` - OPFS/URL support |
| Adapter manifest format | P0 | ✅ DONE | JSON Schema + TypeScript types |
| Local adapter registry | P0 | ✅ DONE | IndexedDB + memory storage |
| Adapter composition (multiple LoRAs) | P1 | ✅ DONE | Weighted sum merge strategy |
| Adapter enable/disable API | P0 | ✅ DONE | `AdapterManager` class |
| Safetensors support | P1 | ✅ DONE | F16/BF16 conversion |
| Unit tests | P0 | ✅ DONE | 40+ test cases |

### 5.2 P2P Adapter Distribution

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Adapter announcement protocol | P0 | ⬜ TODO | Extend shard gossip |
| Adapter fetch from peers | P0 | ⬜ TODO | Same as shard transfer |
| Adapter integrity verification | P0 | ⬜ TODO | SHA256 before use |
| HITL approval workflow | P1 | ⬜ TODO | Human review for risky adapters |
| Adapter reputation system | P2 | ⬜ TODO | Trust scoring |

### 5.3 Personalization

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| User profile schema | P1 | ⬜ TODO | Languages, frameworks, domains |
| Profile → adapter matching | P1 | ⬜ TODO | Auto-attach relevant LoRAs |
| Adapter usage analytics | P2 | ⬜ TODO | Track effectiveness |
| Profile sync across devices | P2 | ⬜ TODO | Optional cloud backup |

### 5.4 Self-Healing (Micro-LoRA)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Failure detection | P2 | ⬜ TODO | Task success/fail tracking |
| Correction capture | P2 | ⬜ TODO | (prompt, bad, good) triplets |
| Micro-LoRA training | P2 | ⬜ TODO | Rank 4, single example |
| Correction validation | P2 | ⬜ TODO | Test before broadcast |
| Correction broadcast | P2 | ⬜ TODO | Share with swarm |

### 5.5 Kernel Evolution

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Kernel variant manifest | P2 | ⬜ TODO | Operation, arch, benchmarks |
| Kernel announcement protocol | P2 | ⬜ TODO | Extend P2P gossip |
| Kernel A/B testing | P2 | ⬜ TODO | Compare performance |
| Kernel adoption consensus | P2 | ⬜ TODO | Majority adoption |
| Background recompilation | P2 | ⬜ TODO | Hot-swap kernels |

---

## Architecture

### Adapter Registry

```typescript
interface AdapterManifest {
  id: string;                    // SHA256 hash
  name: string;                  // Human-readable
  description: string;           // What it does
  baseModelHash: string;         // Compatible base model
  targetExperts: number[];       // Which experts modified
  rank: number;                  // LoRA rank (4, 8, 16)
  size: number;                  // Bytes
  author: string;                // Publisher identity
  signature?: string;            // Cryptographic signature
  hitlApproved: boolean;         // Human approval status
}
```

### Model Evolution Flow

```
1. Base model: verified hash H0
2. LoRA adapter proposed: hash H1
3. HITL review: human approves/rejects
4. If approved: swarm distributes H1
5. Peers can run base (H0) or evolved (H0+H1)
```

### Risk Levels

| Change Type | Risk Level | Approval Required |
|-------------|------------|-------------------|
| Style adapter (formatting) | Low | Auto-approve |
| Domain adapter (React, Rust) | Medium | Peer consensus |
| Core behavior adapter | High | HITL required |
| Kernel replacement | Critical | HITL + testing |

---

## Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Adapter load time | < 100ms | - | ⬜ |
| Personalization preference | 70%+ prefer | - | ⬜ |
| Micro-LoRA fix rate | 50%+ captured | - | ⬜ |
| Kernel improvement | 10%+ throughput | - | ⬜ |
| Adapter adoption latency | < 1 hour | - | ⬜ |

---

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `adapters/index.ts` | Module exports | ✅ |
| `adapters/adapter-manifest.ts` | JSON Schema + types | ✅ |
| `adapters/lora-loader.ts` | LoRA weight loading | ✅ |
| `adapters/adapter-manager.ts` | Enable/disable API | ✅ |
| `adapters/adapter-registry.ts` | Persistent storage | ✅ |
| `tests/unit/lora-infrastructure.test.ts` | Unit tests | ✅ |
| `inference/pipeline/lora.ts` | Module lookup | ✅ (existing) |
| `inference/pipeline/lora-apply.ts` | GPU application | ✅ (existing) |
| `inference/pipeline/lora-types.ts` | Core types | ✅ (existing) |
| `adapters/matcher.ts` | Profile → adapter matching | ⬜ TODO |
| `p2p/adapter-protocol.ts` | P2P adapter distribution | ⬜ TODO |
| `evolution/micro-lora.ts` | Self-healing training | ⬜ TODO |

---

## Dependencies

- **Phase 4:** P2P infrastructure (distribution mechanism)
- **Phase 4:** Hierarchical routing (adapter selection integration)

---

## Open Questions

1. **LoRA training in browser:** Can we train micro-LoRAs in WebGPU? Or server-side?
2. **Adapter composition:** How do multiple LoRAs merge? Addition? Concatenation?
3. **Version conflicts:** Incompatible adapter versions across peers?
4. **Sybil attacks:** Prevent flooding swarm with bad adapters?
5. **Privacy:** Does adapter usage leak user behavior?

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

