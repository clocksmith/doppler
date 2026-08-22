# ADR-0001: Dual-Hexagon Compiler Pipeline with an Immutable Pack Spine

**Status:** Accepted  
**Date:** 2026-08-22  
**Deciders:** Doppler Core Architecture  
**Consulted:** Reploid Core, Tooling & Inference Teams  

---

## Context

Doppler began as a configurable research runtime for forward inference and prefill in WebGPU environments. In that model, runtime execution dynamically resolved device capabilities, remapped kernel paths, applied execution-v1 capability transforms, and progressively mutated shared state during model initialization.

In the August 2026 landscape (Microsoft Foundry Local, Transformers.js v4 C++ WebGPU rewrite, Chrome Built-in AI), local model execution is no longer defined by generic runtime interpreters or universal ONNX operator zoos. 

Doppler's strategic thesis is to become the **AI-native model release foundry for JavaScript and WebGPU**: taking newly released open-weight models and compiling them into small, explicit, verified, zero-daemon JavaScript/WGSL programs faster than any generic runtime can responsibly support them.

The historical pattern of load-time state mutation contradicts the contract of closed, verifiable Program Bundles. 

---

## Decision

We adopt a **dual-hexagon architecture connected by an immutable Pack spine**:

```text
Source Model Checkpoint
    ↓
┌──────────────────────────────────────────────────────────────┐
│                    DOPPLER FORGE                             │
│  Inspect → Normalize → Analyze → Lower → Specialize → Search │
│  → Verify → Qualify → Package → Sign                         │
└──────────────────────────────┬───────────────────────────────┘
                               │
                    Immutable Doppler Pack
                               │
┌──────────────────────────────▼───────────────────────────────┐
│                    DOPPLER RUNTIME                           │
│  Validate → Select qualified target → Bind resources         │
│  → Execute declared commands → Observe                       │
└──────────────────────────────┬───────────────────────────────┘
                               ↓
                   Browser / Electron / Node App
```

### 1. The Core Architectural Invariant

> **The Doppler Invariant Law:**  
> *After a Doppler Pack has been qualified and signed, no Runtime code may change its semantic graph, kernel closure, dtype lane, fusion strategy, or memory model.*

All graph-changing, kernel-changing, fusion-changing, layout-changing, and precision-changing work is performed ahead-of-time in **Doppler Forge**. The **Doppler Runtime** is strictly an uncreative plan–bind–execute engine that selects among pre-qualified target plans.

---

### 2. Three Hashable Representations

To eliminate abstraction blurring across model semantics, device targets, and runtime sessions, Doppler defines three separately content-addressed representations:

1. **`ModelIR` (Semantic Computation):**
   * Hardware-agnostic representation of the model's computation graph: tensor roles, layer topology, attention geometry, normalization, RoPE semantics, FFN/MoE/linear-attention semantics, output topology, and prefill/decode phase definitions.
   * Contains zero WGSL filenames, zero dispatch formulas, and zero hardware-specific tile configurations.

2. **`TargetPlan` (Concrete Hardware Specialization):**
   * Complete implementation for a discrete hardware target class (e.g. `webgpu-f16-subgroups`, `webgpu-f16`, `webgpu-f32-safe`).
   * Selected fusions, storage and compute dtypes, tensor layouts, memory slots and lifetimes, kernel IDs with cryptographic content digests (`sha256:...`), bindings, dispatch formulas, capability predicates, and phase command graphs.

3. **`SessionPlan` (Runtime Instance):**
   * Instance of a `TargetPlan` bound to runtime parameters: actual prompt length, max generation length, sampling parameters, and concrete GPU buffer allocations within the Pack's preflighted envelope.
   * Never alters the target graph or substitutes unqualified kernels.

---

### 3. Ports & Adapters Isolation

Ports and adapters are applied strictly at volatile boundaries:

* **Forge Ports:** `SourceReader`, `ReferenceExecutor`, `CandidateProposer`, `KernelCompiler`, `KernelOracle`, `DeviceLabRunner`, `ArtifactPublisher`, `Signer`, `EvidenceStore`.
  * *Adapters:* SafeTensors/GGUF readers, PyTorch reference runner, AI proposer, Chromium/Node WebGPU runners, Hugging Face/OPFS publishers.
* **Runtime Ports:** `PackSource`, `ArtifactStore`, `GpuDevice`, `PersistentCache`, `WorkerScheduler`, `ObservationSink`.
  * *Adapters:* HTTP, OPFS, Node filesystem, Web Worker, Electron worker, browser `navigator.gpu`, Node WebGPU.

### 4. Dependency Direction Law

```text
runtime ───────▶ pack
forge ─────────▶ pack
cli ───────────▶ forge + runtime

runtime ───X───▶ forge
runtime ───X───▶ internal lab
runtime ───X───▶ model-family code
runtime ───X───▶ cloud routing
```

---

## Consequences

### Positive
* **Deterministic Verification:** Eliminates nondeterministic load-time mutation bugs and precision regressions.
* **Zero Runtime Overhead:** Runtime becomes a lightweight (~small bundle) plan–bind–execute machine with zero generic interpreter bloat.
* **AI-Assisted Foundry Velocity:** AI proposes graphs, fusions, and kernels in Forge where deterministic test oracles (CPU oracles, metamorphic bounds) catch defects before pack signing.
* **First-Class Specialist Support:** Biological sequence models (ESM-2, ESMC, Nucleotide Transformer for Reploid) and Retrieval models enjoy identical AOT packaging guarantees.

### Phased Migration
Existing `execution-v1` and Program Bundle v1 infrastructure serves as the immediate foundation and is migrated progressively without disrupting existing verified model lanes.
