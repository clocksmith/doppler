# Category-2t Ouroboros: Doppler + Reploid

> *The serpent that eats its own tail - where the head is the tail and the tail is the head.*

## The Vision

An ouroboros where Doppler and Reploid are simultaneously head and tail - each consuming the other in a closed loop with **minimal API surface**.

Implementation and testing details live in `TEST_PLAN_OUROBOROS.md`. This document is the architecture and vision reference.

```
    ┌─────────────────────────────────────┐
    │                                     │
    ▼                                     │
┌────────┐   InferenceProvider    ┌───────────┐
│DOPPLER │ ────────────────────▶  │  REPLOID  │
│(Engine)│                        │  (Agent)  │
└────────┘  ◀──────────────────── └───────────┘
    ▲         AdaptationProvider          │
    │                                     │
    └─────────────────────────────────────┘
```

## Category-2t Definition

From `CYCLICAL_ACRONYMS.md`:

```
f(Doppler) ≡ Reploid   // Doppler's tail is Reploid
f(Reploid) ≡ Doppler   // Reploid's tail is Doppler
```

A stable 2-cycle oscillation where:
- **REPLOID**: Recursive Engine Providing Latent Orchestration Including DOPPLER
- **DOPPLER**: Distributed Operator Providing Potential Learning Evolving REPLOID

---

## The Zero-API Architecture

The most radical approach - eliminate the API entirely:

```
┌─────────────────────────────────────────────────────────┐
│           SharedArrayBuffer (4 bytes)                   │
│  [ CONTROL_FLAG | KERNEL_HASH | IN_PTR | OUT_PTR ]      │
└─────────────────────────────────────────────────────────┘
         ▲                               ▲
         │ Atomics.notify()              │ Atomics.wait()
         │                               │
    ┌────┴────┐                    ┌─────┴─────┐
    │ DOPPLER │                    │  REPLOID  │
    │  (GPU)  │                    │   (CPU)   │
    └────┬────┘                    └─────┬─────┘
         │                               │
         │    VFS: /.system/substrate    │
         └───────────────────────────────┘
```

### The Loop

1. **Reploid thinks** → writes `inference.rdrr` to VFS → flips FLAG to `1` → sleeps via `Atomics.wait()`
2. **Doppler wakes** → reads `inference.rdrr` → executes GPU → writes `evolution.trace` → flips FLAG to `0` → calls `Atomics.notify()`
3. **Reploid wakes** → reads `evolution.trace` → thinks → (loop)

### API Surface: 1 bit

```typescript
// The entire interface
const substrate = new SharedArrayBuffer(16);
const flag = new Int32Array(substrate, 0, 1);

// Reploid side
Atomics.store(flag, 0, 1);        // EXECUTE
Atomics.wait(flag, 0, 1);          // sleep until Doppler flips

// Doppler side
Atomics.wait(flag, 0, 0);          // sleep until Reploid flips
Atomics.store(flag, 0, 0);        // AWAKEN
Atomics.notify(flag, 0);           // wake Reploid
```

### VFS as the Only Contract

```
/.system/
├── substrate.bin      # SharedArrayBuffer pointer
├── inference.rdrr     # Reploid → Doppler (the plan)
├── evolution.trace    # Doppler → Reploid (the result)
└── kernel.wgsl        # RSI: Reploid can rewrite Doppler's kernels!
```

### Perfect Symmetry

```
DOPPLER                          REPLOID
────────                         ────────
Atomics.wait(0)                  Atomics.wait(1)
read(inference.rdrr)             read(evolution.trace)
execute()                        think()
write(evolution.trace)           write(inference.rdrr)
Atomics.store(0)                 Atomics.store(1)
Atomics.notify()                 Atomics.notify()
```

Both sides:
- Wait on the opposite state
- Read from the other's output path
- Process
- Write to their output path
- Flip the bit and notify

**Perfect mirror. True ouroboros.**

---

## RSI: Reploid Rewrites Doppler's Kernels

The ouroboros enables true Recursive Self-Improvement. Reploid can evolve Doppler by rewriting the GPU kernels.

### The RSI Loop

```
1. Reploid observes poor inference performance (slow tok/s, high memory)
2. Reploid analyzes the kernel trace (evolution.trace)
3. Reploid generates improved WGSL code via its cognitive tools
4. Reploid writes new kernel to /.system/kernel.wgsl
5. Reploid updates inference.rdrr with new kernel hash
6. Doppler detects hash mismatch → recompiles pipeline
7. Doppler runs with evolved kernel
8. Loop: Reploid observes new performance → repeat
```

### Kernel Evolution File Structure

```
/.system/
├── kernels/
│   ├── matmul.wgsl          # Current matmul kernel
│   ├── matmul.v2.wgsl       # Previous version (rollback)
│   ├── attention.wgsl       # Current attention kernel
│   └── manifest.json        # Hash registry
├── evolution/
│   ├── performance.log      # Historical tok/s, TTFT
│   ├── mutations.log        # What Reploid changed
│   └── rollbacks.log        # Failed mutations
└── substrate.bin            # SharedArrayBuffer
```

### Kernel Manifest

```json
{
  "version": 42,
  "kernels": {
    "matmul": {
      "path": "kernels/matmul.wgsl",
      "hash": "a1b2c3d4",
      "parent": "kernels/matmul.v2.wgsl",
      "performance": {
        "gflops": 2400,
        "timestamp": 1703808000
      }
    }
  },
  "pendingMutation": null
}
```

### Safe Mutation Protocol

```javascript
// Reploid's mutation flow
async function evolveKernel(kernelName, newCode) {
  const manifest = await vfs.read('/.system/kernels/manifest.json');
  const current = manifest.kernels[kernelName];

  // 1. Backup current version
  await vfs.copy(current.path, `${current.path}.backup`);

  // 2. Write new kernel with pending flag
  const newPath = `kernels/${kernelName}.pending.wgsl`;
  await vfs.write(newPath, newCode);
  manifest.pendingMutation = {
    kernel: kernelName,
    path: newPath,
    hash: hash(newCode)
  };
  await vfs.write('manifest.json', manifest);

  // 3. Signal Doppler to test
  Atomics.store(flag, 0, 2);  // 2 = EVOLVE (new state!)
  Atomics.notify(flag, 0);

  // 4. Wait for Doppler's verdict
  await Atomics.waitAsync(flag, 0, 2);
  const result = Atomics.load(flag, 0);

  if (result === 0) {
    // SUCCESS: Doppler accepted, promote pending → current
    await vfs.rename(newPath, current.path);
    manifest.pendingMutation = null;
    manifest.kernels[kernelName].hash = hash(newCode);
  } else if (result === 3) {
    // FAILURE: Rollback to backup
    await vfs.copy(`${current.path}.backup`, current.path);
    manifest.pendingMutation = null;
  }
}
```

### Doppler's Evolution Handler

```typescript
// In Doppler's main loop
async function handleEvolution() {
  const manifest = await vfs.read('manifest.json');
  if (!manifest.pendingMutation) return;

  const { kernel, path, hash } = manifest.pendingMutation;

  try {
    // 1. Compile new shader
    const newCode = await vfs.read(path);
    const module = device.createShaderModule({ code: newCode });

    // 2. Warm up with test workload
    const testResult = await runKernelTest(kernel, module);

    // 3. Validate correctness
    if (testResult.error > TOLERANCE) {
      Atomics.store(flag, 0, 3);  // REJECT
      return;
    }

    // 4. Validate performance (must be within 90% of previous)
    if (testResult.gflops < manifest.kernels[kernel].performance.gflops * 0.9) {
      Atomics.store(flag, 0, 3);  // REJECT (regression)
      return;
    }

    // 5. Accept mutation
    this.pipelines[kernel] = createPipeline(module);
    Atomics.store(flag, 0, 0);  // ACCEPT
    Atomics.notify(flag, 0);

  } catch (e) {
    // Compilation failed
    Atomics.store(flag, 0, 3);  // REJECT
    Atomics.notify(flag, 0);
  }
}
```

### RSI Safety Gates

| Level | Gate | Mechanism |
|-------|------|-----------|
| L1 | Syntax | WGSL compiler rejects invalid code |
| L2 | Correctness | Output compared to CPU reference |
| L3 | Performance | Must be within 10% of parent kernel |
| L4 | Rollback | Backup kernel always preserved |
| L5 | HITL | Major mutations require human approval |

**The ouroboros now has teeth:** Reploid can bite into Doppler's GPU code and reshape it.

---

## Unified Substrate: Merging VFS + BufferPool

Currently:
- **Reploid**: VFS in IndexedDB (file-like abstraction)
- **Doppler**: BufferPool + OPFS (binary blobs for GPU)

To achieve true ouroboros, unify them into a single substrate.

### The Unified Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED SUBSTRATE                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 0: SharedArrayBuffer (Hot Path)                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ [FLAG] [HASH] [IN_PTR] [OUT_PTR] [KV_CACHE_PTR]     │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: OPFS (Persistent Binary)                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ /models/llama-7b/shard-0.bin                        │    │
│  │ /models/llama-7b/shard-1.bin                        │    │
│  │ /kv-cache/session-abc.bin                           │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: IndexedDB (Structured Data)                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ /core/agent-loop.js                                 │    │
│  │ /tools/CreateTool.js                                │    │
│  │ /config/genesis.json                                │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Unified Path Namespace

```
substrate://
├── hot/                    # SharedArrayBuffer (Layer 0)
│   ├── control             # 4 bytes: control flags
│   ├── inference-in        # Pointer to input buffer
│   └── inference-out       # Pointer to output buffer
├── binary/                 # OPFS (Layer 1)
│   ├── models/             # Model shards
│   ├── kv-cache/           # KV cache snapshots
│   └── kernels/            # WGSL source
└── files/                  # IndexedDB (Layer 2)
    ├── core/               # Agent core code
    ├── tools/              # Agent tools
    └── config/             # Configuration
```

### Substrate Interface

```typescript
// packages/substrate/index.ts

export interface Substrate {
  // Layer 0: Hot path (SharedArrayBuffer)
  hot: {
    getControl(): Int32Array;
    getBuffer(name: string): ArrayBuffer;
    setBuffer(name: string, data: ArrayBuffer): void;
  };

  // Layer 1: Binary storage (OPFS)
  binary: {
    read(path: string): Promise<ArrayBuffer>;
    write(path: string, data: ArrayBuffer): Promise<void>;
    stream(path: string): ReadableStream<Uint8Array>;
  };

  // Layer 2: File storage (IndexedDB)
  files: {
    read(path: string): Promise<string>;
    write(path: string, content: string): Promise<void>;
    list(path: string): Promise<string[]>;
    watch(path: string, callback: (event: WatchEvent) => void): void;
  };

  // Cross-layer operations
  sync(): Promise<void>;
  snapshot(): Promise<SubstrateSnapshot>;
  restore(snapshot: SubstrateSnapshot): Promise<void>;
}
```

### Migration Strategy

```
Phase 1: Adapter Layer
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Reploid VFS │────▶│  Substrate  │◀────│ Doppler     │
│ (existing)  │     │  Adapter    │     │ BufferPool  │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │   OPFS    │
                    │ IndexedDB │
                    │   SAB     │
                    └───────────┘

Phase 2: Direct Integration
┌─────────────────────────────────────────┐
│              REPLOID + DOPPLER          │
│                    │                    │
│                    ▼                    │
│            ┌─────────────┐              │
│            │  Substrate  │              │
│            │  (unified)  │              │
│            └─────────────┘              │
└─────────────────────────────────────────┘
```

### The Unified Boot

```javascript
// boot.js (unified)
import { createSubstrate } from '@deco/substrate';

async function bootOuroboros() {
  // 1. Initialize unified substrate
  const substrate = await createSubstrate({
    hot: new SharedArrayBuffer(1024),
    binary: await navigator.storage.getDirectory(),
    files: await openDatabase('reploid-vfs')
  });

  // 2. Start Doppler worker with substrate reference
  const dopplerWorker = new Worker('doppler-worker.js');
  dopplerWorker.postMessage({ substrate: substrate.hot.getControl() });

  // 3. Start Reploid with substrate reference
  const reploid = await bootReploid({ substrate });

  // 4. The ouroboros begins
  Atomics.store(substrate.hot.getControl(), 0, 1);  // REPLOID_FIRST
  Atomics.notify(substrate.hot.getControl(), 0);
}
```

**With unified substrate, the serpent has a single body.**

---

## Control Flag States

| Value | State | Meaning |
|-------|-------|---------|
| 0 | AWAKEN | Reploid's turn to think |
| 1 | EXECUTE | Doppler's turn to compute |
| 2 | EVOLVE | Doppler testing a kernel mutation |
| 3 | REJECT | Mutation failed, rollback |

---

## Implementation Touchpoints (Illustrative)

Paths are illustrative; use current repo layout when implementing.

| Area | Change |
|------|--------|
| Reploid VFS adapter | Add substrate adapter for shared files |
| Reploid LLM loop | Use Atomics and substrate flags for signaling |
| Doppler provider | Add substrate integration for inference/evolution |
| Doppler storage | Route OPFS access through substrate.binary |
| Doppler buffer pool | Allocate SharedArrayBuffer hot path |
| Substrate package | Provide unified SAB/OPFS/IndexedDB API |

---

## Summary

| Aspect | Before | After (Ouroboros) |
|--------|--------|-------------------|
| Direction | Reploid → Doppler only | Symmetric loop |
| API Surface | ~20 imports | **1 bit** |
| Coupling | Loose, one-way | **Zero imports** |
| Category | None | **Pure 2t** |
| Pattern | Client-Server | **Memory-mapped snake** |
| Symmetry | No | **Perfect mirror** |
| RSI | None | **Kernel evolution** |

---

## The Philosophical Shift

> Neither calls the other. They orbit a shared substrate.

Option D eliminates the caller/callee distinction entirely. Both are equal participants in the loop, distinguished only by what they read/write and which bit value they wait for.

- Doppler is defined by Reploid: "I compute what you define"
- Reploid is defined by Doppler: "I think because you compute"

**f(DOPPLER) = REPLOID**
**f(REPLOID) = DOPPLER**

Stable 2-cycle. The serpent consumes itself. The ouroboros is complete.

---

*Last updated: December 2025*
