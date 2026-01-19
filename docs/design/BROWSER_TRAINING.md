# Browser Training Specification
## Browser Training

## Goal

Add LoRA fine-tuning capability to Doppler, enabling in-browser training of FunctionGemma and other small models without external infrastructure.

**Principle:** Doppler provides training primitives (backward kernels, optimizer steps). Reploid/gamma decides training policy (when to train, what data, loss thresholds).

---

## Status

Implemented in `src/training` with a runnable demo harness and test coverage.

Key entry points:
- Training engine overview: `../src/training/README.md`
- Demo harness: `app/training.html` + `app/training-harness.js`
- Training tests: `tests/training/browser/test-page.js` and `npm test -- --training`
- Export spec: `FORMATS.md`

This document now serves as a historical design reference.

---

## Why Browser Training

1. **Self-improvement loop:** Reploid can fine-tune FunctionGemma on its own traces without leaving the browser
2. **Privacy:** Training data never leaves the device
3. **No infra:** No Colab, no cloud GPUs, no Python environment required
4. **Unique capability:** No other browser ML framework offers LoRA training

---

## Target Model: FunctionGemma 270M

**Exact forward graph (ops and shapes for seq_len=128, batch=1):**

```
embed:        [vocab=256128, dim=1152] → [128, 1152]
per layer (×12):
  rmsnorm:    [128, 1152] → [128, 1152]
  qkv_proj:   [1152, 3456] → Q[128,1152], K[128,1152], V[128,1152]
  rope:       Q, K × freqs → Q', K'
  attention:  softmax(Q'K'^T / √d) × V → [128, 1152]
  o_proj:     [1152, 1152] → [128, 1152]
  residual:   + input
  rmsnorm:    [128, 1152] → [128, 1152]
  gate_proj:  [1152, 6912] → [128, 6912]
  up_proj:    [1152, 6912] → [128, 6912]
  silu:       gate × silu(up) → [128, 6912]
  down_proj:  [6912, 1152] → [128, 1152]
  residual:   + input
final_rmsnorm: [128, 1152] → [128, 1152]
lm_head:      [1152, 256128] → [128, 256128]
```

**Backward kernels required for this graph:**

| Op | Backward Kernel | Notes |
|----|-----------------|-------|
| embed | embed_backward | Uses scatter_add for gradient accumulation to embedding table |
| rmsnorm | rmsnorm_backward | Variance chain rule |
| matmul (all projs) | matmul_backward | dW = x.T @ dy, dx = dy @ W.T (reuses forward with transpose) |
| rope | rope_backward | sin/cos derivatives |
| attention | attention_backward | softmax + matmul chain, **includes causal mask backward** |
| softmax | softmax_backward | y * (dy - sum(y * dy)) |
| silu | silu_backward | σ(x) * (1 + x * (1 - σ(x))) |
| gelu | gelu_backward | GELU derivative (if model uses GELU variant) |
| scale | scale_backward | Identity scaled by constant (for LoRA alpha) |
| residual | (none) | Gradient passes through unchanged |
| cross_entropy | cross_entropy_backward | -log(p) derivative, returns logit gradients |

---

## Scope

### Phase 1: LoRA-Only Training

Train LoRA adapters while keeping base model frozen. This minimizes memory and avoids weight-gradient kernels for base model.

**In scope:**
- Autograd tape for forward graph (config-backed op registry)
- Backward kernels for all FunctionGemma ops (see table above)
- LoRA forward/backward through adapter layers
- Adam optimizer kernel
- Cross-entropy loss kernel
- Gradient clipping
- Mixed precision (f16 activations, f32 gradients)
- Adapter export (custom RDRR-LoRA format, not safetensors)

**Out of scope (Phase 2+):**
- Full fine-tuning (weight gradients for base model)
- Gradient checkpointing
- Per-expert MoE LoRA routing
- Distributed training across P2P mesh

**Implementation order:**

1. Autograd tape
2. Backward kernels
3. LoRA layer
4. Optimizer
5. Training loop
6. Data pipeline
7. Testing + parity

---

## Architecture

### Engine/Driver Split

```
┌─────────────────────────────────────────────────────────────┐
│  REPLOID (Driver) - Training Policy                         │
│  ├─ decideWhenToTrain(traces, threshold)                    │
│  ├─ prepareBatch(traces, batchSize)                         │
│  ├─ calculateLoss(logits, targets) → policy                 │
│  ├─ shouldCheckpoint(step, metrics)                         │
│  └─ exportAdapter(path)                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  DOPPLER (Engine) - Training Primitives                     │
│  ├─ forward(input, weights, loraAdapters) → activations     │
│  ├─ backward(loss, tape) → gradients                        │
│  ├─ optimizerStep(params, grads, state) → updated           │
│  ├─ clipGradients(grads, maxNorm) → clipped                 │
│  ├─ crossEntropyLoss(logits, targets) → loss                │
│  └─ exportLoraWeights(adapters) → ArrayBuffer               │
└─────────────────────────────────────────────────────────────┘
```

### Memory Layout

**Realistic estimates for different configurations:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FunctionGemma 270M + LoRA r=16                                             │
│  Config: seq_len=128, batch=1 (MINIMUM)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Base weights (frozen, f16)              540 MB                             │
│  LoRA adapters (trainable, f16)          4 MB   (q,k,v,o × 12 layers)      │
│  Activations (forward, f16)              85 MB  (all intermediate states)  │
│  Activation gradients (backward, f32)    170 MB (2× activations for f32)   │
│  LoRA gradients (f32)                    8 MB                               │
│  Optimizer state (Adam m,v for LoRA)     16 MB  (2× LoRA params, f32)      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  TOTAL                                   ~823 MB                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Config: seq_len=256, batch=2 (REALISTIC)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Base weights (frozen, f16)              540 MB                             │
│  LoRA adapters (trainable, f16)          4 MB                               │
│  Activations (forward, f16)              340 MB (4× minimum)               │
│  Activation gradients (backward, f32)    680 MB (2× activations)           │
│  LoRA gradients (f32)                    8 MB                               │
│  Optimizer state (Adam)                  16 MB                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  TOTAL                                   ~1.6 GB                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Config: seq_len=512, batch=4 (AGGRESSIVE)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Base weights (frozen, f16)              540 MB                             │
│  LoRA adapters (trainable, f16)          4 MB                               │
│  Activations (forward, f16)              1.36 GB (16× minimum)             │
│  Activation gradients (backward, f32)    2.72 GB                            │
│  LoRA gradients + optimizer              24 MB                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  TOTAL                                   ~4.6 GB                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Hardware requirements:**

| Config | Min VRAM | Recommended |
|--------|----------|-------------|
| seq=128, batch=1 | 2GB | 4GB |
| seq=256, batch=2 | 4GB | 8GB |
| seq=512, batch=4 | 6GB | 16GB |

**Note:** Unified RAM Macs can use system memory, so 8GB M1 works for realistic config. Discrete GPUs need the VRAM directly.

---

## Config Wiring

Training config must follow Doppler's "config as code" pattern. This section defines the plumbing.

### Backward Registry

```
src/config/schema/backward-registry.js      # Schema definition
src/config/schema/backward-registry.d.ts
src/config/kernels/backward-registry.json   # Data (op→kernel mapping)
src/config/backward-registry-loader.js      # Loader with validation
src/config/backward-registry-loader.d.ts
```

**Loader integrates with config merge:**

```javascript
// src/config/backward-registry-loader.js
import { validateBackwardRegistry } from './schema/backward-registry.js';
import backwardRegistryData from './kernels/backward-registry.json';

export function loadBackwardRegistry() {
  const validated = validateBackwardRegistry(backwardRegistryData);
  return validated;
}
```

### Training Config Merge

Training defaults merge into Doppler config via `createTrainingConfig()`:

```javascript
// src/config/training-defaults.js
import { DEFAULT_DOPPLER_CONFIG } from './index.js';

export const DEFAULT_TRAINING_CONFIG = {
  ...DEFAULT_DOPPLER_CONFIG,
  training: {
    enabled: false,
    lora: {
      rank: 16,
      alpha: 32,
      dropout: 0.0,
      targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    },
    optimizer: {
      type: 'adam',
      lr: 2e-4,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0.0,
    },
    gradient: {
      maxNorm: 1.0,
      accumSteps: 1,
    },
    precision: {
      activations: 'f16',
      gradients: 'f32',
      loraParams: 'f32',
    },
    attention: {
      recomputeForward: false,  // See "Forward Cache vs Recompute" below
    },
  },
};

export function createTrainingConfig(overrides = {}) {
  return mergeConfig(DEFAULT_TRAINING_CONFIG, overrides);
}
```

### Kernel Registry Wiring

Backward kernels live in `src/gpu/kernels/backward/` but the global kernel registry expects WGSL paths relative to `src/gpu/kernels/`.

**Registry entries for backward kernels (structure matches kernel registry schema):**

```json
// Addition to src/config/kernels/registry.json
{
  "operations": {
    "backward": {
      "description": "Training backward kernels",
      "baseBindings": [
        { "index": 0, "name": "uniforms", "type": "uniform" },
        { "index": 1, "name": "input", "type": "read-only-storage" },
        { "index": 2, "name": "output", "type": "storage" }
      ],
      "baseUniforms": {
        "size": 16,
        "fields": [
          { "name": "size", "type": "u32", "offset": 0 }
        ]
      },
      "variants": {
        "embed_backward": {
          "wgsl": "backward/embed_backward.wgsl",
          "entryPoint": "main",
          "workgroup": [256, 1, 1],
          "requires": []
        },
        "matmul_backward": {
          "wgsl": "backward/matmul_backward.wgsl",
          "entryPoint": "main",
          "workgroup": [16, 16, 1],
          "requires": []
        },
        "attention_backward": {
          "wgsl": "backward/attention_backward.wgsl",
          "entryPoint": "main",
          "workgroup": [256, 1, 1],
          "requires": []
        }
      }
    }
  }
}
```

**Wrappers reference registry paths:**

```javascript
// src/gpu/kernels/backward/matmul_backward.js
import { getKernelConfig } from '../utils.js';

const config = getKernelConfig('backward/matmul_backward');
// config.wgsl resolves to 'backward/matmul_backward.wgsl'
```

### Buffer Usage Constants

Use existing Doppler buffer usage constants, not ad-hoc WebGPU enums:

```javascript
// Use from src/memory/buffer-pool.js
import { BufferUsage } from '../memory/buffer-pool.js';

// BufferUsage.STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
// BufferUsage.STORAGE_READ = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
// BufferUsage.UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
```

LoRA and gradient buffers use `BufferUsage.STORAGE` (read/write for optimizer updates).

---

## Forward Cache vs Recompute Decision

**Problem:** attention_backward needs softmax outputs from forward pass. Two options:

| Strategy | Memory | Compute | When to Use |
|----------|--------|---------|-------------|
| **Cache** | +340MB (seq=256, batch=2) | 1x | VRAM > 2GB available |
| **Recompute** | +0 | ~2x | Memory-constrained |

**Impact on memory estimates:**

```
seq=256, batch=2 with CACHE:
  Activations:     340 MB
  Attn weights:    340 MB  ← cached softmax outputs
  Gradients:       680 MB
  TOTAL:          ~1.9 GB  (was 1.6 GB)

seq=256, batch=2 with RECOMPUTE:
  Activations:     340 MB
  Attn weights:    0 MB    ← recomputed in backward
  Gradients:       680 MB
  TOTAL:          ~1.6 GB
```

**Decision:** Default to `recomputeForward: false` (cache) for seq≤256. Auto-switch to recompute if VRAM pressure detected. Configurable via `training.attention.recomputeForward`.

---

## Implementation Plan

### 1. Autograd Tape

**New files:**
- `src/training/autograd.js` - Tape implementation
- `src/training/autograd.d.ts` - Types
- `src/config/kernels/backward-registry.json` - Op→backward kernel mapping

**Responsibilities:**
- Track operations during forward pass
- Store references to inputs/outputs for backward
- Replay backward in reverse order using registry-defined kernels
- Handle in-place operations safely
- Integrate with buffer pool for gradient allocation

**Config-backed op registry (no ad-hoc strings):**

```json
// src/config/kernels/backward-registry.json
{
  "ops": {
    "embed": {
      "backward": "embed_backward",
      "grads": ["weight"],
      "notes": "Uses scatter_add for gradient accumulation"
    },
    "matmul": {
      "backward": "matmul_backward",
      "grads": ["input", "weight"],
      "requires_transpose": true
    },
    "rmsnorm": {
      "backward": "rmsnorm_backward",
      "grads": ["input", "gamma"]
    },
    "attention": {
      "backward": "attention_backward",
      "grads": ["q", "k", "v"],
      "notes": "Includes causal masking backward"
    },
    "softmax": {
      "backward": "softmax_backward",
      "grads": ["input"]
    },
    "silu": {
      "backward": "silu_backward",
      "grads": ["input"]
    },
    "gelu": {
      "backward": "gelu_backward",
      "grads": ["input"]
    },
    "rope": {
      "backward": "rope_backward",
      "grads": ["input"]
    },
    "scale": {
      "backward": "scale_backward",
      "grads": ["input"],
      "notes": "For LoRA alpha scaling"
    },
    "cross_entropy": {
      "backward": "cross_entropy_backward",
      "grads": ["logits"],
      "notes": "Loss function, returns scalar loss + logit gradients"
    }
  }
}
```

**Usage pattern (ops resolved from registry, not strings):**

```javascript
import { loadBackwardRegistry } from '../config/backward-registry-loader.js';

// Usage pattern - ops are registry keys, not magic strings
const tape = new AutogradTape(loadBackwardRegistry());
tape.watch(input);

const h1 = tape.record(OpType.MATMUL, matmul, [input, W1]);
const h2 = tape.record(OpType.RMSNORM, rmsnorm, [h1, gamma]);
const logits = tape.record(OpType.MATMUL, matmul, [h2, W_out]);

const loss = crossEntropy(logits, targets);
const grads = tape.backward(loss); // Uses registry to dispatch backward kernels
```

### 2. Backward Kernels

**New WGSL files (consistent `_backward` naming):**

| Kernel | File | Complexity | Notes |
|--------|------|------------|-------|
| embed_backward | `embed_backward.wgsl` | Medium | Uses scatter_add for sparse gradient accumulation |
| matmul_backward | `matmul_backward.wgsl` | Low | Reuse matmul with transpose |
| softmax_backward | `softmax_backward.wgsl` | Medium | |
| rmsnorm_backward | `rmsnorm_backward.wgsl` | Medium | Variance chain rule |
| attention_backward | `attention_backward.wgsl` | **High** | Causal mask + numerical stability (see below) |
| rope_backward | `rope_backward.wgsl` | Medium | sin/cos derivatives |
| gelu_backward | `gelu_backward.wgsl` | Low | |
| silu_backward | `silu_backward.wgsl` | Low | |
| scale_backward | `scale_backward.wgsl` | Low | For LoRA alpha/rank scaling |
| cross_entropy_backward | `cross_entropy_backward.wgsl` | Low | Loss gradient to logits |
| adam | `adam.wgsl` | Low | Optimizer step (not backward, but grouped here) |

**attention_backward detailed requirements (highest complexity kernel):**

1. **Causal mask backward:** Gradients must not flow through masked positions. Mask is applied before softmax, so backward must zero gradients for masked (future) positions.

2. **Numerical stability:** Forward uses `softmax(QK^T / √d - mask * 1e9)`. Backward must:
   - Recompute softmax outputs (or cache from forward)
   - Use stable softmax gradient: `dS = S * (dO - sum(S * dO))`
   - Handle the `1/√d` scaling in gradient flow

3. **Gradient outputs:** Must produce `dQ`, `dK`, `dV` from `dO` (output gradient):
   ```
   dV = softmax(QK^T)^T @ dO
   dS = dO @ V^T
   dQK = softmax_backward(dS, S)  # where S = softmax(QK^T/√d)
   dQ = dQK @ K / √d
   dK = dQK^T @ Q / √d
   ```

4. **Memory consideration:** May need to recompute attention weights rather than caching (memory vs compute tradeoff). Flash attention style recomputation could reduce memory at cost of ~2x compute.

**Complexity factors** to account for:
- attention_backward with causal masking is the highest complexity kernel
- embed_backward scatter_add path needs careful testing
- Each kernel needs .js wrapper + .d.ts types

**Key insight:** `matmul_backward` is just matmul with transposed operands:
```
dW = matmul(x.T, dy)  // gradient w.r.t. weights
dx = matmul(dy, W.T)  // gradient w.r.t. input
```

Existing `matmul.wgsl` variants can be reused with transpose flags.

### 3. LoRA Layer

**New files:**
- `src/training/lora.js` - LoRA adapter implementation
- `src/training/lora.d.ts` - Types
- `src/config/schema/lora.js` - LoRA config schema
- `src/config/schema/lora.d.ts`

**Buffer pool integration:** LoRA tensors use buffer pool with explicit dtype + usage flags. No per-step allocations.

```javascript
import { acquireBuffer, releaseBuffer, BufferUsage } from '../memory/buffer-pool.js';
import { createTensor, tensorBytes } from '../gpu/tensor.js';
import { getTrainingConfig } from '../config/training-defaults.js';

// LoRA: h = Wx + α(BA)x where B, A are low-rank adapters
class LoraAdapter {
  /**
   * @param {LoraAdapterConfig} config - From training config schema
   */
  constructor(config) {
    const { inDim, outDim, rank, alpha } = config;
    const { loraParams: dtype } = getTrainingConfig().training.precision;

    // Acquire from buffer pool with explicit dtype + Doppler's BufferUsage constants
    const aBytes = tensorBytes([inDim, rank], dtype);
    const bBytes = tensorBytes([rank, outDim], dtype);

    this.A = createTensor(
      acquireBuffer(aBytes, BufferUsage.STORAGE),  // Uses Doppler's constant
      dtype,
      [inDim, rank],
      'lora_A'
    );
    this.B = createTensor(
      acquireBuffer(bBytes, BufferUsage.STORAGE),  // Uses Doppler's constant
      dtype,
      [rank, outDim],
      'lora_B'
    );
    this.alpha = alpha;
    this.rank = rank;
  }

  forward(x, tape) {
    const down = tape.record(OpType.MATMUL, matmul, [x, this.A]);
    const up = tape.record(OpType.MATMUL, matmul, [down, this.B]);
    // Scale by alpha/rank as per LoRA paper
    return tape.record(OpType.SCALE, scale, [up, this.alpha / this.rank]);
  }

  dispose() {
    releaseBuffer(this.A.buffer);
    releaseBuffer(this.B.buffer);
  }
}
```

**LoRA placement (which layers get adapters):**

| Layer | LoRA Target | Rationale |
|-------|-------------|-----------|
| q_proj | Yes | Attention query transformation |
| k_proj | Yes | Attention key transformation |
| v_proj | Yes | Attention value transformation |
| o_proj | Yes | Attention output |
| gate_proj | Optional | FFN gating |
| up_proj | Optional | FFN expansion |
| down_proj | Optional | FFN contraction |

Default: q, k, v, o only (per gamma's train.py). Configurable via schema.

### 4. Optimizer Kernel

**New files:**
- `src/gpu/kernels/adam.wgsl`
- `src/training/optimizer.js`

```wgsl
// adam.wgsl - ~50 lines
@compute @workgroup_size(256)
fn adam_step(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let i = gid.x;
  let g = grads[i];

  // Update biased first moment
  m[i] = beta1 * m[i] + (1.0 - beta1) * g;

  // Update biased second moment
  v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

  // Bias correction
  let m_hat = m[i] / (1.0 - pow(beta1, f32(step)));
  let v_hat = v[i] / (1.0 - pow(beta2, f32(step)));

  // Update parameters
  params[i] = params[i] - lr * m_hat / (sqrt(v_hat) + eps);
}
```

### 5. Training Loop Integration

**New files:**
- `src/training/trainer.js` - High-level training API
- `src/training/index.js` - Exports

```javascript
// Doppler training primitive
async function trainStep(model, batch, config) {
  const tape = new AutogradTape();

  // Forward
  const logits = model.forward(batch.input, tape);
  const loss = crossEntropyLoss(logits, batch.targets);

  // Backward
  const grads = tape.backward(loss);

  // Clip
  const clipped = clipGradients(grads, config.maxNorm);

  // Update
  optimizer.step(model.loraParams(), clipped);

  return { loss: await loss.read() };
}
```

### 6. Data Pipeline

**New files:**
- `src/training/dataloader.js` - Batch preparation
- `src/training/checkpoint.js` - IndexedDB persistence

```javascript
// Browser-native data loading
class DataLoader {
  constructor(dataset, batchSize, shuffle = true) {
    this.dataset = dataset;
    this.batchSize = batchSize;
    this.shuffle = shuffle;
  }

  async *batches() {
    const indices = this.shuffle ? shuffled(this.dataset.length) : range(this.dataset.length);
    for (let i = 0; i < indices.length; i += this.batchSize) {
      const batch = indices.slice(i, i + this.batchSize);
      yield this.collate(batch);
    }
  }
}
```

### 7. Testing

**Test categories:**

| Category | Tests |
|----------|-------|
| Per-kernel numerical | Each backward kernel vs PyTorch reference |
| Gradient checking | Finite difference verification |
| End-to-end parity | Full training loop vs HuggingFace TRL |
| Memory stability | No leaks over 1000 steps |
| Precision | f16/f32 mixed precision correctness |

**New test files:**
- `tests/training/autograd.test.js`
- `tests/training/backward-kernels.test.js`
- `tests/training/lora.test.js`
- `tests/training/optimizer.test.js`
- `tests/training/e2e-parity.test.js`

---

## File Structure

**Style compliance:** Every `.js` has a matching `.d.ts`. All defaults in schema/config.

```
doppler/src/training/
├── index.js              # Exports
├── index.d.ts            # Types
├── autograd.js           # Tape implementation
├── autograd.d.ts
├── lora.js               # LoRA adapter
├── lora.d.ts
├── optimizer.js          # Adam wrapper
├── optimizer.d.ts
├── dataloader.js         # Batch preparation
├── dataloader.d.ts
├── checkpoint.js         # IndexedDB persistence
├── checkpoint.d.ts
├── trainer.js            # High-level training primitives
└── trainer.d.ts

doppler/src/config/schema/training.js         # Training config schema
doppler/src/config/schema/training.d.ts
doppler/src/config/schema/lora.js             # LoRA config schema
doppler/src/config/schema/lora.d.ts
doppler/src/config/schema/backward-registry.js    # Backward registry schema
doppler/src/config/schema/backward-registry.d.ts
doppler/src/config/training-defaults.js       # Default values (NOT in runtime code)
doppler/src/config/training-defaults.d.ts
doppler/src/config/backward-registry-loader.js    # Loader with validation
doppler/src/config/backward-registry-loader.d.ts

doppler/src/config/kernels/backward-registry.json # Backward op→kernel mapping
doppler/src/config/kernels/registry.json          # (existing, add training category entries)

doppler/src/gpu/kernels/
├── (existing forward kernels)
├── backward/                     # NEW directory for backward kernels
│   ├── embed_backward.wgsl       # Uses scatter_add internally
│   ├── embed_backward.js
│   ├── embed_backward.d.ts
│   ├── matmul_backward.wgsl
│   ├── matmul_backward.js
│   ├── matmul_backward.d.ts
│   ├── softmax_backward.wgsl
│   ├── softmax_backward.js
│   ├── softmax_backward.d.ts
│   ├── rmsnorm_backward.wgsl
│   ├── rmsnorm_backward.js
│   ├── rmsnorm_backward.d.ts
│   ├── attention_backward.wgsl   # Includes causal mask handling
│   ├── attention_backward.js
│   ├── attention_backward.d.ts
│   ├── rope_backward.wgsl
│   ├── rope_backward.js
│   ├── rope_backward.d.ts
│   ├── silu_backward.wgsl
│   ├── silu_backward.js
│   ├── silu_backward.d.ts
│   ├── gelu_backward.wgsl
│   ├── gelu_backward.js
│   ├── gelu_backward.d.ts
│   ├── scale_backward.wgsl       # For LoRA alpha scaling
│   ├── scale_backward.js
│   ├── scale_backward.d.ts
│   ├── cross_entropy_backward.wgsl
│   ├── cross_entropy_backward.js
│   ├── cross_entropy_backward.d.ts
│   ├── adam.wgsl
│   ├── adam.js
│   ├── adam.d.ts
│   ├── index.js                  # Backward kernel exports
│   └── index.d.ts
```

---

## Verification Criteria

Phase 1 complete when:

1. [ ] FunctionGemma 270M LoRA trains in Chrome on M1 Mac (8GB)
2. [ ] Loss decreases on toy dataset (overfitting test)
3. [ ] Exported RDRR-LoRA adapter loads in Doppler inference
4. [ ] Per-kernel tests pass against PyTorch reference
5. [ ] No memory leaks over 1000 training steps
6. [ ] Training speed: >5 samples/sec on M1 (seq=128, batch=1)

**Note on export interop:** RDRR-LoRA is Doppler-native. For Ollama/llama.cpp interop, a separate `rdrr-to-gguf` converter would be needed (out of scope for Phase 1, could be Phase 1.5).

---

## Future Phases

### Phase 2: Full Fine-tune
- Weight gradient kernels for base model
- Gradient checkpointing for memory efficiency
- Larger optimizer state buffers

### Phase 3: Per-Expert MoE LoRA
- Expert-aware LoRA routing
- Per-expert adapter storage
- Dynamic LoRA loading based on router decisions

### Phase 4: Distributed Training
- Gradient aggregation over P2P mesh
- Federated averaging for privacy
- Expert sharding across peers

---

## Dependencies

**Existing Doppler infrastructure used:**
- `memory/buffer-pool.js` - Gradient buffer allocation
- `tensor.js` - Tensor abstraction with dtype
- `kernel-runtime.js` - Kernel dispatch
- `profiler.js` - Performance monitoring
- `device.js` - WebGPU device management

**No new external dependencies required.**

---

## Open Questions

1. **Loss scaling:** Do we need dynamic loss scaling for f16, or is static sufficient for 270M?
   - *Recommendation:* Start with static scaling (1.0), add dynamic if instability observed.

2. **Reploid integration:** Should TraceStore auto-trigger training, or manual only?
   - *Recommendation:* Manual first, auto-trigger as Phase 1.5 enhancement.

3. **LoRA param quantization:** Train in f16 or f32?
   - *Recommendation:* Train in f32 for stability, quantize to f16 on export.

---

## Decisions Made

1. **Export format:** Custom RDRR-LoRA extension, not safetensors.
   - Safetensors requires external dependency or custom writer.
   - RDRR-LoRA keeps everything in Doppler's existing format ecosystem.
   - **Interop path (Phase 1.5):** Add `rdrr-to-gguf` converter for Ollama/llama.cpp if needed.

2. **Max batch size for 8GB:** seq=256, batch=2 is the practical limit (~1.6GB).
   - Leaves headroom for browser overhead and other tabs.

3. **Attention backward includes causal masking.**
   - FunctionGemma uses causal attention; backward must handle mask gradients correctly.
   - Not a separate kernel; integrated into attention_backward.wgsl.

---

## Minimal E2E Training Target

**Toy dataset for validation:**
- 100 FunctionGemma tool-call traces from gamma's training format
- Known loss curve: should reach <0.5 cross-entropy in 50 steps (overfitting test)
- If loss doesn't decrease, backward kernels have bugs

**Parity test:**
- Same 100 traces through gamma's Python train.py
- Compare loss curves step-by-step
- Tolerance: <5% relative difference at each step

**Performance target:**
- >5 samples/sec on M1 Mac (8GB) at seq=128, batch=1
- >2 samples/sec at seq=256, batch=2

---

*Created: January 2026*
*Status: Implemented (engine primitives)*
*Last updated: January 2026 (loss + clipping + adapter export)*


