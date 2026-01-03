# DOPPLER RDRR Format Specification

Defines the Recursive DOPPLER Runtime Registry (RDRR) format for streaming model delivery optimized for browser-based LLM inference.

See also: `docs/ARCHITECTURE.md` for system overview.

---

## Goals

- Enable streaming download and incremental model loading in browsers
- Support P2P distribution with per-shard integrity verification
- Provide browser-native storage via OPFS (Origin Private File System)
- Support multiple quantization formats (Q4_K_M, F16, F32)
- Enable component-level grouping for hot-swap capability (v1)
- Support multiple architectures: transformer, mamba, rwkv, jamba, mixtral, deepseek

---

## Core Concepts

### RDRR

**RDRR** = **R**ecursive **D**OPPLER **R**untime **R**egistry

A streaming model delivery format bridging REPLOID (agent sandbox) and DOPPLER (WebGPU runtime).

| Component | Description |
|-----------|-------------|
| **Recursive** | Self-referential structure supporting nested model components |
| **DOPPLER** | The DOPPLER inference engine (WebGPU runtime) |
| **Runtime** | Execution-ready quantized weights |
| **Registry** | Manifest-based tensor addressing and shard management |

### Sharding

Models are split into fixed-size shards (default 64MB) for streaming download and P2P distribution. Each shard is independently verifiable.

### Manifest

A JSON file describing model metadata, shard layout, and tensor locations within shards.

---

## File Structure

```
model-name-rdrr/
├── manifest.json       # ~5KB - Model metadata + component groups
├── tensors.json        # ~40KB - Tensor locations (external)
├── tokenizer.json      # Tokenizer data
├── shard_00000.bin     # 64MB shards (flat naming)
├── shard_00001.bin
└── ...
```

The v1 format separates tensor locations into `tensors.json` to keep the manifest small (~5KB vs ~68KB).

---

## Manifest Schema

### Required Fields (v1)

```json
{
  "version": 1,
  "modelId": "gemma-2-2b-it-q4",
  "modelType": "transformer",
  "quantization": "Q4_K_M",
  "quantizationInfo": {
    "weights": "q4_k_m",
    "embeddings": "f16",
    "lmHead": "f16",
    "variantTag": "wq4_k_m-embf16"
  },
  "hashAlgorithm": "sha256",
  "architecture": {
    "numLayers": 26,
    "hiddenSize": 2304,
    "intermediateSize": 9216,
    "numAttentionHeads": 8,
    "numKeyValueHeads": 4,
    "headDim": 256,
    "vocabSize": 256000,
    "maxSeqLen": 8192,
    "ropeTheta": 10000
  },
  "groups": {
    "embed": { "type": "embed", "version": "1.0.0", "shards": [0], "tensors": ["model.embed_tokens.weight"], "hash": "..." },
    "layer.0": { "type": "layer", "version": "1.0.0", "shards": [0, 1], "tensors": [...], "hash": "...", "layerIndex": 0 },
    "head": { "type": "head", "version": "1.0.0", "shards": [10], "tensors": [...], "hash": "..." }
  },
  "shards": [...],
  "tensorsFile": "tensors.json",
  "tensorCount": 340,
  "totalSize": 3400000000
}
```

### Model Types

| Type | Description | Group Pattern |
|------|-------------|---------------|
| `transformer` | Dense transformer (Llama, Gemma, etc.) | `embed`, `layer.N`, `head` |
| `mamba` | Pure Mamba SSM | `embed`, `layer.N` (type=mamba), `head` |
| `rwkv` | RWKV architecture | `embed`, `layer.N` (type=rwkv), `head` |
| `mixtral` | MoE transformer | `embed`, `layer.N.shared`, `layer.N.expert.M`, `head` |
| `deepseek` | MoE with shared experts | + `layer.N.shared_expert` |
| `jamba` | Hybrid Mamba + Attention + MoE | `layer.N.attn`, `layer.N.mamba`, + experts |

### Component Groups

Groups enable per-component hot-swap and version tracking:

```json
{
  "groups": {
    "embed": {
      "type": "embed",
      "version": "1.0.0",
      "shards": [0],
      "tensors": ["model.embed_tokens.weight"],
      "hash": "sha256-of-group-data"
    },
    "layer.0": {
      "type": "layer",
      "version": "1.0.0",
      "shards": [0, 1],
      "tensors": ["model.layers.0.self_attn.q_proj.weight", ...],
      "hash": "...",
      "layerIndex": 0
    },
    "layer.0.expert.0": {
      "type": "expert",
      "version": "1.0.0",
      "shards": [5],
      "tensors": ["model.layers.0.mlp.experts.0.gate_proj.weight", ...],
      "hash": "...",
      "layerIndex": 0,
      "expertIndex": 0
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `embed`, `layer`, `head`, `expert`, `shared`, `mamba`, `rwkv`, `attn` |
| `version` | string | Semantic version for hot-swap tracking (e.g., `"1.0.0"`) |
| `shards` | number[] | Indices into manifest.shards[] |
| `tensors` | string[] | Tensor names in this group |
| `hash` | string | SHA256 hash of concatenated tensor data |
| `layerIndex` | number? | Layer index (for layer/expert groups) |
| `expertIndex` | number? | Expert index within layer |

### Shard Entry

```json
{
  "index": 0,
  "fileName": "shard_00000.bin",
  "size": 67108864,
  "hash": "sha256-hex-64-chars",
  "hashAlgorithm": "sha256"
}
```

### External tensors.json

Tensor locations are stored in a separate `tensors.json` file:

```json
{
  "model.embed_tokens.weight": {
    "group": "embed",
    "shard": 0,
    "offset": 0,
    "size": 1179648000,
    "shape": [256000, 2304],
    "dtype": "BF16"
  },
  "model.layers.0.self_attn.q_proj.weight": {
    "group": "layer.0",
    "shard": 5,
    "offset": 0,
    "size": 2654208,
    "shape": [2048, 2304],
    "dtype": "Q4_K_M"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `group` | string | Component group ID this tensor belongs to |
| `shard` | number | Primary shard index |
| `offset` | number | Byte offset within shard |
| `size` | number | Total size in bytes |
| `shape` | number[] | Tensor dimensions |
| `dtype` | string | Data type (Q4_K_M, F16, BF16, F32, etc.) |
| `layout` | string? | `"row"` (default) or `"column"` (pre-transposed) |
| `spans` | array? | For multi-shard tensors (see below) |

### Quantization Metadata (Optional)

`quantizationInfo` provides structured precision details per weight group so
embeddings and lm_head can be distinguished from core weights.

```json
{
  "quantizationInfo": {
    "weights": "q4_k_m",
    "embeddings": "f16",
    "lmHead": "f16",
    "variantTag": "wq4_k_m-embf16"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `weights` | string | Quantization for main weights (`q4_k_m`, `f16`, etc.) |
| `embeddings` | string? | Quantization for embedding table |
| `lmHead` | string? | Quantization for LM head (if different from embeddings) |
| `activations` | string? | Activation precision (optional) |
| `kvCache` | string? | KV cache dtype (optional) |
| `compute` | string? | Compute precision hint (optional) |
| `variantTag` | string? | Canonical name suffix (`wq4_k_m-embf16`) used by converter defaults |

### Multi-Shard Tensors

For tensors spanning multiple shards, use the `spans` field:

```json
{
  "model.embed_tokens.weight": {
    "group": "embed",
    "shard": 0,
    "offset": 0,
    "size": 123456789,
    "shape": [128256, 4096],
    "dtype": "BF16",
    "spans": [
      { "shardIndex": 0, "offset": 0, "size": 67108864 },
      { "shardIndex": 1, "offset": 0, "size": 56347925 }
    ]
  }
}
```

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `tokenizer` | object | Tokenizer configuration |
| `moeConfig` | object | Mixture-of-experts configuration |
| `quantizationInfo` | object | Structured quantization metadata (weights vs embeddings) |
| `runtimeOptimizations` | object | Hints for kernel selection |
| `blake3Full` | string | Full-model BLAKE3 hash |
| `adapterType` | string | Adapter type (e.g. `lora`) |
| `baseModel` | string | Base model reference for adapters |
| `loraConfig` | object | LoRA metadata for adapter manifests |

### runtimeOptimizations Schema

The `runtimeOptimizations` field provides hints to the DOPPLER runtime for kernel selection and performance tuning. These hints influence but do not override capability-based decisions.

```json
{
  "runtimeOptimizations": {
    "preferredKernels": {
      "matmul": "q4_fused",
      "attention": "tiled_f16",
      "rmsnorm": "f16_subgroup"
    },
    "workgroupOverrides": {
      "matmul_f16": [128, 1, 1],
      "rmsnorm": [256, 1, 1]
    },
    "disableFeatures": ["subgroups"],
    "forceF32Accumulation": true,
    "attentionTier": "streaming",
    "targetDevice": "apple-m1"
  }
}
```

#### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `preferredKernels` | object | Map of operation → kernel variant name |
| `workgroupOverrides` | object | Map of kernel → `[x, y, z]` workgroup size |
| `disableFeatures` | string[] | GPU features to disable (e.g., `subgroups`, `shader-f16`) |
| `forceF32Accumulation` | boolean | Force F32 accumulators even when F16 available |
| `attentionTier` | string | Force attention tier: `"tiled"`, `"streaming"`, or `"basic"` |
| `targetDevice` | string | Hint for device-specific tuning (e.g., `"apple-m1"`, `"nvidia-rtx"`) |

#### Kernel Variant Names

Valid values for `preferredKernels`:

| Operation | Valid Variants |
|-----------|----------------|
| `matmul` | `f32`, `f16`, `gemv_f16`, `q4_fused`, `q4_fused_batched` |
| `attention` | `tiled_f16`, `streaming_f16`, `basic_f32` |
| `rmsnorm` | `f32`, `f16`, `f16_subgroup` |
| `dequant` | `q4k`, `q8`, `f16_passthrough` |
| `softmax` | `f32`, `f16_online` |

#### Precedence Rules

1. **GPU capabilities always win**: If hardware doesn't support F16, the `f16` variant won't be used regardless of hints
2. **Hints are advisory**: Runtime may ignore hints if they would cause correctness issues
3. **Auto-tuning overrides**: If auto-tuning is enabled, tuned workgroup sizes take precedence over `workgroupOverrides`

#### Example: Optimized for Apple Silicon

```json
{
  "runtimeOptimizations": {
    "preferredKernels": {
      "matmul": "f16",
      "attention": "tiled_f16"
    },
    "workgroupOverrides": {
      "matmul_f16": [64, 4, 1]
    },
    "targetDevice": "apple-m1"
  }
}
```

#### Example: Conservative Settings for Compatibility

```json
{
  "runtimeOptimizations": {
    "disableFeatures": ["subgroups", "shader-f16"],
    "forceF32Accumulation": true,
    "attentionTier": "basic"
  }
}
```

See [EXECUTION_PIPELINE.md](../EXECUTION_PIPELINE.md#rdrr-runtime-hints) for how these hints integrate with capability-based kernel selection.

---

## LoRA Adapter Extension

LoRA adapters can be stored as RDRR manifests with `adapterType: "lora"`. These manifests reference tensor deltas stored in shards and include LoRA metadata.

```json
{
  "version": "1.0",
  "modelId": "functiongemma-270m-react-lora",
  "modelType": "lora",
  "adapterType": "lora",
  "baseModel": "functiongemma-270m-it",
  "quantization": "F32",
  "loraConfig": {
    "rank": 32,
    "alpha": 64,
    "targetModules": ["q_proj", "k_proj", "v_proj", "o_proj"]
  },
  "shards": [
    { "index": 0, "fileName": "lora_weights.rdrr", "size": 15728640, "hash": "..." }
  ],
  "tensors": {
    "layers.0.q_proj.lora_a": {
      "shard": 0,
      "offset": 0,
      "size": 32768,
      "shape": [4096, 32],
      "dtype": "f32"
    },
    "layers.0.q_proj.lora_b": {
      "shard": 0,
      "offset": 32768,
      "size": 32768,
      "shape": [32, 4096],
      "dtype": "f32"
    }
  }
}
```

Notes:
- Tensor names follow `layers.<idx>.<module>.lora_a|lora_b`.
- `dtype` should be `f32` for LoRA weights.

---

## Design Principles

### Sharded for Streaming

- 64MB default shard size (configurable)
- 4KB alignment for optimal OPFS/disk I/O
- Supports streaming download and incremental loading

### Integrity Verification

- Per-shard hash (SHA-256 supported everywhere, BLAKE3 optional)
- Enables P2P distribution without trusting the source
- Peers can verify shard integrity independently

### Browser-Native

- Stored in OPFS (Origin Private File System)
- Compatible with WebGPU tensor loading
- No WASM file system emulation needed

### Quantization Support

| Format | Description |
|--------|-------------|
| Q4_K_M | 4-bit GGML k-quants |
| F16 | Half precision |
| F32 | Full precision |

---

## Field Normalization

The on-disk `manifest.json` may vary in naming. At runtime, `storage/rdrr-format.ts` normalizes:

| On-Disk | Normalized |
|---------|------------|
| `fileName` or `filename` | `filename` |
| `hash` or `blake3` | `hash` (with `blake3` alias) |
| missing `offset` | computed from previous shards |
| `hashAlgorithm` | inferred from manifest or shard entry |

This ensures compatibility across converter versions.

---

## Usage

### Converting Models

```bash
# From GGUF
npx tsx tools/convert-cli.ts model.gguf ./output-rdrr

# From Safetensors (HuggingFace format)
npx tsx tools/convert-cli.ts ./hf-model-dir ./output-rdrr --quantize q4_k_m
```

### Serving Models

```bash
# Serve converted model
npx tsx tools/serve-cli.ts ./model-rdrr --port 8765

# Convert and serve in one step
npx tsx tools/serve-cli.ts model.gguf
```

### Loading in Browser

```javascript
import { downloadModel, parseManifest, createPipeline } from 'doppler';

// Download model to OPFS
await downloadModel('http://localhost:8765');

// Load and create inference pipeline
const manifest = await loadManifestFromOPFS();
const pipeline = await createPipeline(parseManifest(manifest));

// Generate
for await (const token of pipeline.generate('Hello')) {
  console.log(token);
}
```

---

## Version History

| Version | Changes |
|---------|---------|
| 1 | Component groups, external tensors.json, multi-architecture support |
| 0.x | Initial format with inline tensors (deprecated) |

---

## Related Files

- `storage/rdrr-format.ts`: Parser and validation
- `tools/convert-core.ts`: Platform-agnostic conversion types and functions
- `tools/rdrr-writer.ts`: Node.js writer for CLI conversion
- `browser/model-converter.ts`: Browser conversion with OPFS output
- `storage/shard-manager.ts`: OPFS shard management
- `storage/downloader.ts`: Resumable downloads

---

*Last updated: January 2026*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.
