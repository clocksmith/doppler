# DOPPLER RDRR Format Specification

Defines the Recursive DOPPLER Runtime Registry (RDRR) format for streaming model delivery optimized for browser-based LLM inference.

See also: `docs/ARCHITECTURE.md` for system overview.

---

## Goals

- Enable streaming download and incremental model loading in browsers
- Support P2P distribution with per-shard integrity verification
- Provide browser-native storage via OPFS (Origin Private File System)
- Support multiple quantization formats (Q4_K_M, F16, F32)

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
├── manifest.json       # Model metadata and shard layout
├── shard_00000.bin     # 64MB shard (aligned to 4KB)
├── shard_00001.bin
├── shard_00002.bin
└── ...
```

---

## Manifest Schema

### Required Fields

```json
{
  "version": "1.0",
  "modelId": "sha256-hash-of-model",
  "modelType": "transformer",
  "architecture": "Llama3ForCausalLM",
  "quantization": "Q4_K_M",
  "hashAlgorithm": "sha256",
  "config": {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "vocab_size": 128256
  },
  "shards": [...],
  "tensors": {...},
  "totalSize": 3400000000,
  "tensorCount": 340
}
```

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

### Tensor Entry

```json
{
  "model.embed_tokens.weight": {
    "shard": 0,
    "offset": 0,
    "size": 123456789,
    "shape": [128256, 4096],
    "dtype": "Q4_K_M"
  }
}
```

### Multi-Shard Tensors

For tensors spanning multiple shards, use the `spans` field:

```json
{
  "model.embed_tokens.weight": {
    "shard": 0,
    "offset": 0,
    "size": 123456789,
    "shape": [128256, 4096],
    "dtype": "Q4_K_M",
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
| 1.0 | Initial release with Q4_K_M, F16, F32 support |

---

## Related Files

- `storage/rdrr-format.ts`: Parser and validation
- `tools/rdrr-writer.ts`: Writer for conversion
- `storage/shard-manager.ts`: OPFS shard management
- `storage/downloader.ts`: Resumable downloads

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.
