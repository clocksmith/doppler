# DOPPLER Formats

## RDRR Format

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
  "modelId": "gemma-2-2b-it-wq4k-ef16",
  "modelType": "transformer",
  "quantization": "Q4_K_M",
  "quantizationInfo": {
    "weights": "q4k",
    "embeddings": "f16",
    "lmHead": "f16",
    "variantTag": "wq4k-ef16"
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
  "inference": {
    "attention": {
      "queryPreAttnScalar": 256,
      "attnLogitSoftcapping": null,
      "slidingWindow": null,
      "queryKeyNorm": false
    },
    "normalization": {
      "rmsNormWeightOffset": true,
      "rmsNormEps": 1e-5,
      "postAttentionNorm": true,
      "preFeedforwardNorm": true,
      "postFeedforwardNorm": false
    },
    "ffn": {
      "activation": "gelu",
      "gatedActivation": true
    },
    "rope": {
      "ropeTheta": 10000,
      "ropeLocalTheta": null,
      "ropeScalingType": null,
      "ropeScalingFactor": 1.0,
      "yarnBetaFast": 32,
      "yarnBetaSlow": 1,
      "yarnOriginalMaxPos": 4096
    },
    "output": {
      "finalLogitSoftcapping": null,
      "tieWordEmbeddings": false,
      "scaleEmbeddings": true,
      "embeddingTranspose": false,
      "embeddingVocabSize": null
    },
    "layerPattern": { "type": "all_attention" },
    "chatTemplate": { "type": null, "enabled": false },
    "defaultKernelPath": "gemma2-q4k-dequant-f16a"
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
`inference` is required for all RDRR manifests. See "Manifest Inference (Required)" below.

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
  "filename": "shard_00000.bin",
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
    "weights": "q4k",
    "embeddings": "f16",
    "lmHead": "f16",
    "variantTag": "wq4k-ef16"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `weights` | string | Quantization for main weights (`q4k`, `f16`, etc.) |
| `embeddings` | string? | Quantization for embedding table |
| `lmHead` | string? | Quantization for LM head (if different from embeddings) |
| `vision` | string? | Vision encoder quantization (multimodal models) |
| `audio` | string? | Audio encoder quantization (speech models) |
| `tts` | string? | TTS decoder quantization |
| `projector` | string? | Cross-modal projector quantization |
| `kvCache` | string? | KV cache dtype hint (runtime, not storage) |
| `compute` | string? | Compute precision hint (runtime, not storage) |
| `variantTag` | string? | Canonical name suffix for modelId |

### Naming Convention

DOPPLER uses a concise naming convention that describes **storage only** (not runtime behavior):

```
{model-name}-w{weights}[-e{embeddings}][-h{head}][-v{vision}][-a{audio}][-t{tts}][-p{projector}]
```

**Component prefixes:**

| Prefix | Component | Description |
|--------|-----------|-------------|
| `w` | Weights | Transformer layer weights (required) |
| `e` | Embeddings | Token embedding table |
| `h` | Head | LM head / output projection |
| `v` | Vision | Vision encoder (ViT, SigLIP, CLIP) |
| `a` | Audio | Audio encoder (Whisper, wav2vec) |
| `t` | TTS | Text-to-speech decoder |
| `p` | Projector | Cross-modal projection layers |

**Quantization tokens:**

| Token | Description | Token | Description |
|-------|-------------|-------|-------------|
| `q4k` | Q4_K_M block quant | `f16` | Float16 |
| `q6k` | Q6_K block quant | `bf16` | BFloat16 |
| `q8_0` | Q8_0 quant | `f32` | Float32 |
| `i4` | Int4 | `fp8e4` | Float8 E4M3 |
| `i8` | Int8 | `fp8e5` | Float8 E5M2 |

**Examples:**

| Model ID | Description |
|----------|-------------|
| `gemma-2b-wq4k` | Weights Q4K, embeddings default to weights |
| `gemma-2b-wq4k-ef16` | Weights Q4K, embeddings F16 |
| `llama-8b-wq4k-ef16-hf16` | With explicit head quantization |
| `qwen2-vl-7b-wq4k-vf16-pf16` | Multimodal with vision + projector |
| `phi-3.5-mini-wq4k-ebf16` | BFloat16 embeddings |

**Adapter naming:**

For adapters (separate or merged), the naming extends with `+` or `~`:

| Pattern | Meaning |
|---------|---------|
| `+lora-{name}-{quant}r{rank}` | Standalone adapter (separate file) |
| `~lora-{name}-{quant}r{rank}` | Merged adapter (baked into weights) |

Examples:
- `gemma-2b-wq4k+lora-coding-f16r16` — Standalone coding adapter
- `gemma-2b-wq4k~lora-instruct-f16r32` — Instruct adapter merged in

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
| `optimizations` | object | Kernel path hint (lowest precedence) |
| `blake3Full` | string | Full-model BLAKE3 hash |
| `config` | object | Raw model config (HF/GGUF metadata) |
| `conversion` | object | Conversion provenance (source, command, quantization) |
| `metadata` | object | Arbitrary metadata blob |
| `defaultWeightLayout` | string | Default weight layout hint (`row`/`column`) |
| `adapterType` | string | Adapter type (`lora`, `qlora`) |
| `baseCompatibility` | string[] | Base model compatibility list for adapters |
| `mergedAdapter` | object | Merged adapter metadata (baked into weights) |
| `adapterConfig` | object | Adapter configuration (standalone adapter) |
| `provenance` | object | Provenance metadata for merged/frankenstein models |
| `baseModel` | string | Base model reference (legacy adapter hint) |
| `loraConfig` | object | LoRA metadata (legacy adapter hint) |

### optimizations Schema

The `optimizations` field provides a kernel path hint used for fallback selection.

```json
{
  "optimizations": {
    "kernelPath": "gemma2-q4k-fused-f16a"
  }
}
```

#### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `kernelPath` | string/object | Kernel path hint (see `docs/design/KERNEL_PATHS.md`) |

#### Precedence Rules (low → high)

1. manifest `optimizations.kernelPath`
2. manifest `inference.defaultKernelPath`
3. runtime config `runtime.inference.kernelPath`
4. per-run context override (pipeline context only)

See [EXECUTION_PIPELINE.md](../EXECUTION_PIPELINE.md#kernel-path-overrides) for how kernel paths integrate with capability-based kernel selection.

---

## Adapter Manifests (LoRA/QLoRA)

LoRA adapters use the adapter manifest schema in `src/adapters/adapter-manifest.js`
and are loaded by the adapter loader, not the RDRR parser. RDRR manifests may
carry adapter metadata fields, but adapters themselves are not RDRR bundles.

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

## Manifest Inference (Required)

RDRR manifests include an `inference` field with all model-specific runtime
parameters (attention, norms, RoPE, FFN, output). The converter writes this
at conversion time; runtime merges overrides. Missing fields fail validation.
Use `null` to explicitly disable features; `undefined` is invalid.

---

## Field Normalization

The on-disk `manifest.json` may vary in naming. At runtime, `formats/rdrr/manifest.js` normalizes:

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
npx tsx src/converter/node-converter.js model.gguf ./output-rdrr

# From Safetensors (HuggingFace format)
npx tsx src/converter/node-converter.js ./hf-model-dir ./output-rdrr --quantize q4_k_m
```

### Serving Models

```bash
# Serve converted model
npx tsx cli/commands/serve.js ./model-rdrr --port 8765

# Convert and serve in one step
npx tsx cli/commands/serve.js model.gguf
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

- `src/formats/rdrr/manifest.js`: Parser and validation
- `src/converter/core.js`: Platform-agnostic conversion types and functions
- `src/converter/writer.js`: Node.js writer for CLI conversion
- `src/browser/browser-converter.js`: Browser conversion with OPFS output
- `storage/shard-manager.js`: OPFS shard management
- `storage/downloader.js`: Resumable downloads

---

*Last updated: January 2026*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.


## RDRR LoRA Format

This document defines Doppler's LoRA adapter manifest format and its on-disk layout.
It is designed to be small, self-describing, and compatible with the runtime adapter loader.

Related:
- `src/adapters/adapter-manifest.js`
- `src/adapters/lora-loader.js`
- `src/training/export.js`

## Overview

An RDRR-LoRA adapter is a JSON manifest with optional inline tensors.

- Manifest: metadata + tensor list
- Tensors: LoRA matrices `A` and `B` stored inline (base64 or array)
- Loader: maps tensor names to layer/module weights at runtime

## Manifest Schema (summary)

Required fields:
- `id`: string
- `name`: string
- `baseModel`: string (model ID the adapter was trained for)
- `rank`: integer
- `alpha`: number
- `targetModules`: array of module names

Optional fields:
- `version`, `description`
- `checksum`, `checksumAlgorithm`
- `weightsFormat`, `weightsPath`, `weightsSize`
- `tensors`: inline tensor specs
- `metadata`

See full schema in `src/adapters/adapter-manifest.js`.

## Tensor Naming

Tensor names must follow:

```
layer.{L}.{module}.lora_{a|b}
```

Examples:
- `layer.0.q_proj.lora_a`
- `layer.12.o_proj.lora_b`

Module names map through `LORA_MODULE_ALIASES` in `src/inference/pipeline/lora-types.js`.

## Inline Tensor Spec

Each tensor entry includes:

```
{
  "name": "layer.0.q_proj.lora_a",
  "shape": [inDim, rank],
  "dtype": "f32",
  "base64": "...",  // or "data": [...]
  "opfsPath": "...",
  "url": "..."
}
```

Doppler currently loads `f32` tensors for LoRA. If your source is `f16`, convert to `f32` before export.

## Adapter Export

Use the training export helper:

```
import { exportLoRAAdapter } from '../src/training/export.js';
```

This creates a manifest with inline tensors that can be loaded by `adapter-manager`.

## GGUF Interop (Optional)

RDRR-LoRA is optimized for Doppler. If you need GGUF:

1. Export the adapter to JSON (inline tensors).
2. Convert tensors to a LoRA safetensors/npz format.
3. Use llama.cpp conversion tooling to emit GGUF.

An optional helper script is provided:

```
node tools/rdrr-lora-to-gguf.js --manifest adapter.json --out ./out
```

The script emits recommended conversion steps and paths, but does not run external tools.


## Adapter Manifest

Defines the JSON manifest format used by the LoRA/QLoRA adapter loader.
Adapters are not RDRR bundles; they use a separate schema and loader path.

Schema source: `src/adapters/adapter-manifest.js`

---

## Required Fields

```json
{
  "id": "gemma-3-1b-coding",
  "name": "Gemma 3 Coding Adapter",
  "baseModel": "gemma-3-1b",
  "rank": 16,
  "alpha": 32,
  "targetModules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "tensors": []
}
```

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | Slug/ID, `[a-zA-Z0-9_-]+` |
| `name` | string | Human-readable name |
| `baseModel` | string | Base model identifier |
| `rank` | number | LoRA rank (integer) |
| `alpha` | number | LoRA alpha scaling |
| `targetModules` | string[] | Modules to modify |

---

## Optional Fields

| Field | Type | Notes |
|-------|------|-------|
| `version` | string | SemVer, default `1.0.0` |
| `description` | string | Adapter description |
| `checksum` | string | SHA-256 or BLAKE3 hash |
| `checksumAlgorithm` | string | `sha256` (default) or `blake3` |
| `weightsFormat` | string | `safetensors`, `npz`, `json`, `binary` |
| `weightsPath` | string | Path/URL to weight file |
| `weightsSize` | number | Size in bytes |
| `tensors` | array | Inline tensor specs (see below) |
| `metadata` | object | Arbitrary metadata |

---

## Tensor Entries

Inline tensors are provided as objects in `tensors`. Each tensor must include
`name` and `shape`, and must have data in one of `data`, `base64`, `opfsPath`,
or `url`.

```json
{
  "name": "layers.0.q_proj.lora_a",
  "shape": [4096, 16],
  "dtype": "f32",
  "base64": "..."
}
```

| Field | Type | Notes |
|-------|------|-------|
| `name` | string | `layers.<idx>.<module>.lora_a|lora_b` |
| `shape` | number[] | 2D tensor shape |
| `dtype` | string | `f32`, `f16`, or `bf16` |
| `data` | number[] | Inline float data |
| `base64` | string | Base64-encoded buffer |
| `opfsPath` | string | Path in OPFS |
| `url` | string | URL to tensor data |

The loader normalizes module names using `LORA_MODULE_ALIASES` and skips
unknown tensors.

---

## Loading

Adapters are loaded via the adapter loader:

```javascript
import { loadLoRAFromUrl } from './adapters/lora-loader.js';

const adapter = await loadLoRAFromUrl('https://.../adapter.json');
```

Checksum verification runs when `checksum` is present and `skipVerify` is not
set in loader options.
