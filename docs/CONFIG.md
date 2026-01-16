# DOPPLER Config

## Kernel Paths

Kernel paths are explicit, ordered specifications of which GPU kernels execute during inference.

## Overview

A kernel path defines:
- **Which kernels** run for each operation
- **In what order** they execute
- **With what configuration** (entry points, override constants)
- **Activation dtype** for the path (`activationDtype`, required)

This replaces the implicit `q4kStrategy` and `fusedFFNQ4K` configuration flags with fully declarative paths.

## Path Structure

```json
{
  "id": "gemma2-q4k-fused-f16a",
  "name": "Gemma 2 Q4K Fused F16A",
  "description": "Q4K weights with fused dequant+matmul using F16 activations",
  "activationDtype": "f16",

  "decode": {
    "steps": [
      { "op": "input_norm", "kernel": "rmsnorm_f16.wgsl", "entry": "main" },
      { "op": "q_proj", "kernel": "fused_matmul_q4_multicol_f16a.wgsl", "entry": "main_multicol_f16a" },
      ...
    ]
  },

  "prefill": {
    "steps": [...]
  },

  "preLayer": [...],
  "postLayer": [...],
  "sampling": [...]
}
```

`activationDtype` is required. Kernel paths must declare the activation dtype they expect so runtime dtypes stay consistent.

## Step Schema

Each step specifies a single kernel dispatch:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `op` | string | Yes | Logical operation name (for tracing) |
| `kernel` | string | Yes | Kernel file name |
| `entry` | string | No | Entry point (default: `main`) |
| `constants` | object | No | Override constants for pipeline creation |
| `weights` | string | No | Weight buffer reference (template: `layer.{L}.name`) |
| `input` | string | No | Input buffer slot (default: `hidden_state`) |
| `output` | string | No | Output buffer slot (default: `hidden_state`) |

## Built-in Paths

### Q4K Models

| Path | Description | Performance | Accuracy |
|------|-------------|-------------|----------|
| `gemma2-q4k-fused-f16a` | Fused dequant+matmul with F16 activations | Best (F16) | Good |
| `gemma2-q4k-fused-f32a` | Fused dequant+matmul with F32 activations | Best (F32) | Good |
| `gemma2-q4k-dequant-f16a` | Pre-dequant to F16 with F16 activations | Balanced | Good |
| `gemma2-q4k-dequant-f32a` | Pre-dequant to F32 with F32 activations | Slower | Best |

### F16 Models

| Path | Description |
|------|-------------|
| `gemma2-f16-f16a` | F16 weights with F16 activations |
| `gemma2-f16-f32a` | F16 weights with F32 activations |

Note: `activationDtype` is explicit in the path. When a kernel path is selected,
the pipeline aligns `runtime.inference.compute.activationDtype` and
`runtime.inference.kvcache.kvDtype` to the path so kernel dtypes stay consistent.
For LM head overrides, `lm_head_prefill` can be added to `postLayer` to supply a
batched matmul kernel for prefill while keeping `lm_head` on GEMV for decode.

## Path Comparison

### Q4K Fused (13 dispatches/layer)
```
input_norm → q_proj → k_proj → v_proj → rope_q → rope_k →
attention → o_proj → attn_residual → post_attn_norm →
ffn_gate_up → down_proj → ffn_residual
```

### Q4K Dequant F32 (15 dispatches/layer)
```
input_norm → q_proj → k_proj → v_proj → rope_q → rope_k →
attention → o_proj → attn_residual → post_attn_norm →
gate_proj → up_proj → activation → down_proj → ffn_residual
```

The fused path has 2 fewer dispatches because:
- `ffn_gate_up` fuses: gate_proj + up_proj + activation

Note: `gemma2-q4k-fused-f16a` uses separate gate/up matmuls because `fused_ffn_q4k` requires F32 activations, so expect 2 extra dispatches vs the fused FFN path.

## Usage

### In Model Preset

Use `kernelPaths` to select a `defaultKernelPath` at conversion time based on weight
quantization and activation dtype (from `quantizationInfo.compute`):

```json
{
  "id": "gemma2",
  "inference": {
    "kernelPaths": {
      "f16": {
        "f16": "gemma2-f16-f16a",
        "f32": "gemma2-f16-f32a"
      },
      "q4k": {
        "f16": "gemma2-q4k-fused-f16a",
        "f32": "gemma2-q4k-fused-f32a"
      }
    }
  }
}
```

### In Runtime Config

```json
{
  "inference": {
    "kernelPath": "gemma2-q4k-fused-f16a"
  }
}
```

### CLI

Kernel selection is config-only; CLI flags must not set kernel paths.

## Creating Custom Paths

1. Copy an existing preset from `src/config/presets/kernel-paths/`
2. Set `activationDtype` (`f16` or `f32`)
3. Modify the steps as needed
4. Register in `src/config/kernel-path-loader.js`

### Override Constants

Constants are set at pipeline creation (compile-time eliminated):

```json
{
  "op": "input_norm",
  "kernel": "rmsnorm.wgsl",
  "constants": {
    "RMS_NORM_OFFSET": true,
    "HIDDEN_SIZE": 2304
  }
}
```

### Weight References

Use `{L}` template for layer-specific weights:

```json
{
  "op": "q_proj",
  "kernel": "matmul_f32.wgsl",
  "weights": "layer.{L}.self_attn.q_proj"
}
```

At layer 5, this resolves to `layer.5.self_attn.q_proj`.

## Migration from q4kStrategy

| Old Config | New Kernel Path |
|------------|-----------------|
| `q4kStrategy: "fused_q4k"` + `fusedFFNQ4K: true` | `gemma2-q4k-fused-f16a` (F16 activations) or `gemma2-q4k-fused-f32a` (F32 activations) |
| `q4kStrategy: "dequant_f32"` | `gemma2-q4k-dequant-f32a` |
| `q4kStrategy: "dequant_f16"` | `gemma2-q4k-dequant-f16a` |

## See Also

- [WGSL Style Guide](../style/WGSL_STYLE_GUIDE.md) - Entry points vs override constants
- [Kernel Registry](../../src/config/kernels/registry.json) - All available kernels


## Error Codes

Doppler errors carry a stable code in the message prefix and on the error
object when created via `createDopplerError()`. Format:

```
[DOPPLER_*] Message text...
```

Use these codes in tests and user-facing error handling.

## Config Errors

- `DOPPLER_CONFIG_PRESET_UNKNOWN`
  - Unknown runtime preset requested by id.

## GPU Errors

- `DOPPLER_GPU_UNAVAILABLE`
  - WebGPU is not available in the current browser/worker context.
- `DOPPLER_GPU_DEVICE_FAILED`
  - Adapter/device creation failed.

## Loader Errors

- `DOPPLER_LOADER_MANIFEST_INVALID`
  - RDRR manifest failed validation.
- `DOPPLER_LOADER_SHARD_INDEX_INVALID`
  - Requested shard index does not exist in the manifest.
