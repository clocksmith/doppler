# Kernel Paths

Kernel paths are explicit, ordered specifications of which GPU kernels execute during inference.

## Overview

A kernel path defines:
- **Which kernels** run for each operation
- **In what order** they execute
- **With what configuration** (entry points, override constants)

This replaces the implicit `q4kStrategy` and `fusedFFNQ4K` configuration flags with fully declarative paths.

## Path Structure

```json
{
  "id": "gemma2-q4k-fused",
  "name": "Gemma 2 Q4K Fused",
  "description": "Q4K weights with fused dequant+matmul",

  "decode": {
    "steps": [
      { "op": "input_norm", "kernel": "rmsnorm.wgsl", "entry": "main" },
      { "op": "q_proj", "kernel": "fused_matmul_q4.wgsl", "entry": "main_multicol" },
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
| `gemma2-q4k-fused` | Fused dequant+matmul, fused FFN | Best | Good |
| `gemma2-q4k-dequant-f32` | Pre-dequant to F32, separate matmuls | Slower | Best |

### F16 Models

| Path | Description |
|------|-------------|
| `gemma2-f16-native` | Native F16 weights, no dequantization |

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

## Usage

### In Model Preset

```json
{
  "id": "gemma2",
  "inference": {
    "kernelPath": "gemma2-q4k-dequant-f32"
  }
}
```

### In Runtime Config

```json
{
  "inference": {
    "kernelPath": "gemma2-q4k-fused"
  }
}
```

### CLI

```bash
npm run debug -- -m MODEL --kernel-path gemma2-q4k-fused
```

## Creating Custom Paths

1. Copy an existing preset from `src/config/presets/kernel-paths/`
2. Modify the steps as needed
3. Register in `src/config/kernel-path-loader.js`

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
| `q4kStrategy: "fused_q4k"` + `fusedFFNQ4K: true` | `gemma2-q4k-fused` |
| `q4kStrategy: "dequant_f32"` | `gemma2-q4k-dequant-f32` |
| `q4kStrategy: "dequant_f16"` | `gemma2-q4k-dequant-f16` (not recommended) |

## See Also

- [WGSL Style Guide](../style/WGSL_STYLE_GUIDE.md) - Entry points vs override constants
- [Kernel Registry](../../src/config/kernels/registry.json) - All available kernels
