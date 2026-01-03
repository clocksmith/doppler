# Kernel Compatibility Matrix

This chart separates **packing/layout** (how weights are stored in RDRR) from **runtime kernel mode** (how they are executed). Packing controls storage and memory access; kernel hints select the execution path.

## Runtime Kernel Modes (Overrides)

Use CLI overrides or config files to force runtime kernel mode (without repacking):

```bash
# Via config file (recommended for reproducibility)
npm run bench -- -m MODEL --config kernel-config.json

# Shorthand CLI flag
npm run bench -- -m MODEL --force-fused-q4k

# JSON kernel hints via CLI (merged with config/flags)
npm run bench -- -m MODEL --kernel-hints '{"q4kMatmul":"fused_q4k","computePrecision":"f16"}'
```

Example `kernel-config.json`:
```json
{
  "runtime": {
    "kernelHints": {
      "q4kMatmul": "fused_q4k",
      "computePrecision": "f16"
    }
  }
}
```

Priority (low to high): config file → `--kernel-hints` JSON → individual CLI flags (e.g., `--force-fused-q4k`).

Supported runtime hints:
- `computePrecision`: `auto | f16 | f32`
- `q4kMatmul`: `auto | fused_q4k | dequant_f16 | dequant_f32`
- `f16Matmul`: `auto | gemv_subgroup`
- `attentionPrefill`: `auto | tiled_large | tiled_small | streaming`
- `attentionDecode`: `auto | tiled_large | tiled_small | streaming`

## RDRR Layout vs Runtime Kernels

| RDRR Quantization | Layout Metadata | Runtime Kernel Mode | Requirements | Notes |
|---|---|---|---|---|
| F16 / BF16 | `defaultWeightLayout=row` or `column` | `f16Matmul` / `f32` | `shader-f16` for F16 | Layout affects transpose; kernel mode controls arithmetic. |
| Q4_K_M | `q4kLayout=row_wise` | `fused_q4k` or `dequant_f16/f32` | `subgroups` for fused; `shader-f16` for F16 | Row-wise layout is required for fused Q4K. |
| Q4_K_M | `q4kLayout=column_wise` | `dequant_f16/f32` | `shader-f16` for F16 | Column-wise packs are **not** fused-compatible. |
| Q4_K_M | `q4kLayout=flat` | `dequant_f16/f32` | `shader-f16` for F16 | Flat packing is legacy; no fused kernel. |
| MXFP4 | N/A | `dequant_mxfp4` (then F16/F32 matmul) | `shader-f16` for F16 | Used for MoE experts; no fused matmul yet. |
| Q8_0 / Q8_K | N/A | `dequant_f16/f32` (planned) | `shader-f16` for F16 | Loader runtime kernels are planned; treat as packing only today. |

## OPFS Purge Helper

Manifest updates in OPFS require a purge to take effect:

```bash
npx tsx doppler/tools/purge-opfs.ts --model gemma-1b-q4-row
```

This removes the cached model directory from OPFS for the current browser profile.

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

