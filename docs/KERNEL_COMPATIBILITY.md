# Kernel Compatibility Matrix

This chart separates **packing/layout** (how weights are stored in RDRR) from **runtime kernel mode** (how they are executed). Packing controls storage and memory access; the kernel path selects the execution path and pipeline order.

## Runtime Kernel Modes (Overrides)

Use config files to force runtime kernel mode (without repacking):

```bash
# Via config file (recommended for reproducibility)
npm run bench -- -m MODEL --config kernel-config.json

# Kernel path override (preset ID)
npm run bench -- -m MODEL --config kernel-config.json
```

Example `kernel-config.json`:
```json
{
  "runtime": {
    "inference": {
      "kernelPath": "gemma2-q4k-fused-f16a"
    }
  }
}
```

Priority (low to high): manifest `optimizations.kernelPath` → manifest `inference.defaultKernelPath` → runtime config `runtime.inference.kernelPath`.

Kernel path notes:
- Kernel paths are explicit dispatch sequences (see `docs/design/KERNEL_PATHS.md`).

## RDRR Layout vs Runtime Kernels

| RDRR Quantization | Layout Metadata | Runtime Kernel Mode | Requirements | Notes |
|---|---|---|---|---|
| F16 / BF16 | `defaultWeightLayout=row` or `column` | `gemma2-f16-f16a` (F16 activations) or `gemma2-f16-f32a` (F32 activations) | `shader-f16` for F16 | Layout affects transpose; kernel path controls arithmetic. |
| Q4_K_M | `q4kLayout=row_wise` | `gemma2-q4k-fused-f16a`, `gemma2-q4k-fused-f32a`, or `gemma2-q4k-dequant-f16a/f32a` | `subgroups` for fused; `shader-f16` for F16 | Row-wise layout is required for fused Q4K. |
| Q4_K_M | `q4kLayout=column_wise` | `gemma2-q4k-dequant-f16a/f32a` | `shader-f16` for F16 | Column-wise packs are **not** fused-compatible. |
| Q4_K_M | `q4kLayout=flat` | `gemma2-q4k-dequant-f16a/f32a` | `shader-f16` for F16 | Flat packing is legacy; no fused kernel. |
| MXFP4 | N/A | dequant + matmul (no dedicated kernel path yet) | `shader-f16` for F16 | Used for MoE experts; no fused matmul yet. |
| Q8_0 / Q8_K | N/A | dequant + matmul (planned) | `shader-f16` for F16 | Loader runtime kernels are planned; treat as packing only today. |

## OPFS Purge Helper

Manifest updates in OPFS require a purge to take effect:

```bash
npx tsx doppler/tools/purge-opfs.js --model gemma-1b-q4-row
```

This removes the cached model directory from OPFS for the current browser profile.

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes and the OPFS purge helper.
