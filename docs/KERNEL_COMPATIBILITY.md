# Kernel Compatibility Matrix

This chart separates **packing/layout** (how weights are stored in RDRR) from **runtime kernel mode** (how they are executed). Packing controls storage and memory access; the kernel path selects the execution path and pipeline order.

## Runtime Kernel Modes (Overrides)

Use CLI overrides or config files to force runtime kernel mode (without repacking):

```bash
# Via config file (recommended for reproducibility)
npm run bench -- -m MODEL --config kernel-config.json

# Preset profile (fast/safe/debug/fused/apple)
npm run bench -- -m MODEL --kernel-profile fused

# Kernel path override (preset ID or inline JSON)
npm run bench -- -m MODEL --kernel-path q4k-fused
```

Example `kernel-config.json`:
```json
{
  "runtime": {
    "inference": {
      "kernelPath": "q4k-fused"
    }
  }
}
```

Priority (low to high): manifest `optimizations.kernelPath` → manifest `inference.defaultKernelPath` → runtime config `runtime.inference.kernelPath` → CLI `--kernel-path`.

Kernel path notes:
- Kernel paths are explicit dispatch sequences (see `docs/design/KERNEL_PATHS.md`).

## RDRR Layout vs Runtime Kernels

| RDRR Quantization | Layout Metadata | Runtime Kernel Mode | Requirements | Notes |
|---|---|---|---|---|
| F16 / BF16 | `defaultWeightLayout=row` or `column` | `f16-native` | `shader-f16` for F16 | Layout affects transpose; kernel path controls arithmetic. |
| Q4_K_M | `q4kLayout=row_wise` | `q4k-fused` or `q4k-dequant-f16/f32` | `subgroups` for fused; `shader-f16` for F16 | Row-wise layout is required for fused Q4K. |
| Q4_K_M | `q4kLayout=column_wise` | `q4k-dequant-f16/f32` | `shader-f16` for F16 | Column-wise packs are **not** fused-compatible. |
| Q4_K_M | `q4kLayout=flat` | `q4k-dequant-f16/f32` | `shader-f16` for F16 | Flat packing is legacy; no fused kernel. |
| MXFP4 | N/A | dequant + matmul (no dedicated kernel path yet) | `shader-f16` for F16 | Used for MoE experts; no fused matmul yet. |
| Q8_0 / Q8_K | N/A | dequant + matmul (planned) | `shader-f16` for F16 | Loader runtime kernels are planned; treat as packing only today. |

## OPFS Purge Helper

Manifest updates in OPFS require a purge to take effect:

```bash
npx tsx doppler/tools/purge-opfs.ts --model gemma-1b-q4-row
```

This removes the cached model directory from OPFS for the current browser profile.

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes, CLI flags (`--kernel-path`, `--kernel-profile`), and the OPFS purge helper.
