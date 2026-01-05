# Kernel Compatibility Matrix

This chart separates **packing/layout** (how weights are stored in RDRR) from **runtime kernel mode** (how they are executed). Packing controls storage and memory access; the kernel plan selects the execution path and pipeline order.

## Runtime Kernel Modes (Overrides)

Use CLI overrides or config files to force runtime kernel mode (without repacking):

```bash
# Via config file (recommended for reproducibility)
npm run bench -- -m MODEL --config kernel-config.json

# Preset profile (fast/safe/debug/fused/apple)
npm run bench -- -m MODEL --kernel-profile fused

# JSON kernel plan override (merged with config/profile)
npm run bench -- -m MODEL --kernel-plan '{"q4kStrategy":"fused_q4k","variants":{"attention":{"prefill":"tiled_small","decode":"streaming"}}}'
```

Example `kernel-config.json`:
```json
{
  "runtime": {
    "inference": {
      "kernelPlan": {
        "q4kStrategy": "fused_q4k",
        "variants": {
          "attention": {
            "prefill": "tiled_small",
            "decode": "streaming"
          }
        }
      }
    }
  }
}
```

Priority (low to high): `--kernel-profile` → config `runtime.inference.kernelPlan` → `--kernel-plan`.

Kernel plan fields:
- `q4kStrategy`: `auto | fused_q4k | dequant_f16 | dequant_f32`
- `strict`: `true | false` (throw on invalid variants vs warn + fallback)
- `variants.attention`: `default | prefill | decode | roles`
- `variants.matmul`: `default | roles`
- `variants.*`: operation-specific overrides (see `config/schema/kernel-plan.schema.ts`)
- `layerPipeline`: override per-layer op ordering (see `config/schema/inference.schema.ts`)

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
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--kernel-plan`, `--kernel-profile`), and the OPFS purge helper.
