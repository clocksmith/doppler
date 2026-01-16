# Gemma 2 Inference Guide

This doc collects the config paths, default kernel selection, and benchmark
checklist for Gemma 2 models.

---

## Presets and Kernel Paths

Model preset (conversion-time):

- `src/config/presets/models/gemma2.json`
  - Sliding window + softcap defaults
  - Layer pattern: alternating, global=odd
  - Kernel path mapping for f16 and q4k

Kernel path presets:

- `src/config/presets/kernel-paths/gemma2-f16-f16a.json`
- `src/config/presets/kernel-paths/gemma2-f16-f32a.json`
- `src/config/presets/kernel-paths/gemma2-q4k-dequant-f16a.json`
- `src/config/presets/kernel-paths/gemma2-q4k-fused-f32a.json`
- `src/config/presets/kernel-paths/gemma2-q4k-fused-f16a.json` (experimental, not default)

Runtime presets (debug/pipeline):

- `src/config/presets/runtime/model/gemma2-pipeline.json`
- `src/config/presets/runtime/model/gemma2-pipeline-debug.json`
- `src/config/presets/runtime/model/gemma2-debug.json`

---

## Default Kernel Path Selection

The converter sets `manifest.inference.defaultKernelPath` using the model preset
kernel mapping. At runtime, the pipeline resolves kernel paths in this order:

1. `contexts.runtime.kernelPath` (explicit context override)
2. `runtime.inference.kernelPath` (runtime config)
3. `manifest.inference.defaultKernelPath` (from conversion)
4. `manifest.optimizations.kernelPath` (legacy)

If a Gemma 2 manifest predates kernel path support, it will fall back to auto
kernel selection. That can select fused kernels that are slower for Gemma 2
dimensions. Fix by reconverting or by setting `runtime.inference.kernelPath`
explicitly in a config preset.

---

## Unused / Experimental Configs

- `gemma2-q4k-fused-f16a` exists for experimentation but is not referenced by
  the Gemma 2 model preset (default q4k f16 uses dequant path).
- `gemma2-pipeline` and `gemma2-pipeline-debug` are opt-in runtime presets.

---

## Benchmark Checklist (Gemma 2)

### Kernel Changes

1. `npm test -- --filter matmul`
2. `npm test -- --filter rmsnorm`
3. `npm run bench -- --kernels --config bench`

### Inference Changes

1. `npm test -- --inference --config bench`
2. `npm run bench -- --config bench --model gemma-2-2b-it-wf16`

### Kernel Path A/B

Create a small runtime config that pins `runtime.inference.kernelPath`:

```json
{
  "runtime": {
    "inference": {
      "kernelPath": "gemma2-f16-f16a"
    }
  }
}
```

Then run:

```bash
npm run bench -- --config ./gemma2-f16a.json --model gemma-2-2b-it-wf16
```

Record results in `tests/results/` and update `tests/baselines.json` if the
numbers are stable.
