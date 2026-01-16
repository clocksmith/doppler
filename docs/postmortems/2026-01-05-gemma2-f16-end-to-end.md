# Gemma 2 F16 End-to-End Pipeline Post-Mortem

**Status**: RESOLVED
**Date**: 2026-01-05
**Models**: `gemma-2-2b-it-wf16`, `gemma-2-2b-it-wq4k-ef16`
**Prompt**: `Explain why the sky is blue.`
**Expected**: Coherent multi-token response with F16 activations
**Actual (broken)**: Short garbage tokens (`<unused*>`), dtype mismatch warnings, inconsistent kernel selection

---

## Summary

F16 activations were enabled, but critical pieces of the pipeline still assumed F32. Attention, RoPE, logits, and sampling used F32 paths or mismatched dtype handling, and kernel-path selection did not account for activation dtype. The result was incoherent output and hidden dtype mismatches.

---

## Impact

- Incoherent output on Gemma 2 F16 and Q4K runs.
- Kernel traces showed fallbacks and mismatched variants.
- F16 activation benefits (memory/bandwidth) were not realized end-to-end.

---

## Root Causes

1. **Missing F16 variants in attention/rope/sampling selection**
   - Attention and RoPE selections were keyed on KV dtype only.
   - Sampling always assumed F32 logits.

2. **Logits dtype not plumbed through GPU sampling**
   - F16 logits were read back or sampled as F32, causing size mismatches and garbage.

3. **Kernel-path defaults ignored activation dtype**
   - F16 models could still route to F32 kernel paths, leaving mixed precision in place.
4. **KV cache forced to F32 under attention softcapping**
   - Gemma 2 softcapping flipped KV cache to F32, leaving Q/K/V in F32 while F16 attention kernels were selected.

---

## Fixes Implemented

1. **F16 attention + RoPE + sampling kernels**
   - Added f16 variants and selection logic based on Q/K/V dtype.

2. **Logits dtype propagation**
   - `computeLogitsGPU` now returns `logitsDtype`.
   - GPU sampling and readbacks respect F16 vs F32 logits.

3. **Kernel-path defaults are explicit**
   - Added `gemma2-f16-f16a`, `gemma2-f16-f32a`, `gemma2-q4k-fused-f16a`.
   - Converter populates `inference.defaultKernelPath` from `kernelPaths` + `quantizationInfo.compute`.

4. **Manifest defaults updated**
   - `quantizationInfo.compute` defaults to `f16` in existing manifests.
5. **Softcap KV cache is configurable**
   - Added `runtime.inference.kvcache.forceF32Softcap` to optionally force F32.
6. **Decode buffers respect activation dtype**
   - Decode buffer allocation now matches `runtime.inference.compute.activationDtype`.

---

## Verification

Run:

```bash
npm run debug -- --model gemma-2-2b-it-wf16 --max-tokens 8 --chat \
  --text "Explain why the sky is blue." --trace kernels --gpu-profile

npm run debug -- --model gemma-2-2b-it-wq4k-ef16 --max-tokens 8 --chat \
  --text "Explain why the sky is blue." --trace kernels --gpu-profile
```

Expected:
- Coherent multi-token output
- No dtype mismatch warnings
- Kernel trace shows F16 attention/logits path

Observed (F16 native):
- Coherent output: `Here's the breakdown of why the`
- Kernel trace: `attention_small_f16`, `sample_f16`, F16 matmul variants
- Decode profile dominated by matmul (~70%) and attention (~25-30%)

Observed (Q4K fused F16A):
- Coherent output: `The sky appears blue because of a phenomenon`
- Kernel trace: `fused_matmul_q4_*_f16a`, `attention_small_f16`, `sample_f16`
- Decode profile dominated by matmul (~87-89%) and attention (~10-12%)

---

## Preventative Actions

1. Add a runtime assert for logits dtype vs sampling kernel selection.
2. Extend kernel-path selection tests to include activation dtype.
3. Add a debug trace that prints logits dtype and sample kernel variant.

---

*Last updated: January 2026*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `../style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.
