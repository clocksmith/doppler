# Gemma 2 Incoherent Output (Activation Dtype Mismatch) Post-Mortem

**Status**: RESOLVED
**Date**: 2026-01-04
**Model**: `gemma-2-2b-it-wf16`
**Prompt**: `Explain why the sky is blue.`
**Expected**: Coherent multi-token explanation
**Actual (broken)**: `<unused17>` / short garbage tokens, occasional GPU copy warnings

---

## Summary

Two issues combined to produce incoherent output during Gemma 2 debugging:

1. **Activation dtype mismatch**: Kernel-path overrides forced `matmul_f16.wgsl` (F16 output) during prefill while the runtime activation dtype was still F32. The pipeline then treated the buffer as F32, leading to invalid copy sizes and zeroed logits.
2. **Kernel variant ambiguity**: Attention variant lookup ignored prefill/decode phase, so decode-only entrypoints could be selected during prefill (or vice versa), producing warnings and fallback behavior.

---

## Impact

- Debug runs with `--trace kernels` showed invalid GPU buffer copy ranges.
- Short outputs (1-2 tokens) with `<unused*>` tokens instead of coherent text.
- Kernel path overrides became unreliable due to ambiguous variant selection.

---

## Timeline

| Time (UTC) | Event |
|-----------:|-------|
| 2026-01-04 | Reproduced incoherent output with Gemma 2 debug preset. |
| 2026-01-04 | Observed `copyBufferToBuffer` size mismatch when F16 outputs were treated as F32. |
| 2026-01-04 | Identified phase-agnostic kernel variant selection for attention. |
| 2026-01-04 | Implemented dtype-safe kernel override selection and phase-aware variant picking. |
| 2026-01-04 | Set global activation dtype default to F16 and aligned Gemma 2 presets. |
| 2026-01-04 | Verified coherent output on Gemma 2 debug run. |

---

## Symptoms

- Warnings/errors about invalid buffer copy ranges during decode/prefill.
- Kernel path logs showed `matmul_f16` overrides in F32 pipelines.
- Output tokens were `<unused*>` or only 1-2 tokens long.

---

## Root Causes

### 1) Kernel override output dtype mismatch

**What happened**

- `gemma2-f16-f16a` prefill overrides used `matmul_f16.wgsl`.
- Runtime activation dtype defaulted to F32.
- The pipeline assumed F32 output buffers, but the kernel wrote F16.

**Why it broke output**

F16 outputs are half the byte size of F32. When the pipeline copied or interpreted the output as F32, it read past the end of buffers or copied too much, resulting in invalid command buffers and zero logits.

### 2) Phase-agnostic attention variant lookup

**What happened**

Multiple attention kernels share the same entrypoint names (e.g., `attention_small`) across prefill/decode WGSL files. The loader matched by entrypoint only, so decode variants could be selected for prefill or vice versa.

**Why it mattered**

Phase-specific kernels have different expected shapes, workgroup dimensions, and cache behavior. The mismatch triggered warnings and fallbacks, masking the real dtype bug.

---

## Fixes Implemented

1. **Phase-aware kernel variant selection**
   - Kernel lookup now uses `prefill_`/`decode_` phase hints when multiple entrypoint matches exist.

2. **Dtype-safe kernel path overrides**
   - Kernel path overrides are rejected if the variant output dtype does not match the requested output dtype.

3. **Align Gemma 2 kernel path defaults**
   - Prefill matmuls use `matmul_f16w_f32a.wgsl` when F32 activations are requested.

4. **Default F16 activations**
   - Runtime default `activationDtype` set to `f16` (schema + default preset).
   - Gemma 2 presets explicitly set `activationDtype: f16`.
   - Converter default `computePrecision` set to `f16`, populating `quantizationInfo.compute`.

---

## Verification

Command:

```bash
doppler --config <ref>
# <ref>: extends "debug", cli.command="debug", model="gemma-2-2b-it-wf16"
# runtime.inference.prompt="Explain why the sky is blue."
# runtime.inference.batching.maxTokens=16
# runtime.inference.chatTemplate.enabled=true
# runtime.shared.debug.trace.enabled=true
# runtime.shared.debug.trace.categories=["attn","logits"], layers=[0]
```

Observed:

- No invalid copy errors.
- Output begins: `The sky appears blue due to a phenomenon called Rayleigh scattering.`

---

## Preventative Actions

1. Add a validation step that asserts kernel output dtype matches the buffer dtype expected by the pipeline.
2. Extend kernel registry tests to cover prefill/decode variant selection.
3. Add a debug trace that logs activation dtype, selected kernel file, and entrypoint per matmul.
4. Ensure converter manifests emit `quantizationInfo.compute` by default (now `f16`).

---

*Last updated: January 2026*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `../style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.
