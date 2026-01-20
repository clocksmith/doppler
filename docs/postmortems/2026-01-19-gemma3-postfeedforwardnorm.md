# Gemma 3 Post-Feedforward Norm Bug Post-Mortem

**Date**: 2026-01-19
**Status**: RESOLVED
**Severity**: Critical (complete model failure)
**Duration**: Multi-session investigation

## Executive Summary

Gemma 3 models produced gibberish output (token 138 = double-space repeated indefinitely) due to a missing `postFeedforwardNorm: true` flag in the model preset. The weight tensors existed but were not being used.

## Symptoms

- Model output: `"  "` repeated infinitely instead of meaningful text
- Prompt: "The color of the sky is"
- Expected: "blue" or similar coherent text
- Token 138 (double-space) had logit +14.1 instead of expected -5.5

## Investigation Timeline

### Phase 1: Tokenization Bug (Red Herring)

**Observation**: DOPPLER tokenized the prompt into 39 tokens; HuggingFace produced 15.

**Finding**: BPE merge loading failed silently - merges stored as `["a", "b"]` arrays but code expected `"a b"` strings.

**Fix**: Updated BPE merge loading to handle array format.

**Result**: Tokenization fixed, but model still outputs garbage.

**Verdict**: Real bug, but not THE bug.

### Phase 2: Probe Dtype Mismatch (Red Herring)

**Observation**: Debug probes showed `ffn_in` as all zeros during prefill.

**Finding**: Probes read F16 GPU buffers as F32, producing garbage values.

**Fix**: Added `dtype` parameter to probe calls in `sandwich.js`:
```javascript
// Before
await runProbes('ffn_in', ffnInput.buffer, { layerIdx, numTokens, hiddenSize, ... });

// After
await runProbes('ffn_in', ffnInput.buffer, { layerIdx, numTokens, hiddenSize, dtype: ffnInput.dtype, ... });
```

**Result**: Probes show valid intermediate values, but model still outputs garbage.

**Verdict**: Diagnostic bug, not model bug.

### Phase 3: Logits Analysis (Breakthrough)

**Observation**: After fixing probes, all intermediate values looked valid:

| Stage | Values | Status |
|-------|--------|--------|
| embed_out | `[0.978, -0.592, ...]` | Valid |
| attn_out | `[2.186, 7.113, ...]` | Valid |
| ffn_in | `[1.431, -0.023, ...]` | Valid |
| final_norm | `[-5.523, 2.201, ...]` | Valid |
| **logits[138]** | **+14.13** | WRONG |

**Key Finding**: Token 138 had logit +14.1 in DOPPLER but -5.5 in HuggingFace.

### Phase 4: Architecture Deep Dive (Root Cause Found)

**Discovery**: Manifest contained:
```json
"normalization": {
  "postAttentionNorm": true,
  "preFeedforwardNorm": true,
  "postFeedforwardNorm": false  // <-- WRONG
}
```

**But model has these tensors**:
```
model.layers.0.post_feedforward_layernorm.weight  <- EXISTS!
model.layers.1.post_feedforward_layernorm.weight  <- EXISTS!
...
```

**Research**: Gemma 3 uses "Peri-LN" (sandwich normalization) - normalizes both input AND output of FFN sublayers.

## Root Cause

The `gemma3.json` model preset was missing `postFeedforwardNorm: true`:

```json
// src/config/presets/models/gemma3.json (BEFORE)
"normalization": {
  "rmsNormWeightOffset": true,
  "rmsNormEps": 1e-6,
  "postAttentionNorm": true,
  "preFeedforwardNorm": true
  // postFeedforwardNorm NOT SET -> defaults to false
}
```

The default value in `inference-defaults.schema.js` is `false`. Without post-FFN norm, the residual stream accumulated unnormalized FFN outputs across 26 layers, causing numerical divergence.

## The Fix

### 1. Model Preset (`src/config/presets/models/gemma3.json`)

```json
"normalization": {
  "rmsNormWeightOffset": true,
  "rmsNormEps": 1e-6,
  "postAttentionNorm": true,
  "preFeedforwardNorm": true,
  "postFeedforwardNorm": true  // ADDED
}
```

### 2. Existing Manifest (`models/gemma-3-1b-it-wf16/manifest.json`)

```json
"postFeedforwardNorm": true  // Changed from false
```

## Verification

| Metric | Before | After |
|--------|--------|-------|
| Token 138 logit | +14.13 | -5.47 |
| HuggingFace reference | -5.5 | Match |
| Output | `"  "  "  "...` | `"The color of the sky is often described as **blue**..."` |
| Throughput | N/A | 8.7 tok/s |

## Preventive Measures Implemented

### 1. Tensor-Config Consistency Validator

Created `src/formats/rdrr/tensor-config-validator.js` that validates:
- `postFeedforwardNorm` flag matches presence of `post_feedforward_layernorm.weight`
- `preFeedforwardNorm` flag matches presence of `pre_feedforward_layernorm.weight`
- `postAttentionNorm` flag matches presence of `post_attention_layernorm.weight`
- `queryKeyNorm` flag matches presence of `q_norm/k_norm` weights

Integrated into `src/formats/rdrr/validation.js` - errors on mismatch.

### 2. Auto-Detection in Converter

Updated `src/converter/manifest-inference.js` to auto-detect normalization flags from tensor names:
- If `post_feedforward_layernorm.weight` exists, set `postFeedforwardNorm: true`
- Auto-detected values override preset defaults

## Files Modified

| File | Change |
|------|--------|
| `src/config/presets/models/gemma3.json` | Added `postFeedforwardNorm: true` |
| `models/gemma-3-1b-it-wf16/manifest.json` | Fixed `postFeedforwardNorm: true` |
| `src/inference/pipeline/ffn/sandwich.js` | Added dtype to probe calls |
| `src/formats/rdrr/validation.js` | Integrated tensor-config validation |
| `src/formats/rdrr/tensor-config-validator.js` | NEW - consistency validator |
| `src/converter/manifest-inference.js` | Added auto-detection from tensors |

## Key Learnings

### 1. Silent Defaults Are Dangerous

Default `postFeedforwardNorm: false` was reasonable for older architectures but became a trap for Gemma 3. Critical architecture flags should either:
- Have no default (require explicit setting)
- Be auto-detected from tensor presence

### 2. Trust the Math, Trace the Values

The breakthrough came from comparing specific logit values between DOPPLER and HuggingFace. Token 138 having +14 vs -5 was the smoking gun.

### 3. Probe Bugs Can Mask Real Bugs

We spent time investigating "all zeros" in FFN input (dtype mismatch). Always verify diagnostic tools first.

### 4. Multiple Bugs Can Coexist

The BPE merge bug was real and needed fixing, but wasn't causing the gibberish. Don't assume the first bug found is THE bug.

### 5. Tensor Names Are Ground Truth

If `post_feedforward_layernorm.weight` exists in the model, the model uses it. Config flags should reflect tensor reality, not the other way around.

## Common Patterns (Added to Postmortem Index)

### Config-Tensor Mismatch

When config flags don't match tensor presence:
- Weights exist but feature is disabled
- Validation at conversion time catches this
- Auto-detection from tensor names prevents it

### Preset Inheritance Gaps

When presets don't set all required flags:
- Missing flag falls back to default
- Default may be wrong for the architecture
- Solution: validate tensor-config consistency

## Related Postmortems

- [2025-12-16-gemma3-debug](./2025-12-16-gemma3-debug.md) - Q4K quantization format
- [2025-12-25-gemma3-qknorm-offset](./2025-12-25-gemma3-qknorm-offset.md) - Missing +1 offset for q/k norms

## Verification Checklist

After similar issues, verify:
- [ ] All normalization tensors have corresponding config flags enabled
- [ ] `validateTensorConfigConsistency()` passes for the manifest
- [ ] Logit values for common tokens match HuggingFace reference (+/- 0.1)
- [ ] Model generates coherent text for standard prompts

## Action Items

| Priority | Action | Status |
|----------|--------|--------|
| P0 | Fix gemma3.json preset | Done |
| P0 | Fix existing manifest | Done |
| P1 | Add tensor-config validator | Done |
| P1 | Add auto-detection in converter | Done |
| P2 | Add logit comparison integration test | TODO |
| P2 | Add style guide section on tensor-config consistency | TODO |

---

*Created: 2026-01-19*
