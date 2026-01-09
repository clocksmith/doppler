# Buffer Pool Padding Causes False Positive Trace Explosions

**Status**: RESOLVED
**Date**: 2025-12-30
**Model**: Gemma 3 1B IT Q4_K_M
**Symptom**: Kernel trace showing "explosions" with values 78714.59 and 462351.88
**Root Cause**: Trace system reading garbage from buffer pool padding beyond valid data

---

## Summary

Debug trace showed alarming "value explosions" during decode:
```
L0.input_norm [1,1152] min=-24351.95 max=78714.59 EXPLOSION
L0.o_proj [1,1152] min=-24351.95 max=78714.59 EXPLOSION
[LAYER_OUT] L0: maxAbs=78714.5859 at idx=1219 (token=1, dim=67)
```

These were **false positives**. The actual computation was correct; the trace was reading garbage from buffer pool padding.

---

## Root Cause

### Buffer Pool Bucketing

The buffer pool rounds up allocation sizes for reuse efficiency:
- Decode needs: 1 token × 1152 floats × 4 bytes = **4608 bytes**
- Pool allocates: **8192 bytes** (8KB bucket)
- Extra padding: 896 floats of **uninitialized/stale data**

### Trace Reading Full Buffer

`snapshotFromArray()` in `kernel-trace.js` iterated over `arr.length` (full buffer) instead of the valid element count from `shape`:

```typescript
// BUG: Reads garbage beyond valid data
for (let i = 0; i < arr.length; i++) {  // arr.length = 2048 (pooled)
  // But only 1152 elements are valid!
}
```

### The "token=1" Clue

The trace reported `idx=1219 (token=1, dim=67)` - but decode only has **one token** (token=0). Token index 1 was garbage in the padding region:
- Valid data: indices 0-1151 (token 0)
- Garbage: indices 1152-2047 (stale pool data)

---

## Fixes Applied

### 1. kernel-trace.js:134-136 - snapshotFromArray

```typescript
// Only iterate over valid elements based on shape, not full buffer
const numElements = shape.reduce((a, b) => a * b, 1);
const limit = Math.min(arr.length, numElements);
for (let i = 0; i < limit; i++) {
```

### 2. layer.js:813-814 - LAYER_OUT debug

```typescript
// Only read valid elements (numTokens * hiddenSize), not full pooled buffer
const validSize = numTokens * hiddenSize * 4;
const staging = device.createBuffer({ size: validSize, ... });
```

### 3. pipeline.js:1090-1092 - Decode embed check

```typescript
// Only read valid elements (1 token * hiddenSize)
const validSize = config.hiddenSize * 4;
const embedData = await readBuffer(hiddenStates, validSize);
```

---

## Other Fixes in This Session

### Matmul Kernel Selection (matmul.js:94-100)

When `outputDtype='f16'` was requested with f16 weights, the code fell through to `return 'f32'` instead of using `f16w_f32a`. The f32 kernel can't read f16-packed weights, causing garbage output.

```typescript
// Fixed: Use f16w_f32a for any case with f16 weights
if (preferF16 && weightsAreF16 && capabilities.hasF16) {
  return 'f16w_f32a';
}
```

### Fused Kernel Naming (fused_matmul_rmsnorm.js:99,186)

Kernel config registered as `fused_matmul_rmsnorm` but code called `matmul_rmsnorm_fused`:
```typescript
// Fixed: Use correct name
const pipeline = await createPipeline('fused_matmul_rmsnorm', variant);
```

---

## Lessons Learned

1. **Buffer pool padding is invisible** - GPU buffers may be larger than requested data
2. **Always use shape to bound reads** - Never iterate over `buffer.size` or `array.length` without checking valid range
3. **Trace garbage values are suspiciously consistent** - The same values (78714.59, 462351.88) appearing everywhere is a sign of reading stale/uninitialized memory
4. **"token=1" in single-token decode is impossible** - A clear signal that indices were out of bounds

---

## Verification

After fixes, trace shows realistic values:
```
L0.o_proj [1,1152] min=-11.11 max=4.29
L0.post_attn_norm [1,1152] min=-6.59 max=6.59
```

No more 78714/462351 "explosions". The remaining ~1000 values in `post_attn_norm` are legitimate cumulative residual growth (documented separately in GEMMA3-MAGNITUDE-GAP-2025-12-21.md).
