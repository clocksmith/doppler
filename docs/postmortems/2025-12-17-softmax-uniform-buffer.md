# Softmax Uniform Buffer Layout Bug Post-Mortem

**Status**: RESOLVED
**Date**: 2025-12-17
**Author**: Claude (Opus 4.5)
**Impact**: Softmax kernel producing incorrect results (maxError=0.137)

---

## Summary

The softmax kernel was failing correctness tests with maxError=0.137 (should be <1e-5). The root cause was a swapped uniform buffer layout between the TypeScript host code and the WGSL shader.

---

## Timeline

| Time | Event |
|------|-------|
| 2025-12-17 | Softmax test discovered failing during kernel correctness sweep |
| 2025-12-17 | Investigated uniform buffer layout mismatch |
| 2025-12-17 | Fixed `runSoftmax` and `recordSoftmax` in softmax.js |
| 2025-12-17 | Verified softmax test passes |

---

## Root Cause

### The Bug

The WGSL shader defined the uniform struct as:

```wgsl
// softmax.wgsl:12-17
struct SoftmaxUniforms {
    innerSize: u32,    // offset 0 - Size of dimension to softmax over
    outerSize: u32,    // offset 4 - Product of all other dimensions
    temperature: f32,  // offset 8
    _pad: u32,         // offset 12
}
```

But the TypeScript code was writing the values in the **wrong order**:

```typescript
// softmax.js:45-47 (BEFORE - WRONG)
uniformView.setUint32(0, batchSize, true);      // Writing outerSize at offset 0
uniformView.setUint32(4, inferredSize, true);   // Writing innerSize at offset 4
uniformView.setFloat32(8, temperature, true);
```

This caused `innerSize` and `outerSize` to be swapped when read by the GPU shader.

### Why This Caused Incorrect Results

The softmax kernel uses these dimensions to:
1. Dispatch one workgroup per row (`outerSize` rows)
2. Compute softmax over `innerSize` elements per row

With swapped dimensions:
- Wrong number of workgroups dispatched
- Each workgroup processed wrong number of elements
- Row boundaries misaligned
- Results completely wrong (maxError=0.137 instead of <1e-5)

---

## Fix

### Changes Made

**File:** `doppler/gpu/kernels/softmax.js`

#### 1. Fixed `runSoftmax` (lines 45-48)

```typescript
// AFTER - CORRECT
// WGSL struct: { innerSize: u32, outerSize: u32, temperature: f32, _pad: u32 }
uniformView.setUint32(0, inferredSize, true);  // innerSize at offset 0
uniformView.setUint32(4, batchSize, true);     // outerSize at offset 4
uniformView.setFloat32(8, temperature, true);
```

#### 2. Fixed `recordSoftmax` (lines 177-180)

```typescript
// AFTER - CORRECT
// WGSL struct: { innerSize: u32, outerSize: u32, temperature: f32, _pad: u32 }
uniformView.setUint32(0, inferredSeqLen, true);  // innerSize at offset 0
uniformView.setUint32(4, batchSize, true);       // outerSize at offset 4
uniformView.setFloat32(8, 1.0, true);            // temperature (default 1.0)
```

---

## Verification

### Test Command
```bash
doppler --config <ref>
# <ref>: cli.command="test", cli.suite="kernels", cli.filter="softmax"
```

### Before Fix
```
FAIL softmax (maxError=0.137)
```

### After Fix
```
PASS softmax (345ms)
```

### Full Kernel Suite
```
correctness: 11 passed, 3 failed (1.5s)
- matmul-q4k: FAIL (SwiftShader lacks subgroups - expected)
- matmul-q4k-large: FAIL (SwiftShader lacks subgroups - expected)
- scatter-add: FAIL (pre-existing issue)
```

---

## Impact Assessment

### What Was Affected

| Component | Impact |
|-----------|--------|
| Standalone softmax kernel | Incorrect results |
| MoE routing (uses softmax) | Potentially affected |
| TopK with softmax | Potentially affected |

### What Was NOT Affected

| Component | Reason |
|-----------|--------|
| Attention softmax | Uses inline implementation in attention.wgsl |
| Main inference pipeline | Attention doesn't call standalone softmax kernel |

This explains why fixing softmax didn't resolve the inference garbage output - the attention mechanism uses its own embedded softmax implementation.

---

## Lessons Learned

### 1. Uniform Buffer Layout is Error-Prone

The mismatch between TypeScript and WGSL uniform layouts is a common source of bugs. Consider:
- Adding a shared schema definition
- Generating TypeScript bindings from WGSL structs
- Adding compile-time or runtime validation

### 2. Parameter Naming Matters

The bug was partially obscured by inconsistent naming:
- TypeScript uses `batchSize` and `inferredSize`
- WGSL uses `outerSize` and `innerSize`

Using consistent names across languages would make mismatches more obvious.

### 3. Test Coverage Caught This

The kernel correctness test suite successfully identified this bug. Maintaining comprehensive kernel tests is essential.

---

## Prevention Recommendations

1. **Add comments documenting WGSL struct layout** at every uniform buffer write site (implemented in fix)

2. **Consider generating uniform buffer writers** from WGSL struct definitions

3. **Review all kernel uniform buffers** for similar layout mismatches

4. **Add integration tests** that verify end-to-end softmax behavior in realistic scenarios

---

## Related Issues

- Inference still produces garbage output (separate root cause)
- Hidden state explosion through layers (maxAbs: 17 → 630)
- See: [2025-12-17-positive-bias-hidden-states.md](2025-12-17-positive-bias-hidden-states.md)

---

## Files Modified

| File | Change |
|------|--------|
| `gpu/kernels/softmax.js` | Fixed uniform buffer layout in runSoftmax and recordSoftmax |
| `../ROADMAP.md` | Added softmax fix to completed fixes |

---

## Appendix: Full Diff

```diff
--- a/doppler/gpu/kernels/softmax.js
+++ b/doppler/gpu/kernels/softmax.js
@@ -42,9 +42,10 @@ export async function runSoftmax(

   // Create uniform buffer
+  // WGSL struct: { innerSize: u32, outerSize: u32, temperature: f32, _pad: u32 }
   const uniformData = new ArrayBuffer(16);
   const uniformView = new DataView(uniformData);
-  uniformView.setUint32(0, batchSize, true);
-  uniformView.setUint32(4, inferredSize, true);
+  uniformView.setUint32(0, inferredSize, true);  // innerSize at offset 0
+  uniformView.setUint32(4, batchSize, true);     // outerSize at offset 4
   uniformView.setFloat32(8, temperature, true);

@@ -174,9 +175,11 @@ export async function recordSoftmax(

   // Uniform buffer
+  // WGSL struct: { innerSize: u32, outerSize: u32, temperature: f32, _pad: u32 }
   const uniformData = new ArrayBuffer(16);
   const uniformView = new DataView(uniformData);
-  uniformView.setUint32(0, batchSize, true);
-  uniformView.setUint32(4, inferredSeqLen, true);
+  uniformView.setUint32(0, inferredSeqLen, true);  // innerSize at offset 0
+  uniformView.setUint32(4, batchSize, true);       // outerSize at offset 4
+  uniformView.setFloat32(8, 1.0, true);            // temperature (default 1.0)
```

---

*Post-mortem written: 2025-12-17*
*Status: RESOLVED*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `../style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.
