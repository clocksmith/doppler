# Ping-Pong Buffer Incorrectly Released During Decode

**Status**: RESOLVED
**Date**: 2025-12-31
**Model**: All models (gpt-oss-20b reported)
**Symptom**: Decode regression after ping-pong buffer optimization
**Root Cause**: Buffer release check used index-dependent lookup that missed one of the two buffers

---

## Summary

After commit d1b40f0 ("Pre-allocated ping-pong buffers for decode layer output"), decode inference could fail or produce garbage. The ping-pong buffer B was incorrectly released to the buffer pool after odd-numbered layers (1, 3, 5...), allowing it to be reused while still needed.

---

## Root Cause

### Ping-Pong Buffer System

Decode uses two pre-allocated buffers (A and B) that alternate as input/output:
- Layer 0: reads A, writes B
- Layer 1: reads B, writes A
- Layer 2: reads A, writes B
- ...

After each layer, `swapPingPong()` toggles the index (0→1→0→1...).

### The Bug

In `_decodeStep()`, the code checked if `prevStates` was a pre-allocated buffer:

```typescript
// BEFORE: Bug - decodeAltBuffer changes based on current index
const decodeHiddenBuffer = this.decodeBufferManager.getHiddenBuffer();  // Captured ONCE at start (A)

for (let l = 0; l < numLayers; l++) {
  // ... layer processing ...
  this.decodeBufferManager.swapPingPong();

  // BUG: This re-fetches based on CURRENT index after swap
  const decodeAltBuffer = this.decodeBufferManager.getHiddenBuffer();
  const isPreAllocated = prevStates === decodeHiddenBuffer || prevStates === decodeAltBuffer;
}
```

### Index-Dependent Failure

| After Layer | Index | decodeHiddenBuffer | decodeAltBuffer | prevStates | Check Result |
|-------------|-------|-------------------|-----------------|------------|--------------|
| 0 | 1 | A | B | A | A===A ✓ |
| 1 | 0 | A | A | B | B===A ✗ **RELEASED!** |
| 2 | 1 | A | B | A | A===A ✓ |
| 3 | 0 | A | A | B | B===A ✗ **RELEASED!** |

After odd layers, the index swapped back to 0, making both `decodeHiddenBuffer` and `decodeAltBuffer` point to buffer A. Buffer B was then released, causing:
- Next layer reads from potentially reused/corrupted buffer
- Use-after-free behavior
- Garbage output or crashes

---

## Fix Applied

### pipeline.js:1139-1142 - Capture Both Buffers Upfront

```typescript
// AFTER: Capture both buffers once, before the loop
const decodeHiddenBuffer = this.decodeBufferManager.getHiddenBuffer();      // A
const decodeAltBuffer = this.decodeBufferManager.getOutputHiddenBuffer();   // B

for (let l = 0; l < numLayers; l++) {
  // ... layer processing ...
  this.decodeBufferManager.swapPingPong();

  // Now both A and B are always checked, regardless of current index
  const isPreAllocated = prevStates === decodeHiddenBuffer || prevStates === decodeAltBuffer;
}
```

By capturing both buffers before the loop starts (when index=0), we get:
- `decodeHiddenBuffer` = A (input buffer at index 0)
- `decodeAltBuffer` = B (output buffer at index 0)

These references remain stable throughout the loop, correctly identifying both ping-pong buffers regardless of the current index.

---

## Additional Fix

### gpu/kernels/fused_ffn.js:148,223 - Trace API Signature

Build failed due to `trace.ffn()` being called with 1 argument instead of 2:

```typescript
// BEFORE: Wrong signature
trace.ffn(`FusedFFN: variant=${variant}...`);

// AFTER: Use trace.kernels() which takes single message argument
trace.kernels(`FusedFFN: variant=${variant}...`);
```

The `trace.ffn()` function requires `(layerIdx: number, message: string)` for layer-specific tracing. Since `fused_ffn.js` is a kernel file without layer context, `trace.kernels()` is more appropriate.

---

## Lessons Learned

1. **Index-based lookups are fragile** - When state alternates, capturing values at a single point can miss half the cases
2. **Test buffer lifecycle across multiple layers** - Single-layer tests wouldn't catch this bug since it only manifests after odd layers
3. **Pre-allocated buffers need explicit tracking** - Don't rely on dynamic index-based methods to identify them
4. **Ping-pong patterns need careful trace-through** - Walk through at least 4 iterations to see the full cycle (0→1→0→1)

---

## Verification

After fix:
- `npm run build` succeeds
- `npm test` kernel tests pass
- Ping-pong buffers A and B both correctly identified as pre-allocated

---

## Related

- Commit d1b40f0: Original ping-pong buffer implementation
- `inference/decode-buffers.js`: DecodeBufferManager implementation
- `inference/pipeline.js:_decodeStep()`: Decode token generation loop
