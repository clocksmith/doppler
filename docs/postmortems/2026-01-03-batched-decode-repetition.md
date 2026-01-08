# Batched Decode Repetition Bug

**Date:** 2026-01-03
**Status:** Open (workaround applied)
**Severity:** P1 - Correctness bug in performance-critical path

## Summary

Batched decode (batchSize > 1) produces repetitive/incorrect output while single-token decode (batchSize=1) produces correct output. The bug causes generated text to repeat patterns after ~15 tokens.

## Symptoms

```
# batchSize=1 (correct)
"The sky is" → " a beautiful blue color, and it is a very beautiful day to be out of"

# batchSize=4 (incorrect - repetition)
"The sky is" → " a beautiful blue color, and it is a very beautiful day to go to the city of the day to go to the city of the day..."
```

Key observations:
- First ~12-14 tokens are identical between paths
- Divergence starts around token 14
- After divergence, batched path enters repetition loop
- Loop length ~7-8 tokens

## Investigation

### Verified Working

1. **RoPE positions** - Logged `startPos` in `recordRoPE()`:
   ```
   [RoPE] startPos=2  (iteration 0, all 26 layers × 2)
   [RoPE] startPos=3  (iteration 1, all 26 layers × 2)
   [RoPE] startPos=4  (iteration 2, all 26 layers × 2)
   ```
   Positions increment correctly per iteration.

2. **Context updates** - `context.currentSeqLen` updates correctly:
   ```
   [Batch] Iteration 0: currentPos=2, context.currentSeqLen=2
   [Batch] Iteration 1: currentPos=3, context.currentSeqLen=3
   [Batch] Iteration 2: currentPos=4, context.currentSeqLen=4
   ```

3. **KV cache seqLen** - `recordUpdateFromGPU()` updates `layer.seqLen` immediately:
   ```typescript
   // kv-cache.ts:434
   layer.seqLen = Math.max(layer.seqLen, startPos + numTokens);
   ```

### Suspected Root Cause

The batched decode records N iterations into a single command buffer:

```typescript
for (let i = 0; i < N; i++) {
  context.currentSeqLen = currentPos;
  // Record: embed → 26 layers → logits → sample → copy token
}
recorder.submit();  // Execute all N iterations at once
```

**Hypothesis:** Within the recorded command buffer, iteration i+1's attention reads from the KV cache, but iteration i's K/V writes haven't been visible yet because:

1. All iterations are RECORDED before any execution
2. The GPU copies (K/V updates) are ordered correctly, but...
3. The `kvLenForAttention` read from `gpuBuffers.seqLen` (CPU-side) may be stale

**Smoking gun location** - `attention.ts:832`:
```typescript
kvLenForAttention = gpuBuffers.seqLen;
```

This reads the CPU-side `seqLen` which IS updated during recording. But something else must be wrong because the tokens diverge.

### Primary Hypothesis: Uniform Cache Eviction (2026-01-03 investigation)

The uniform buffer cache in `src/gpu/uniform-cache.ts` has 256 max entries. When full, it evicts LRU entries and **destroys** the GPU buffer immediately:

```typescript
// uniform-cache.ts:158
private evictLRU(): void {
  // ... find oldest entry
  entry.buffer.destroy();  // DESTROYS buffer still in pending command buffer!
}
```

**Why this explains the bug:**
- Each decode step creates ~10+ uniform buffers per layer × 26 layers = ~260+ per token
- Batched recording creates N iterations worth of commands BEFORE any execute
- After ~12 tokens (3 batches × 4), cache hits 256 capacity
- Eviction destroys buffer that's still referenced by pending (not yet executed) commands
- GPU reads garbage data → wrong attention → repetitive output

**Supporting evidence:**
- Bug manifests after EXACTLY 3 batches (12 tokens) - deterministic, not random
- First 12 tokens are identical between batchSize=1 and batchSize=4
- Divergence at token 13 = exactly when cache likely overflows

**Fix direction:** Don't destroy evicted buffers immediately. Either:
1. Mark as "pending destruction" and destroy after `onSubmittedWorkDone()` callback
2. Increase cache size for batched mode
3. Disable caching during batched recording (use fresh buffers per iteration)

### Secondary Issue: isDecodeMode=false

The batched path at `pipeline.ts:1499` passes `isDecodeMode=false`:
```typescript
const context = this._buildLayerContext(recorder);  // isDecodeMode defaults to false
```

While single-token path at `pipeline.ts:1141` passes `true`:
```typescript
const context = this._buildLayerContext(recorder, true);  // isDecodeMode = TRUE
```

This means batched decode doesn't use pre-allocated decode buffers from `DecodeBufferManager`, potentially causing more buffer churn and cache pressure.

### Ruled Out Hypotheses

1. **Uniform buffer cache collision** - Content-addressed cache uses FNV-1a hash on 40 bytes, collision probability is low

2. **Hidden state buffer reuse** - DecodeBufferManager ping-pong is correct (verified)

3. **Attention mask computation** - `startPosForMask` updates correctly (verified via logging)

## Workaround

Set `batchSize: 1` in:
- `src/config/schema/inference-defaults.schema.ts:37`
- `src/config/inference/defaults.json:3`

This uses the single-token decode path which works correctly.

## Impact

| Metric | batchSize=1 | batchSize=8 (broken) |
|--------|-------------|----------------------|
| Decode | ~7-8 tok/s | ~11 tok/s (wrong output) |
| GPU submits | ~128 | ~16 |
| Quality | Correct | Repetitive garbage |

Batching would provide 37% speedup if working.

## Files Involved

- `src/inference/pipeline.ts:1499-1677` - `_generateNTokensGPU()` (batched decode)
- `src/inference/pipeline.ts:1141` - Single-token decode (`isDecodeMode=true`)
- `src/gpu/uniform-cache.ts:133-163` - LRU eviction with buffer destruction
- `src/inference/pipeline/attention.ts:791-860` - `recordAttention()`
- `src/inference/kv-cache.ts:398-439` - `recordUpdateFromGPU()`
- `src/gpu/kernels/rope.ts:94-150` - `recordRoPE()`

## Reproduction

```bash
# Set batchSize: 4 in defaults, then:
npm run debug -- -m gemma-2-2b-it --prompt "The sky is" --max-tokens 32 --temperature 0 --no-chat

# Compare with batchSize: 1
```

## Investigation Progress (2026-01-03)

### Confirmed Behavior
- With `batchSize=4`, tokens 1-12 are IDENTICAL to `batchSize=1`
- Divergence occurs at token 13 (exactly after 3 batches)
- Pattern: `batchSize=1` produces "be", `batchSize=4` produces "go"
- After divergence, enters repetition loop: "go to the city of the day" repeating

### Ruled Out
- ☒ RoPE positions (logged, increment correctly)
- ☒ `context.currentSeqLen` updates (updates correctly in loop)
- ☒ KV cache `layer.seqLen` (updates correctly via `recordUpdateFromGPU`)
- ☒ Uniform buffer hash collisions (FNV-1a on 40 bytes, low collision probability)

### Current Hypotheses
1. **Attention startPos/kvLen mismatch** - The attention uniform buffer may have stale values
2. **Decoder buffer misuse** - `_generateNTokensGPU` doesn't pass `isDecodeMode=true` to `_buildLayerContext`
3. **Cross-batch state corruption** - Something accumulates incorrectly across batch boundaries

### Key Observation
The bug manifests after EXACTLY 3 batches (12 tokens). This suggests:
- The bug isn't random - it's deterministic
- Something accumulates or corrupts after a specific pattern
- Likely related to how batch boundaries interact with KV cache or context state

## Next Steps

1. Add explicit logging in `_generateNTokensGPU` to trace:
   - `this.currentSeqLen` at batch start
   - `context.currentSeqLen` for each iteration
   - `layer.seqLen` after each `recordUpdateFromGPU`
2. Test with `batchSize=2` and `batchSize=8` to see if divergence pattern changes
3. Compare attention uniform buffer contents between single-token and batch paths
4. Check if `isDecodeMode=true` affects the bug (pass to `_buildLayerContext`)

## Lessons Learned

- Batched recording is subtle - CPU-side metadata updates vs GPU-side buffer updates
- Always test correctness before performance optimization
- The single-token path is the reference implementation
