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

### Alternative Hypotheses

1. **Uniform buffer cache collision** - Content-addressed cache might return wrong buffer if hash collides (unlikely - different content = different hash)

2. **Hidden state buffer reuse** - DecodeBufferManager ping-pong might be incorrect in batched mode

3. **Attention mask computation** - `startPosForMask` might be wrong for iterations > 0

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

- `src/inference/pipeline.ts:1509-1677` - `_generateNTokensGPU()`
- `src/inference/pipeline/attention.ts:791-860` - `recordAttention()`
- `src/inference/kv-cache.ts:398-439` - `recordUpdateFromGPU()`
- `src/gpu/kernels/rope.ts:94-150` - `recordRoPE()`

## Reproduction

```bash
# Set batchSize: 4 in defaults, then:
npm run debug -- -m gemma-2-2b-it --prompt "The sky is" --max-tokens 32 --temperature 0 --no-chat

# Compare with batchSize: 1
```

## Next Steps

1. Add per-layer `kvLen` logging in `recordAttention()` to trace exact values
2. Dump attention mask values for iteration 0 vs iteration 1
3. Check if hidden state buffers are being reused incorrectly
4. Consider if WebGPU needs explicit barriers between iterations (shouldn't - command order is preserved)

## Lessons Learned

- Batched recording is subtle - CPU-side metadata updates vs GPU-side buffer updates
- Always test correctness before performance optimization
- The single-token path is the reference implementation
