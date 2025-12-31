# DOPPLER Memory Patterns

This document describes GPU buffer allocation patterns, memory lifecycles, and optimization opportunities in DOPPLER's inference pipeline.

## Buffer Lifecycle Summary

| Buffer Type | Allocation Point | Lifetime | Reuse Pattern | Size |
|-------------|------------------|----------|---------------|------|
| Q4_K weights | Model load | Model lifetime | None (read-only) | ~144 bytes/256 elements |
| Dequant F16/F32 output | Per-layer forward | Layer computation | Pooled by size bucket | 2-4x weight size |
| Uniform buffers | Per-dispatch | **Destroyed immediately** | None (fresh each time) | 16-64 bytes |
| KV cache | Inference start | Sequence lifetime | Contiguous or paged | seq_len * hidden * layers |
| Activation buffers | Per-layer | Layer computation | Pooled by size bucket | batch * seq * hidden |

## Memory Flow Diagrams

### Non-Fused Dequantization Path
```
Q4_K buffer (VRAM)     F16/F32 buffer (VRAM)     Output (VRAM)
     144 bytes/block        512-1024 bytes/block
           |                      |                    |
           v                      v                    v
    [dequant.ts:96-152]    [matmul.ts]           [result]
           |                      |
           +---- BOTH IN MEMORY --+

Memory: Q4_K (N) + F16 (4N) = 5N  (doubling!)
```

### Fused Path (fused_matmul_q4.wgsl)
```
Q4_K buffer (VRAM)                              Output (VRAM)
     144 bytes/block
           |                                         |
           v                                         v
    [fused kernel - dequant in registers]       [result]

Memory: Q4_K (N) only = N  (no intermediate)
```

## Memory Doubling Hotspots

### 1. Dequantization Intermediate Buffer

**Location**: `gpu/kernels/dequant.ts:118`

```typescript
const output = outputBuffer || acquireBuffer(outputSize, undefined, 'dequant_output');
```

**Problem**: When using non-fused path, both Q4_K weights and dequantized F16/F32 output exist simultaneously.

**Impact** (example: 7B model):
- Q4_K weights: ~3.5 GB
- F16 dequantized: ~7 GB (2x)
- F32 dequantized: ~14 GB (4x)
- **Total with dequant**: 10.5-17.5 GB vs 3.5 GB fused

**Mitigation**: Use `--kernel-profile fused` to avoid intermediate buffers.

### 2. Uniform Buffer Churn

**Location**: `gpu/kernels/utils.ts:1149-1160`

```typescript
export function createUniformBufferWithView(...): GPUBuffer {
  const data = new ArrayBuffer(byteLength);
  const view = new DataView(data);
  writer(view);
  return createUniformBufferFromData(label, data, recorder, deviceOverride);
}
```

**Problem**: Every kernel dispatch creates a fresh uniform buffer (16-64 bytes), uses it once, then destroys it (`uniformBuffer.destroy()` in dequant.ts:148).

**Impact**: Thousands of small allocations per inference pass. While individually small, this creates:
- Allocation overhead on GPU driver
- Memory fragmentation
- Unnecessary CPU-GPU synchronization

**Opportunity**: Cache uniform buffers by content hash (WebLLM pattern).

### 3. Per-Layer Activation Allocation

**Location**: Throughout `inference/pipeline.ts`

**Problem**: Each transformer layer allocates new activation buffers from pool, releases previous.

**Pattern**:
```
Layer 0: acquire(buf_a) -> compute -> release(buf_a)
Layer 1: acquire(buf_b) -> compute -> release(buf_b)
Layer 2: acquire(buf_c) -> compute -> release(buf_c)
...
```

**Opportunity**: Ping-pong between two buffers:
```
Layer 0: buf_a -> buf_b
Layer 1: buf_b -> buf_a
Layer 2: buf_a -> buf_b
...
```

## Buffer Pool Strategy

**Location**: `gpu/buffer-pool.ts`

Current pooling:
- **Size bucketing**: Power-of-2 for buffers < 32MB, 16MB steps for larger
- **Per-bucket limit**: 8 buffers max per size bucket
- **Total limit**: 64 pooled buffers globally
- **Alignment**: 256 bytes

```typescript
// Size bucket calculation (buffer-pool.ts:79-109)
function getSizeBucket(size: number, maxAllowedSize: number = Infinity): number {
  const minBucket = 256;
  if (size <= minBucket) return minBucket;

  // Large buffer handling (>32MB): 16MB steps to avoid 2x blowup
  const largeThreshold = 32 * 1024 * 1024;
  if (size >= largeThreshold) {
    const largeStep = 16 * 1024 * 1024;
    return Math.ceil(size / largeStep) * largeStep;
  }

  // Small buffers: power-of-2 rounding
  const bits = 32 - Math.clz32(size - 1);
  return Math.pow(2, bits);
}
```

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `gpu/kernels/dequant.ts` | 96-152 | Dequant buffer allocation |
| `gpu/kernels/utils.ts` | 1149-1160 | Uniform buffer creation |
| `gpu/buffer-pool.ts` | 79-109, 148-150 | Pool config and bucketing |
| `gpu/kernels/fused_matmul_q4.wgsl` | 167-179 | Fused dequant in registers |
| `loader/doppler-loader.ts` | 970-1020 | Model load dequant path |
| `inference/pipeline.ts` | - | Layer orchestration |

## Optimization Opportunities

### Implemented
- [x] Fused Q4K matmul (dequant in registers)
- [x] Size-bucketed buffer pooling
- [x] Large buffer 16MB-step bucketing (avoids 2x OOM)
- [x] Uniform buffer caching by value (`gpu/uniform-cache.ts`)
- [x] Decode-step buffer pre-allocation (`inference/decode-buffers.ts`)
- [x] Ping-pong activation buffers (built into DecodeBufferManager)

### Not Yet Implemented
- [ ] Cross-layer buffer aliasing
- [ ] Full integration of decode buffers with layer processing

## New Buffer Reuse Systems

### Uniform Buffer Cache (`gpu/uniform-cache.ts`)

Caches small uniform buffers by content hash to avoid repeated allocations.

```typescript
// Before: fresh allocation each dispatch
const uniformBuffer = createUniformBufferWithView(...);
// ... use buffer ...
uniformBuffer.destroy();  // Destroyed immediately

// After: cached by content hash
const uniformBuffer = createUniformBufferWithView(...);  // Returns cached if same data
// ... use buffer ...
releaseUniformBuffer(uniformBuffer);  // Returns to cache, not destroyed
```

**Benefits**:
- Eliminates thousands of small allocations per inference
- Reduces GPU driver overhead
- Content-addressed: same uniform data reuses same buffer

### Decode Buffer Manager (`inference/decode-buffers.ts`)

Pre-allocates fixed-size buffers for decode (M=1) operations.

```typescript
// Pre-allocated at model load
decodeBufferManager.ensureBuffers({
  hiddenSize: 4096,
  intermediateSize: 11008,
  enablePingPong: true,
});

// Per decode step: get pre-allocated buffers
const input = decodeBufferManager.getHiddenBuffer();
const output = decodeBufferManager.getOutputHiddenBuffer();
// ... process layer ...
decodeBufferManager.swapPingPong();  // Swap for next layer
```

**Benefits**:
- Zero allocation during decode steps
- Ping-pong avoids pool acquire/release churn
- Fixed memory footprint for decode phase

## Comparison: DOPPLER vs WebLLM

| Aspect | DOPPLER | WebLLM |
|--------|---------|--------|
| Weight format | Q4_K (runtime dequant) | Pre-compiled (TVM) |
| Intermediate buffers | F16 buffer in non-fused | None (compiled out) |
| Uniform buffers | Fresh per dispatch | Cached by value |
| Buffer reuse | Pool by size | Pool + decode buffers |
| Dynamic shapes | Yes | Yes (native) |

WebLLM's key advantage: TVM pre-compilation eliminates runtime dequantization entirely. DOPPLER's fused kernels achieve similar memory efficiency but with runtime overhead.
