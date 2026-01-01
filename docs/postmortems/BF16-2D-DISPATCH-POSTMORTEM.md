# BF16->F32 2D Dispatch Bug Post-Mortem

This document covers the debugging session for the BF16->F32 conversion kernel which caused zero embeddings for tokens with IDs > ~8192.

## Symptoms

- Models with large vocabularies (Mistral 7B, GPT-OSS 20B) produced garbage output
- Embedding lookups for high token IDs returned all zeros
- Chat template tokens (often in high ID ranges) were broken
- Gemma 1B (vocab 262144) appeared broken despite correct weights

## Root Cause: 2D Dispatch Without Linearization

The BF16->F32 kernel used 2D dispatch for large tensors but only used `global_id.x` in the kernel, ignoring `global_id.y`.

### The Bug

When a tensor exceeds 65535 workgroups in the X dimension, WebGPU requires 2D dispatch:
```javascript
// kernel-selector.js
if (totalWorkgroups > MAX_WORKGROUPS) {
  workgroupsX = MAX_WORKGROUPS;
  workgroupsY = Math.ceil(totalWorkgroups / MAX_WORKGROUPS);
}
dispatchWorkgroups(workgroupsX, workgroupsY);
```

But the kernel only used the X index:
```wgsl
// BROKEN: bf16_to_f32.wgsl
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;  // Only processes row 0!
    // ...
}
```

### Why This Caused Zero Embeddings

The Mistral 7B embedding table is 32000 vocab × 4096 hidden = 131M elements. With 2 elements per thread (BF16 pair), we need ~65.5M workgroups, which exceeds 65535.

With 2D dispatch of (65535, 2):
- Correct: 65535 × 2 × 256 × 2 = 67,106,816 elements covered
- Broken: 65535 × 256 × 2 = 33,553,920 elements covered

Only the first ~33M elements (tokens 0-8191) were converted. Tokens 8192+ returned zeros.

### The Fix

Updated the kernel to compute linear index from 2D dispatch:

```wgsl
// FIXED: bf16_to_f32.wgsl
struct Uniforms {
    numElements: u32,
    workgroupsX: u32,  // Actual X workgroups dispatched
    _pad2: u32,
    _pad3: u32,
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Compute linear thread index from 2D dispatch
    let threads_per_row = uniforms.workgroupsX * WORKGROUP_SIZE;
    let linear_idx = global_id.y * threads_per_row + global_id.x;

    let pair_idx = linear_idx;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= uniforms.numElements) {
        return;
    }
    // ... rest of conversion
}
```

Updated kernel-selector.js to pass `workgroupsX` in uniforms:
```javascript
uniformView.setUint32(4, workgroupsX, true);  // For kernel linearization
```

## Files Modified

| File | Change |
|------|--------|
| `gpu/kernels/bf16_to_f32.wgsl` | Added 2D linearization using `workgroupsX` uniform |
| `gpu/kernel-selector.js` | Pass `workgroupsX` to kernel for linearization |

## Verification

After fix:
- Gemma 1B produces coherent output: "sky" (54.9%), "color" (42.7%) for "The sky is blue because..."
- Mistral 7B embeddings non-zero for all token IDs
- Multi-shard BF16 tensor loading verified working

## Key Learnings

1. **Always linearize 2D dispatch**: When using 2D dispatch for large workloads, the kernel must compute a linear index from both X and Y dimensions

2. **Check WebGPU limits**: The 65535 workgroup limit per dimension is easy to forget when initially writing for small models

3. **Large vocab models expose bugs**: 32K vocab fits in 1D dispatch for most operations, but 262K vocab (Gemma 3) forces 2D dispatch

4. **Pass dispatch info to kernels**: The kernel needs to know the actual X workgroup count to linearize correctly, not just the total count

## Related Kernels to Audit

Other kernels that may have similar 2D dispatch issues:
- `f16_to_f32.wgsl` - if it exists and uses 2D dispatch
- Any kernel processing embeddings or large tensors
- Custom quantization kernels with manual dispatch

## Debugging Techniques Used

1. **xxd verification**: Used `xxd` to verify shard files contained non-zero data
2. **Debug logging**: Added span-by-span logging to trace multi-shard tensor assembly
3. **Buffer statistics**: Logged non-zero byte counts after each loading step
4. **Selective testing**: Tested specific token IDs to isolate the boundary

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

