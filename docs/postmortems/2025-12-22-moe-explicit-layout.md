# MoE Explicit Bind Group Layout Bug Post-Mortem

This document covers the debugging session for the MoE gather kernel which was not executing despite compiling successfully.

## Symptoms

- GPT-OSS 20B MoE model failed with "MoE tokenOffsets incomplete at i=0"
- `tokenCounts` array was ALL ZEROS after `count_and_map` kernel execution
- Shader compiled successfully, pipeline created, commands submitted
- But the kernel did not write any values to the output buffer

## Root Cause: WebGPU 'auto' Layout Binding Mismatch

The `moe_gather.wgsl` shader has 6 bindings but the `count_and_map` entry point only uses 4 of them:

```
Bindings in shader:
  @binding(0) uniforms        - USED by count_and_map
  @binding(1) hiddenStates    - NOT USED by count_and_map (only by gather)
  @binding(2) expertIndices   - USED by count_and_map
  @binding(3) gathered        - NOT USED by count_and_map (only by gather)
  @binding(4) tokenCounts     - USED by count_and_map
  @binding(5) tokenMap        - USED by count_and_map
```

When using WebGPU's `layout: 'auto'`, the pipeline layout only includes bindings actually used by the entry point. The `countPipeline.getBindGroupLayout(0)` returns a layout with only 4 entries, but we were trying to create a bind group with all 6 entries - causing a silent mismatch.

### Why This Was Hard to Debug

1. No WebGPU validation error was thrown
2. The shader compiled successfully
3. The pipeline was created successfully
4. Commands were submitted successfully
5. `device.queue.onSubmittedWorkDone()` resolved successfully
6. But the kernel simply did not execute

A minimal test kernel with 1 binding (storage read_write) worked correctly, proving the WebGPU implementation was functional. The issue was specific to the 6-binding configuration with 'auto' layout.

## The Fix

Created an explicit bind group layout with all 6 bindings and used it for both pipelines:

```javascript
const explicitBindGroupLayout = device.createBindGroupLayout({
  label: 'moe_gather_explicit_layout',
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ],
});

// Pass explicit layout to createPipeline
const countPipeline = await createPipeline('moe_gather', 'count', explicitBindGroupLayout);
const gatherPipeline = await createPipeline('moe_gather', gatherVariant, explicitBindGroupLayout);

// Create single bind group that works for both pipelines
const bindGroup = device.createBindGroup({
  layout: explicitBindGroupLayout,
  entries: bindGroupEntries,
});
```

## Files Modified

| File | Change |
|------|--------|
| `gpu/kernels/moe.js` | Create explicit bind group layout for MoE kernels |
| `gpu/kernels/moe_gather.wgsl` | Added note about explicit layout requirement |

## Verification

After fix:
- `tokenCounts` shows correct distribution: `e3:25, e5:25, e12:22, e13:3, e20:25`
- Total tokens mapped: 100 (25 tokens * 4 topK)
- GPT-OSS 20B test passes

## Key Learnings

1. **Avoid 'auto' layout for multi-entry-point shaders**: When a shader has multiple entry points that use different subsets of bindings, create an explicit layout that includes ALL bindings.

2. **Silent failures are dangerous**: WebGPU can fail silently when bind group/layout mismatches occur without throwing errors.

3. **Test with minimal kernels**: A 1-binding test kernel helped isolate the issue to the binding configuration, not the WebGPU implementation.

4. **Validate all bindings are used**: If using 'auto' layout, ensure each entry point touches all bindings (even with dummy reads) to force inclusion in the layout.

## Related Issues

- [BF16 2D Dispatch Bug](./BF16-2D-DISPATCH-POSTMORTEM.md) - Another WebGPU kernel bug with similar silent failure pattern

---

*Fixed: 2025-12-14*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes, CLI flags (`--kernel-path`, `--kernel-profile`), and the OPFS purge helper.
