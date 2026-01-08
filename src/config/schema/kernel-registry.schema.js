/**
 * Kernel Registry Schema Definitions
 *
 * Defines the structure for kernel metadata: variants, bindings, uniforms,
 * workgroup sizes, and GPU feature requirements.
 *
 * @module config/schema/kernel-registry
 */

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Merge base and variant bindings.
 * Variant bindings with matching indices override base bindings.
 */
export function mergeBindings(
  base,
  override
) {
  if (!override || override.length === 0) {
    return [...base];
  }

  const result = [...base];
  for (const binding of override) {
    const existingIdx = result.findIndex(b => b.index === binding.index);
    if (existingIdx >= 0) {
      result[existingIdx] = binding;
    } else {
      result.push(binding);
    }
  }

  return result.sort((a, b) => a.index - b.index);
}

/**
 * Resolve a kernel variant to a complete configuration.
 */
export function resolveKernelConfig(
  operation,
  variant,
  opSchema,
  variantSchema
) {
  return {
    operation,
    variant,
    wgsl: variantSchema.wgsl,
    entryPoint: variantSchema.entryPoint,
    workgroup: variantSchema.workgroup,
    requires: variantSchema.requires ?? [],
    bindings: mergeBindings(opSchema.baseBindings, variantSchema.bindingsOverride),
    uniforms: variantSchema.uniformsOverride ?? opSchema.baseUniforms,
    wgslOverrides: variantSchema.wgslOverrides ?? {},
    sharedMemory: variantSchema.sharedMemory ?? 0,
  };
}
