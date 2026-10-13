// =============================================================================
// Helper Functions
// =============================================================================

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

export function resolveKernelConfig(
  operation,
  variant,
  opSchema,
  variantSchema
) {
  if (variantSchema.bindings !== undefined && variantSchema.bindingsOverride !== undefined) {
    throw new Error(`Kernel "${operation}/${variant}" cannot declare both bindings and bindingsOverride.`);
  }
  // Omitted on legacy registries: submission, not GPU completion, ends the call.
  const immediateCompletion = variantSchema.immediateCompletion === undefined
    ? 'submitted' : variantSchema.immediateCompletion;
  if (immediateCompletion !== 'submitted' && immediateCompletion !== 'settled') {
    throw new Error(`Kernel "${operation}/${variant}" has invalid immediateCompletion.`);
  }
  return {
    operation,
    variant,
    wgsl: variantSchema.wgsl,
    entryPoint: variantSchema.entryPoint,
    immediateCompletion,
    workgroup: variantSchema.workgroup,
    requires: variantSchema.requires ?? [],
    requiredWgslFeatures: variantSchema.requiredWgslFeatures ?? [],
    bindings: variantSchema.bindings ?? mergeBindings(opSchema.baseBindings, variantSchema.bindingsOverride),
    uniforms: variantSchema.uniformsOverride !== undefined ? variantSchema.uniformsOverride : opSchema.baseUniforms,
    wgslOverrides: variantSchema.wgslOverrides ?? {},
    sharedMemory: variantSchema.sharedMemory ?? 0,
    outputDtype: variantSchema.outputDtype ?? null,
    weightDtype: variantSchema.weightDtype ?? null,
    variantMetadata: variantSchema.variantMetadata ?? null,
  };
}
