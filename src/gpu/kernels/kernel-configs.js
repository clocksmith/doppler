const loadJson = async (path) => {
  const response = await fetch(new URL(path, import.meta.url));
  if (!response.ok) throw new Error(`Failed to load registry: ${path}`);
  return response.json();
};

const registry = await loadJson('../../config/kernels/registry.json');
import { resolveKernelConfig } from '../../config/schema/kernel-registry.schema.js';

export const KERNEL_CONFIGS = Object.fromEntries(
  Object.entries(registry.operations).map(([operation, opSchema]) => {
    const variants = Object.fromEntries(
      Object.entries(opSchema.variants).map(([variant, variantSchema]) => {
        const resolved = resolveKernelConfig(operation, variant, opSchema, variantSchema);
        return [
          variant,
          {
            shaderFile: resolved.wgsl,
            entryPoint: resolved.entryPoint,
            workgroupSize: resolved.workgroup,
            requires: resolved.requires,
            bindings: resolved.bindings,
            uniforms: resolved.uniforms,
            wgslOverrides: resolved.wgslOverrides,
            sharedMemory: resolved.sharedMemory,
            outputDtype: resolved.outputDtype ?? undefined,
            variantMetadata: resolved.variantMetadata ?? undefined,
          },
        ];
      })
    );

    return [operation, variants];
  })
);

export function getKernelConfig(operation, variant) {
  const config = KERNEL_CONFIGS[operation]?.[variant];
  if (!config) {
    throw new Error(`Unknown kernel: ${operation}/${variant}`);
  }
  return config;
}

export function setKernelValidator(
  operation,
  variant,
  validator
) {
  const config = KERNEL_CONFIGS[operation]?.[variant];
  if (config) {
    config.validate = validator;
  }
}
