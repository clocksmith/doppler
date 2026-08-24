import { loadJson } from '../utils/load-json.js';
import { resolveKernelConfig } from './schema/kernel-registry.schema.js';

const registry = await loadJson('./kernels/registry.json', import.meta.url, 'Failed to load registry');

export const KERNEL_CONFIGS = Object.fromEntries(
  Object.entries(registry.operations).map(([operation, operationSchema]) => {
    const variants = Object.fromEntries(
      Object.entries(operationSchema.variants).map(([variant, variantSchema]) => {
        const resolved = resolveKernelConfig(operation, variant, operationSchema, variantSchema);
        if (!resolved.wgsl || typeof resolved.wgsl !== 'string') {
          throw new Error(
            `Kernel config ${operation}/${variant} is missing required field "shaderFile" (wgsl).`
          );
        }
        if (!resolved.entryPoint || typeof resolved.entryPoint !== 'string') {
          throw new Error(
            `Kernel config ${operation}/${variant} is missing required field "entryPoint".`
          );
        }
        return [variant, {
          shaderFile: resolved.wgsl,
          entryPoint: resolved.entryPoint,
          workgroupSize: resolved.workgroup,
          requires: resolved.requires,
          bindings: resolved.bindings,
          uniforms: resolved.uniforms,
          wgslOverrides: resolved.wgslOverrides,
          sharedMemory: resolved.sharedMemory,
          outputDtype: resolved.outputDtype ?? undefined,
          weightDtype: resolved.weightDtype ?? undefined,
          variantMetadata: resolved.variantMetadata ?? undefined,
        }];
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
