import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { loadJson } from '../formats/load-json.js';
import { resolveKernelConfig } from './schema/kernel-registry.schema.js';

const registry = await loadJson('./kernels/registry.json', import.meta.url, 'Failed to load registry');

function freezeConfig(value) {
  if (value && typeof value === 'object') {
    if (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) {
      throw new Error('Kernel registry metadata must contain only JSON objects and arrays.');
    }
    for (const child of Object.values(value)) freezeConfig(child);
    Object.freeze(value);
  }
  return value;
}

const instances = new WeakSet();
export function isKernelRegistry(value) { return instances.has(value); }

export function createKernelRegistry({ extensions = {}, validators = {} } = {}) {
  const operations = structuredClone(registry.operations);
  for (const [name, extension] of Object.entries(structuredClone(extensions))) {
    if (['__proto__', 'constructor', 'prototype'].includes(name)) throw new Error('Invalid kernel operation name.');
    operations[name] = { ...operations[name], ...extension,
      variants: { ...operations[name]?.variants, ...extension.variants } };
  }
  const configs = Object.fromEntries(

  Object.entries(operations).map(([operation, operationSchema]) => {
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
        const config = {
          operation,
          variant,
          shaderFile: resolved.wgsl,
          entryPoint: resolved.entryPoint,
          workgroupSize: resolved.workgroup,
          requires: resolved.requires,
          requiredWgslFeatures: resolved.requiredWgslFeatures,
          bindings: resolved.bindings,
          uniforms: resolved.uniforms,
          wgslOverrides: resolved.wgslOverrides,
          sharedMemory: resolved.sharedMemory,
          outputDtype: resolved.outputDtype ?? undefined,
          weightDtype: resolved.weightDtype ?? undefined,
          variantMetadata: resolved.variantMetadata ?? undefined,
        };
        return [variant, config];
      })
    );
    return [operation, variants];
  })
);

  const companion = {};
  const validatorIdentities = {};
  for (const [operation, variants] of Object.entries(validators)) {
    companion[operation] = {};
    validatorIdentities[operation] = {};
    for (const [variant, descriptor] of Object.entries(variants)) {
      if (!Object.hasOwn(configs, operation) || !Object.hasOwn(configs[operation], variant)) {
        throw new Error(`Validator references unknown kernel: ${operation}/${variant}`);
      }
      if (typeof descriptor?.id !== 'string' || !descriptor.id || typeof descriptor.validate !== 'function') {
        throw new Error(`Kernel validator ${operation}/${variant} requires an id and validate function.`);
      }
      companion[operation][variant] = Object.freeze({ id: descriptor.id, validate: descriptor.validate });
      validatorIdentities[operation][variant] = descriptor.id;
    }
  }
  const identity = computeCanonicalSha256({ configs, validators: validatorIdentities });
  freezeConfig(configs);
  freezeConfig(companion);
  const instance = Object.freeze({
    identity, configs, validators: companion,
    getKernelConfig(operation, variant) {
      if (!Object.hasOwn(configs, operation) || !Object.hasOwn(configs[operation], variant)) {
        throw new Error(`Unknown kernel: ${operation}/${variant}`);
      }
      return configs[operation][variant];
    },
    getKernelValidator(operation, variant) {
      instance.getKernelConfig(operation, variant);
      return companion[operation]?.[variant]?.validate ?? null;
    },
  });
  instances.add(instance);
  return instance;
}

export const DEFAULT_KERNEL_REGISTRY = createKernelRegistry();
export const KERNEL_CONFIGS = DEFAULT_KERNEL_REGISTRY.configs;
let activeRegistry = null;
export function getActiveKernelRegistry() { return activeRegistry; }
export function enterKernelRegistry(registry) {
  if (!isKernelRegistry(registry)) throw new Error('Expected a constructed kernel registry instance.');
  const previous = activeRegistry;
  activeRegistry = registry;
  return () => { activeRegistry = previous; };
}
export function getKernelConfig(operation, variant) {
  return (activeRegistry ?? DEFAULT_KERNEL_REGISTRY).getKernelConfig(operation, variant);
}
export function getKernelConfigs() { return (activeRegistry ?? DEFAULT_KERNEL_REGISTRY).configs; }

