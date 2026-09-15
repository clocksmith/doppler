import { getOrCreateBindGroupLayout } from './pipeline-cache.js';

export function getKernelBindGroupLayout(config, device) {
  return getOrCreateBindGroupLayout(`${config.operation}/${config.variant}`, config.bindings.map((binding) => ({
    binding: binding.index,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: binding.type },
  })), device);
}

export function createKernelBindingEntries(config, resources) {
  const entries = [];
  for (const binding of config.bindings) {
    const resource = resources[binding.name];
    if (resource == null && binding.optional === true) continue;
    const buffer = resource?.buffer;
    if (!buffer || (typeof GPUBuffer !== 'undefined' && !(buffer instanceof GPUBuffer))) {
      throw new Error(`Kernel "${config.operation}/${config.variant}" binding "${binding.name}" (index ${binding.index}) requires a GPUBuffer.`);
    }
    entries.push({ binding: binding.index, resource });
  }
  return entries;
}
