import { getOrCreateBindGroupLayout } from './pipeline-cache.js';
import { isDeviceLost } from '../../memory/device-residency.js';

const preparedLayouts = new WeakMap();

export function getKernelBindGroupLayout(config, device) {
  if (device && isDeviceLost(device)) throw new Error('Cannot create a layout on a lost GPU device.');
  const immutable = device && Object.isFrozen(config) && Object.isFrozen(config.bindings)
    && config.bindings.every(Object.isFrozen);
  const prepared = immutable ? preparedLayouts.get(device)?.get(config) : null;
  if (prepared) return prepared;
  const layout = getOrCreateBindGroupLayout(`${config.operation}/${config.variant}`, config.bindings.map((binding) => ({
    binding: binding.index,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: binding.type },
  })), device);
  if (immutable) {
    let layouts = preparedLayouts.get(device);
    if (!layouts) preparedLayouts.set(device, layouts = new WeakMap());
    layouts.set(config, layout);
  }
  return layout;
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
