import { log, isTraceEnabled } from '../../debug/index.js';


export async function createBindGroupWithValidation(device, descriptor, contextLabel) {
  if (!isTraceEnabled('buffers')) {
    return device.createBindGroup(descriptor);
  }

  device.pushErrorScope('validation');
  const bindGroup = device.createBindGroup(descriptor);
  const error = await device.popErrorScope();
  if (error) {
    log.error('Kernels', `${contextLabel} bindGroup validation: ${error.message}`);
  }
  return bindGroup;
}
