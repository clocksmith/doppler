/** @type {import('./buffer-identity.js').isNativeGpuBuffer} */
export function isNativeGpuBuffer(value) {
  return typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer;
}
