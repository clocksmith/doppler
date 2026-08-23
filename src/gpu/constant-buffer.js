import { getDevice } from './device.js';

const constantVectorCache = new WeakMap();

export function getConstantVectorBuffer(size, value, label = 'constant_vector') {
  if (!Number.isInteger(size) || size <= 0 || !Number.isFinite(value)) {
    throw new Error(`Constant vector requires a positive size and finite value; got size=${size}, value=${value}.`);
  }
  const device = getDevice();
  if (!device) throw new Error('No GPU device available for constant vector buffer');
  let perDeviceCache = constantVectorCache.get(device);
  if (!perDeviceCache) {
    perDeviceCache = new Map();
    constantVectorCache.set(device, perDeviceCache);
  }
  const key = `${size}:${value}`;
  const cached = perDeviceCache.get(key);
  if (cached) return cached;
  const data = new Float32Array(size);
  data.fill(value);
  const buffer = device.createBuffer({
    label: `${label}_${size}`,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  perDeviceCache.set(key, buffer);
  return buffer;
}
