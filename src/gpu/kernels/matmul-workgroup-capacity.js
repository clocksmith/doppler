import { getDeviceLimits } from '../device.js';

// Capacity is checked against the current device even for a cached program plan.
/** @type {import('./matmul-workgroup-capacity.js').validateMatmulWorkgroupCapacity} */
export function validateMatmulWorkgroupCapacity(config) {
  if (config.variantMetadata?.workgroupConstants === undefined) return;
  const bytesPerInvocation = config.variantMetadata.workgroupStorageBytesPerInvocation;
  if (typeof bytesPerInvocation !== 'number' || !Number.isSafeInteger(bytesPerInvocation) || bytesPerInvocation <= 0
    || typeof config.sharedMemory !== 'number' || !Number.isSafeInteger(config.sharedMemory) || config.sharedMemory < 0) {
    throw new Error(`Matmul kernel "${config.variant}" requires resolved nonnegative static and positive per-invocation workgroup storage bytes.`);
  }
  const invocations = config.workgroupSize.reduce((product, value) => product * value, 1);
  const storageBytes = config.sharedMemory + invocations * bytesPerInvocation;
  if (!Number.isSafeInteger(storageBytes)) {
    throw new Error(`Matmul kernel "${config.variant}" workgroup storage byte count is not a safe integer.`);
  }
  const limits = getDeviceLimits();
  if (!limits) return;
  /** @type {Array<[keyof import('../device.js').DeviceLimits, number]>} */
  const requirements = [
    ['maxComputeWorkgroupSizeX', config.workgroupSize[0]],
    ['maxComputeWorkgroupSizeY', config.workgroupSize[1]],
    ['maxComputeWorkgroupSizeZ', config.workgroupSize[2]],
    ['maxComputeInvocationsPerWorkgroup', invocations],
    ['maxComputeWorkgroupStorageSize', storageBytes],
  ];
  for (const [name, required] of requirements) {
    const limit = limits[name];
    if (!Number.isSafeInteger(limit) || limit <= 0) {
      throw new Error(`Matmul kernel "${config.variant}" requires a positive device limit ${name}.`);
    }
    if (required > limit) {
      throw new Error(`Matmul kernel "${config.variant}" requires ${required} ${name}, exceeding device limit ${limit}.`);
    }
  }
}
