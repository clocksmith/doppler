import { getRuntimeConfig } from '../config/runtime.js';
import { log } from '../debug/index.js';

let didLogQueryClamp = false;
let didLogQueryFallback = false;

/** @type {import('./recorder-timestamps.js').createRecorderTimestampResources} */
export function createRecorderTimestampResources(device, label) {
  let querySet = null;
  let queryBuffer = null;
  let readbackBuffer = null;
  try {
    const runtimeProfiler = getRuntimeConfig().shared?.debug?.profiler;
    if (!runtimeProfiler) {
      throw new Error('runtime.shared.debug.profiler is required.');
    }
    const { maxQueries, defaultQueryLimit } = runtimeProfiler;
    const deviceLimit = device.limits?.maxQuerySetSize;
    const hasDeviceLimit = typeof deviceLimit === 'number' && Number.isFinite(deviceLimit) && deviceLimit > 0;
    const limit = hasDeviceLimit
      ? deviceLimit
      : defaultQueryLimit;
    const capacity = Math.min(maxQueries, limit);
    if (hasDeviceLimit && capacity < maxQueries && !didLogQueryClamp) {
      log.warn(
        'CommandRecorder',
        `Clamping MAX_QUERIES to device limit: ${capacity}/${maxQueries}`
      );
      didLogQueryClamp = true;
    } else if (!hasDeviceLimit && !didLogQueryFallback) {
      log.warn(
        'CommandRecorder',
        `maxQuerySetSize unavailable; using fallback ${defaultQueryLimit}`
      );
      didLogQueryFallback = true;
    }

    querySet = device.createQuerySet({
      type: 'timestamp',
      count: capacity,
    });

    // Buffer to hold query results (8 bytes per timestamp = BigUint64)
    queryBuffer = device.createBuffer({
      label: `${label}_query_buffer`,
      size: capacity * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });

    // Readback buffer
    readbackBuffer = device.createBuffer({
      label: `${label}_readback_buffer`,
      size: capacity * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    return { querySet, queryBuffer, readbackBuffer, capacity };
  } catch (error) {
    readbackBuffer?.destroy();
    queryBuffer?.destroy();
    querySet?.destroy();
    throw error;
  }
}
