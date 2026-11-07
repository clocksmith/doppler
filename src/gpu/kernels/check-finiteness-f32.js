import { unifiedKernelWrapper } from './kernel-execution.js';
import { getKernelConfig } from './kernel-configs.js';

/** A recorded F32 check. Status is caller-owned and initialized before recording.
 * @type {typeof import('./check-finiteness-f32.js').recordCheckFinitenessF32} */
export async function recordCheckFinitenessF32(recorder, input, size, status, layer, step, absThreshold) {
  if (!recorder || recorder.getStats().submitted || !Number.isSafeInteger(size) || size < 1
    || size > 0xffffffff || input.size < size * 4 || !(input.usage & GPUBufferUsage.STORAGE)
    || status.size < 16 || !(status.usage & GPUBufferUsage.STORAGE) || input === status
    || ![layer, step].every(value => Number.isSafeInteger(value) && value >= 0 && value <= 0xffffffff)
    || !Number.isFinite(absThreshold) || absThreshold <= 0 || !Number.isFinite(Math.fround(absThreshold))) {
    throw new Error('F32 finiteness requires bounded input, separate status storage and explicit finite limits.');
  }
  const config = getKernelConfig('check_finiteness', 'f32');
  const groups = Math.ceil(size / config.workgroupSize[0]);
  if (groups > recorder.device.limits.maxComputeWorkgroupsPerDimension) {
    throw new Error('F32 finiteness check exceeds its declared one-dimensional dispatch capacity.');
  }
  await unifiedKernelWrapper('check_finiteness', recorder, 'f32', [input, status], {
    size, layer, step, abs_threshold: absThreshold,
  }, [groups, 1, 1]);
}
