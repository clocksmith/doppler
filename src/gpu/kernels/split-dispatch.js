import { GPU_LIMITS, WORKGROUP_SIZES } from './constants.js';

/** @type {typeof import('./split-dispatch.js').planSplitQKVDispatch} */
export function planSplitQKVDispatch(totalElements) {
  if (!Number.isSafeInteger(totalElements) || totalElements <= 0) {
    throw new Error(`split_qkv requires a positive safe integer element count, got ${totalElements}.`);
  }
  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  const x = Math.min(workgroups, GPU_LIMITS.MAX_WORKGROUPS);
  const y = Math.ceil(workgroups / x);
  if (y > GPU_LIMITS.MAX_WORKGROUPS) {
    throw new Error(
      `split_qkv requires ${workgroups} workgroups, exceeding the two-axis WebGPU capacity.`
    );
  }
  return [x, y, 1];
}

