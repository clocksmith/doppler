const WORKGROUP_SIZE = 256;

/** @type {typeof import('./rope-dispatch.js').planRoPEPrecomputeDispatch} */
export function planRoPEPrecomputeDispatch(device, count) {
  if (!Number.isSafeInteger(count) || count <= 0) {
    throw new Error(`[RoPE] frequency element count must be a positive safe integer; got "${count}".`);
  }
  const declaredLimit = device?.limits?.maxComputeWorkgroupsPerDimension;
  const maxPerDimension = typeof declaredLimit === 'number' && Number.isFinite(declaredLimit) && declaredLimit > 0
    ? Math.floor(declaredLimit)
    : 65535;
  const totalWorkgroups = Math.ceil(count / WORKGROUP_SIZE);
  const y = Math.ceil(totalWorkgroups / maxPerDimension);
  if (y > maxPerDimension) {
    throw new Error(
      `[RoPE] frequency precompute requires ${totalWorkgroups} workgroups, exceeding ` +
      `the device's ${maxPerDimension} x ${maxPerDimension} dispatch capacity.`
    );
  }
  const x = Math.ceil(totalWorkgroups / y);
  return {
    dispatchStride: x * WORKGROUP_SIZE,
    workgroups: [x, y, 1],
  };
}

