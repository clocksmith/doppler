/** @type {import('./subgroup-partition.js').assertSubgroupPartition} */
export function assertSubgroupPartition(lanes, info) {
  const min = info?.subgroupMinSize, max = info?.subgroupMaxSize;
  if (!Number.isSafeInteger(lanes) || lanes < 1 || (lanes & (lanes - 1)) !== 0) {
    throw new Error('Subgroup partition lanes must be a positive power of two.');
  }
  if (typeof min !== 'number' || typeof max !== 'number' || !Number.isSafeInteger(min)
      || !Number.isSafeInteger(max) || min < lanes || max < min || min % lanes !== 0 || max % lanes !== 0) {
    throw new Error(`Kernel requires subgroup partitions of ${lanes} lanes; adapter range ${min}..${max} is unsupported or unavailable.`);
  }
}
