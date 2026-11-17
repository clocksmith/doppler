import { validateQwen35RoutedConfig } from '../../config/schema/runtime/loading.schema.js';

/** @type {typeof import('./tiered-expert-plan.js').planTieredExpertPlacement} */
export function planTieredExpertPlacement(plans, policy, limits, minimumSlots) {
  validateQwen35RoutedConfig(policy);
  const tiered = policy.tiered;
  if (policy.storageMode !== 'tiered-compact' || !tiered || !plans.length
    || !Number.isSafeInteger(minimumSlots) || minimumSlots < 1
    || ![limits.maxBufferSize, limits.maxStorageBufferBindingSize].every(value => Number.isSafeInteger(value) && value > 0)) {
    throw new Error('Tiered expert placement requires assigned plans, declared top-K and device limits.');
  }
  const numLayers = plans[0].numLayers;
  if (new Set(plans.map(plan => plan.layer)).size !== plans.length
    || plans.some(plan => plan.numLayers !== numLayers || !Number.isSafeInteger(plan.layer)
      || plan.layer < 0 || plan.layer >= numLayers || !Number.isSafeInteger(plan.bankBytes) || plan.bankBytes < 1
      || !Number.isSafeInteger(plan.expertBytes) || plan.expertBytes < 1 || plan.expertBytes % 4
      || !Number.isSafeInteger(plan.numExperts) || plan.numExperts < minimumSlots
      || plan.bankBytes !== plan.numExperts * plan.expertBytes
      || !plan.planes.length || plan.planes.some(plane => !Number.isSafeInteger(plane.expertBytes)
        || plane.expertBytes < 1 || plane.expertBytes % 4 || plane.bytes !== plane.expertBytes * plan.numExperts)
      || plan.planes.reduce((sum, plane) => sum + plane.bytes, 0) !== plan.bankBytes)
    || [...tiered.residentLayers, ...tiered.parkedLayers].some(layer => layer >= numLayers)) {
    throw new Error('Tiered expert geometry must preserve assigned original layers and complete packed planes.');
  }
  const resident = new Set(tiered.residentLayers), parked = new Set(tiered.parkedLayers);
  const residentBytes = plans.filter(plan => resident.has(plan.layer)).reduce((sum, plan) => sum + plan.bankBytes, 0);
  const parkedBytes = plans.filter(plan => parked.has(plan.layer)).reduce((sum, plan) => sum + plan.bankBytes, 0);
  const offloaded = plans.filter(plan => !resident.has(plan.layer));
  const remaining = policy.cacheBytes - residentBytes - parkedBytes - tiered.mappedCacheBytes;
  if (!Number.isSafeInteger(remaining) || remaining < 0
    || plans.some(plan => resident.has(plan.layer)
      ? plan.planes.some(plane => plane.bytes > Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize))
      : plan.expertBytes > limits.maxBufferSize)) {
    throw new Error('Resident, parked and mapped expert storage exceeds the declared cache or device bounds.');
  }
  const supplied = policy.compactLayerBytes;
  if (supplied !== null && (Object.keys(supplied).length !== offloaded.length
    || offloaded.some(plan => !Object.hasOwn(supplied, plan.layer)))) {
    throw new Error('Tiered compact quotas must cover exactly the assigned nonresident layers.');
  }
  if (supplied !== null && Object.values(supplied).reduce((sum, bytes) => sum + bytes, 0) > remaining) {
    throw new Error('Tiered compact quotas exceed the remaining model cache budget.');
  }
  /** @type {Map<number, import('./tiered-expert-plan.js').TieredExpertLayer>} */
  const layers = new Map();
  /** @type {Record<string, number>} */
  const compactLayerBytes = {};
  let compactBytes = 0;
  for (const plan of plans) {
    if (resident.has(plan.layer)) {
      layers.set(plan.layer, { plan, mode: 'resident', capacity: plan.numExperts, quotaBytes: plan.bankBytes });
      continue;
    }
    const quota = supplied === null ? Math.floor(remaining / offloaded.length / 4) * 4 : supplied[plan.layer];
    const mapBytes = plan.numExperts * 4;
    const capacity = Math.min(plan.numExperts, Math.floor((quota - mapBytes) / plan.expertBytes),
      ...plan.planes.map(plane => Math.floor(Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize) / plane.expertBytes)));
    if (!Number.isSafeInteger(capacity) || capacity < minimumSlots
      || mapBytes > Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize)) {
      throw new Error(`Tiered expert layer ${plan.layer} cannot hold one complete top-K selection.`);
    }
    const actual = mapBytes + capacity * plan.expertBytes;
    compactBytes += actual;
    compactLayerBytes[plan.layer] = actual;
    layers.set(plan.layer, { plan, mode: parked.has(plan.layer) ? 'parked' : 'streamed', capacity, quotaBytes: actual });
  }
  if (compactBytes > remaining) throw new Error('Tiered compact quotas exceed the remaining model cache budget.');
  const cacheUpperBytes = residentBytes + parkedBytes + tiered.mappedCacheBytes + compactBytes;
  if (!Number.isSafeInteger(cacheUpperBytes) || cacheUpperBytes > policy.cacheBytes) {
    throw new Error('Tiered expert allocation sum exceeds cacheBytes.');
  }
  return { layers, residentBytes, parkedBytes, mappedCacheBytes: tiered.mappedCacheBytes,
    compactBytes, cacheUpperBytes, compactLayerBytes };
}
