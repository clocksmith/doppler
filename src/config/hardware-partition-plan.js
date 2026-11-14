import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { createOrderedLayerPartitionPlanFromManifest } from './layer-partition-plan.js';

/** @param {number} value */
const finiteNonnegative = value => Number.isFinite(value) && value >= 0;
/** @param {number} value */
const bytes = value => Number.isSafeInteger(value) && value >= 0;
/** @param {import('./hardware-partition-plan.js').PartitionHardwareProfile['layers'][number]} layer */
const validLayer = layer => layer && bytes(layer.residentBytes) && bytes(layer.scratchBytes) && finiteNonnegative(layer.elapsedMs);
/** @param {string} value */
const digest = value => typeof value === 'string' && /^sha256:[a-f0-9]{64}$/.test(value);

/** Prefix costs and suffix scratch maxima keep candidate enumeration linear, not quadratic.
 * @param {import('./hardware-partition-plan.js').PartitionHardwareProfile} profile */
function costs(profile) {
  const resident = [0], elapsed = [0], prefixScratch = [0];
  const suffixScratch = new Array(profile.layers.length + 1).fill(0);
  for (const layer of profile.layers) {
    resident.push(resident[resident.length - 1] + layer.residentBytes);
    elapsed.push(elapsed[elapsed.length - 1] + layer.elapsedMs);
    prefixScratch.push(Math.max(prefixScratch[prefixScratch.length - 1], layer.scratchBytes));
  }
  for (let index = profile.layers.length - 1; index >= 0; index--) {
    suffixScratch[index] = Math.max(suffixScratch[index + 1], profile.layers[index].scratchBytes);
  }
  return { resident, elapsed, prefixScratch, suffixScratch };
}

/** No GPU work, implicit hardware priors, numerical changes or existing-route selection.
 * @type {import('./hardware-partition-plan.js').proposeHardwarePartitionPlan} */
export function proposeHardwarePartitionPlan(manifest, request) {
  if (!request || !bytes(request.maxLayers) || request.maxLayers < 2
    || !bytes(request.nowMs) || !bytes(request.maxProfileAgeMs) || !digest(request.workloadIdentity)
    || !Array.isArray(request.profiles) || request.profiles.length !== 2) {
    throw new Error('Hardware partition request requires two profiles and explicit workload, freshness and layer bounds.');
  }
  // Canonical owner validates architecture, precision and complete residual-stream boundaries.
  const seed = createOrderedLayerPartitionPlanFromManifest(manifest, { ...request.activation, splitLayers: [1] });
  const count = seed.totalLayers;
  if (count > request.maxLayers) throw new Error('Model exceeds the requested hardware partition layer bound.');
  if (!Array.isArray(request.transferMs) || request.transferMs.length !== count - 1
    || !request.transferMs.every(finiteNonnegative)) throw new Error('Measured transfer estimate required for every interior boundary.');
  const manifestHash = computeCanonicalSha256(manifest);
  for (const profile of request.profiles) {
    if (!profile || typeof profile.hardwareId !== 'string' || !profile.hardwareId
      || typeof profile.sourceReceipt !== 'string' || !profile.sourceReceipt
      || profile.manifestHash !== manifestHash || profile.workloadIdentity !== request.workloadIdentity
      || !bytes(profile.measuredAtMs) || profile.measuredAtMs > request.nowMs
      || request.nowMs - profile.measuredAtMs > request.maxProfileAgeMs
      || !bytes(profile.memoryBudgetBytes) || !bytes(profile.fixedBytes) || !finiteNonnegative(profile.fixedMs)
      || !Array.isArray(profile.layers) || profile.layers.length !== count
      || !profile.layers.every(validLayer)) {
      throw new Error('Hardware profile must match the manifest/workload with fresh evidence and complete memory/timing costs.');
    }
  }
  const [first, second] = request.profiles;
  if (first.hardwareId === second.hardwareId) throw new Error('Hardware partition devices must be distinct.');
  const [a, b] = request.profiles.map(costs);
  let bestSplit = 0, estimatedMs = Infinity, feasibleSplits = 0;
  /** @type {number[]} */
  let memoryBytes = [];
  for (let split = 1; split < count; split++) {
    const memory = [first.fixedBytes + a.resident[split] + a.prefixScratch[split],
      second.fixedBytes + b.resident[count] - b.resident[split] + b.suffixScratch[split]];
    const elapsed = first.fixedMs + a.elapsed[split] + second.fixedMs
      + b.elapsed[count] - b.elapsed[split] + request.transferMs[split - 1];
    if (!memory.every(bytes) || !Number.isFinite(elapsed)) throw new Error('Hardware profile aggregate overflows supported cost bounds.');
    if (memory[0] > first.memoryBudgetBytes || memory[1] > second.memoryBudgetBytes) continue;
    feasibleSplits++;
    // Equal estimates retain the earlier boundary for deterministic output.
    if (elapsed < estimatedMs) { bestSplit = split; estimatedMs = elapsed; memoryBytes = memory; }
  }
  if (!bestSplit) throw new Error('No feasible nonempty two-device partition fits the declared memory budgets.');
  const plan = createOrderedLayerPartitionPlanFromManifest(manifest, { ...request.activation, splitLayers: [bestSplit] });
  return Object.freeze({ plan, manifestHash, workloadIdentity: request.workloadIdentity, estimatedMs,
    memoryBytes: Object.freeze(memoryBytes), hardwareIds: Object.freeze(request.profiles.map(profile => profile.hardwareId)),
    sourceReceipts: Object.freeze(request.profiles.map(profile => profile.sourceReceipt)), evaluatedSplits: count - 1, feasibleSplits });
}
