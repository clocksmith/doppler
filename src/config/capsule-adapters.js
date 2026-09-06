import policy from './capsule-adapters.json' with { type: 'json' };
import { validateCapsuleAdapterExecution } from './capsule-adapter-policy.js';
export { validateCapsuleAdapterExecution } from './capsule-adapter-policy.js';
import { freezeCapsuleV2 } from './capsule-v2.js';
import { normalizeCapsuleObservation } from './capsule-operation.js';

export const CAPSULE_ADAPTER_POLICY = freezeCapsuleV2(policy);
const assert = (ok, message) => { if (!ok) throw new Error(`Capsule adapter: ${message}`); };
const hash = value => /^sha256:[a-f0-9]{64}$/.test(value);


export function resolveCapsuleAdapterSet(input, { capsule, targetPlan, operation }) {
  const entries = freezeCapsuleV2(normalizeCapsuleObservation(input));
  assert(Array.isArray(entries), 'explicit adapter set required');
  if (!entries.length) return entries;
  const declared = validateCapsuleAdapterExecution(targetPlan);
  assert(entries.length <= declared.maxAdapters && declared.operations.includes(operation), 'adapter operation or count is not declared');
  for (const entry of entries) {
    assert(entry.schema === 'doppler.capsule-adapter/v1' && hash(entry.identity), 'exact adapter identity required');
    assert(declared.formats.includes(entry.format), 'format outside declared execution policy');
    assert(entry.baseModel?.modelId === capsule.modelId && entry.baseModel.semanticRoot === capsule.semanticRoot
      && entry.baseModel.envelopeDigest === capsule.envelopeDigest && entry.baseModel.artifactClosureDigest === capsule.artifactClosureDigest,
    'exact base model mismatch');
    const manifest = entry.manifest, artifact = entry.artifact;
    assert(manifest?.baseModel === capsule.modelId && manifest.id === artifact?.artifactId, 'manifest model or adapter mismatch');
    assert(artifact.role === 'lora-weights' && hash(artifact.hash) && Number.isSafeInteger(artifact.sizeBytes) && artifact.sizeBytes > 0,
      'exact adapter artifact required');
    assert((manifest.checksum?.startsWith('sha256:') ? manifest.checksum : `sha256:${manifest.checksum}`) === artifact.hash && manifest.checksumAlgorithm === 'sha256'
      && manifest.weightsSize === artifact.sizeBytes && manifest.weightsPath === artifact.path && typeof artifact.path === 'string'
      && artifact.path.length > 0 && manifest.weightsFormat === 'safetensors'
      && !Object.hasOwn(manifest, 'tensors'), 'single verified safetensors weight artifact required');
    assert(Number.isSafeInteger(manifest.rank) && manifest.rank > 0 && Number.isFinite(manifest.alpha) && manifest.alpha > 0
      && Array.isArray(manifest.targetModules) && manifest.targetModules.length > 0, 'explicit adapter geometry required');
  }
  assert(new Set(entries.map(entry => entry.identity)).size === entries.length, 'duplicate adapters');
  return entries;
}
