import { q4kBlockCount, Q4K_BLOCK_BYTES } from '../../config/schema/quantization.schema.js';
import { resolveWeightLayout } from '../manifest-config.js';
import { getQwen35ExpertTensorCandidates } from './expert-loader.js';

/** @type {typeof import('./routed-expert-plan.js').planRoutedExpertBank} */
export function planRoutedExpertBank(manifest, locations, layer) {
  if (!manifest.architecture || typeof manifest.architecture !== 'object'
    || typeof manifest.architecture.numLayers !== 'number' || typeof manifest.architecture.hiddenSize !== 'number'
    || manifest.moeConfig?.expertFormat !== 'qwen3_5' || !Number.isSafeInteger(layer)
    || layer < 0 || layer >= manifest.architecture.numLayers) {
    throw new Error('Routed expert loading requires a declared Qwen packed layer.');
  }
  const numExperts = manifest.moeConfig.numExperts;
  const hiddenSize = manifest.architecture.hiddenSize;
  const intermediateSize = manifest.moeConfig.expertIntermediateSize;
  if (typeof intermediateSize !== 'number' || ![numExperts, hiddenSize, intermediateSize].every(value => Number.isSafeInteger(value) && value > 0)) {
    throw new Error('Routed expert loading requires explicit positive expert geometry.');
  }
  const candidates = getQwen35ExpertTensorCandidates(layer);
  /** @param {string[]} names */
  const first = names => names.find(name => locations.has(name));
  const combined = first(candidates[0]);
  /** @type {Array<[string, string | undefined]>} */
  const names = combined ? [['gateUp', combined], ['down', first(candidates[1])]]
    : [['gate', `blk.${layer}.ffn_gate_exps.weight`], ['up', `blk.${layer}.ffn_up_exps.weight`], ['down', first(candidates[1])]];
  const planes = names.map(([role, name]) => {
    const location = name ? locations.get(name) : null;
    if (!name || !location || location.sourceTransform != null || resolveWeightLayout(location) !== 'row'
      || !['Q4_K', 'Q4_K_M'].includes(location.dtype)) {
      throw new Error(`Routed experts require untransformed row-major Q4K storage: ${name}.`);
    }
    const shape = role === 'down' ? [numExperts, hiddenSize, intermediateSize]
      : [numExperts, (role === 'gateUp' ? 2 : 1) * intermediateSize, hiddenSize];
    if (JSON.stringify(location.shape) !== JSON.stringify(shape)) {
      throw new Error(`Routed expert tensor geometry disagrees with the manifest: ${name}.`);
    }
    const bytes = shape[0] * shape[1] * q4kBlockCount(shape[2]) * Q4K_BLOCK_BYTES;
    if (!Number.isSafeInteger(bytes) || bytes <= 0 || bytes !== location.size) {
      throw new Error(`Routed expert tensor extent disagrees with Q4K blocks: ${name}.`);
    }
    return { role, name, location, bytes, expertBytes: bytes / numExperts };
  });
  return { layer, numExperts, intermediateSize, hiddenSize, planes,
    bankBytes: planes.reduce((sum, plane) => sum + plane.bytes, 0),
    expertBytes: planes.reduce((sum, plane) => sum + plane.expertBytes, 0) };
}
