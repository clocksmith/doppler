import { ggufBlockFormat, isGgufBlockSourceDtype } from '../formats/gguf/block-storage.js';
import { getKernelPathMatmulVariant } from '../config/kernel-path-loader.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { getKernelConfigs } from '../config/kernel-registry-contract.js';
import { LAYER_PREFIXES, LAYER_WEIGHT_CONSUMERS } from './layer-tensor-aliases.js';

// Resolve the same first-existing binding used by layer loading, before any
// weight allocation. Equal immutable storage and tied endpoints share demand.
/** @type {import('./weight-representation-plan.js').compileWeightRepresentationPlan} */
export function compileWeightRepresentationPlan(locations, manifest, paths, defaults) {
  const architecture = manifest.architecture;
  if (!architecture || typeof architecture !== 'object' || !('numLayers' in architecture)
    || typeof architecture.numLayers !== 'number' || !Number.isInteger(architecture.numLayers) || architecture.numLayers < 1) {
    throw new Error('Representation planning requires explicit architecture.numLayers.');
  }
  const bound = new Set();
  /** @typedef {{names: string[], consumers: import('./weight-representation-plan.js').WeightConsumer[], dtypes: Set<string>, unresolved: boolean}} Demand */
  /** @type {Map<string, Demand>} */
  const groups = new Map();
  /** @type {Map<string, Demand>} */
  const entries = new Map();
  for (const [name, location] of locations) {
    const ranges = location.spans ?? (location.shardIndex != null && location.offset != null
      ? [{ shardIndex: location.shardIndex, offset: location.offset, size: location.size }] : null);
    const key = ranges ? JSON.stringify([ranges, location.dtype, location.shape,
      location.layout ?? null, location.sourceTransform ?? null, location.storage ?? null, location.originalShape ?? null]) : name;
    let group = groups.get(key);
    if (!group) { group = { names: [], consumers: [], dtypes: new Set(), unresolved: false }; groups.set(key, group); }
    group.names.push(name); entries.set(name, group);
  }
  /** @param {string} name @param {string} role @param {number} layer
   * @param {readonly ('prefill' | 'decode')[]} phases @param {boolean} [optional] */
  const add = (name, role, layer, phases, optional = false) => {
    bound.add(name);
    const group = entries.get(name);
    if (!group) throw new Error(`Unknown tensor ${name}`);
    for (let index = 0; index < paths.length; index++) {
      const path = paths[index];
      for (const phase of phases) {
        const variant = getKernelPathMatmulVariant(role, phase, layer, path);
        const config = variant == null ? null : getKernelConfigs().matmul?.[variant];
        if (optional && variant == null) continue;
        const declaredWeightDtype = config?.weightDtype ?? null;
        const storedDtype = locations.get(name)?.dtype.toLowerCase() ?? null;
        const floatFallback = ((storedDtype === 'f16' && declaredWeightDtype === 'q4k') || (['f16','f32','bf16'].includes(storedDtype ?? '') && declaredWeightDtype === 'q4_0'))
          && selectRuleValue('kernels', 'matmul', 'weightDtypeMismatchPolicy', {
            actualWeightDtype: storedDtype, requiredWeightDtype: declaredWeightDtype,
          }) === 'fallthrough';
        const dtype = floatFallback ? storedDtype : declaredWeightDtype;
        group.consumers.push(Object.freeze({ tensor: name, role, layer, phase,
          path: path.id, variant, declaredWeightDtype, weightDtype: dtype }));
        if (dtype) group.dtypes.add(dtype); else group.unresolved = true;
      }
    }
  };
  for (let layer = 0; layer < architecture.numLayers; layer++) {
    for (const { role, suffixes, additionalRoles } of LAYER_WEIGHT_CONSUMERS) {
      let found;
      for (const prefix of LAYER_PREFIXES(layer)) {
        found = suffixes.map(suffix => `${prefix}.${suffix}`).find(name => locations.has(name));
        if (found) break;
      }
      if (found) {
        add(found, role, layer, ['prefill', 'decode']);
        for (const additional of additionalRoles) add(found, additional, layer, ['prefill', 'decode'], true);
      }
    }
  }
  const embeddings = [...locations].filter(([, loc]) => loc.role === 'embedding');
  const heads = [...locations].filter(([, loc]) => loc.role === 'lm_head');
  for (const [name, location] of embeddings) {
    const group = entries.get(name);
    if (!group) throw new Error(`Unknown embedding ${name}`);
    group.dtypes.add(['F16', 'F32', 'BF16'].includes(location.dtype) ? location.dtype.toLowerCase() : 'f32');
    group.unresolved = true;
  }
  for (const [name] of heads) { add(name, 'lm_head', 0, ['decode']); add(name, 'lm_head_prefill', 0, ['prefill']); }
  if (manifest.inference?.output?.tieWordEmbeddings === true) {
    for (const [name, embedding] of embeddings) {
      if (heads.length === 0) { add(name, 'lm_head', 0, ['decode']); add(name, 'lm_head_prefill', 0, ['prefill']); }
      for (const [headName, head] of heads) {
        if (!['F16', 'F32', 'BF16'].includes(head.dtype) || head.dtype !== embedding.dtype
          || JSON.stringify(head.shape) !== JSON.stringify(embedding.shape)
          || (head.layout ?? 'row') !== (embedding.layout ?? 'row')) continue;
        const target = entries.get(name), source = entries.get(headName);
        if (!target || !source) throw new Error(`Unknown tied endpoint ${name}`);
        if (target === source) continue;
        target.names.push(...source.names); target.consumers.push(...source.consumers);
        for (const dtype of source.dtypes) target.dtypes.add(dtype);
        target.unresolved ||= source.unresolved;
        for (const alias of source.names) entries.set(alias, target);
      }
    }
  }
  for (const [name, group] of entries) {
    if (!bound.has(name)) group.unresolved = true;
  }
  /** @type {Record<string, import('./weight-representation-plan.js').WeightRepresentationRequirement>} */
  const result = Object.create(null);
  for (const [name, location] of locations) {
    const group = entries.get(name);
    if (!group) throw new Error(`Unknown tensor ${name}`);
    if ((isGgufBlockSourceDtype(location.dtype) && location.dtype !== 'Q6_K') || (location.dtype === 'Q6_K' && manifest.quantizationInfo?.preserveSourceTensors === true)) {
      const packedDtype=ggufBlockFormat(location.dtype).variant;
      const incompatible = group.consumers.filter(consumer => consumer.weightDtype != null && consumer.weightDtype !== packedDtype);
      if (incompatible.length) throw new Error(`Q4_0 tensor ${name} has consumers declaring another storage encoding.`);
      result[name] = Object.freeze({ mode: 'packed-only', primaryDtype: null, complete: !group.unresolved,
        dtypes: Object.freeze([packedDtype]), aliases: Object.freeze([...group.names]),
        consumers: Object.freeze([...group.consumers]), keepF32Weights: false });
      continue;
    }
    if (location.dtype === 'FP8_E4M3FN' || location.dtype === 'FP8_E5M2') {
      const expected = selectRuleValue('loader', 'weights', 'matmulWeightDtype', { locationDtype: location.dtype });
      if (!group.consumers.length || group.unresolved || [...group.dtypes].some(dtype => dtype !== expected)) {
        throw new Error(`FP8 tensor ${name} requires every selected consumer to declare its exact packed encoding.`);
      }
      result[name] = Object.freeze({ mode: 'packed-only', primaryDtype: null, complete: true,
        dtypes: Object.freeze([...group.dtypes]), aliases: Object.freeze([...group.names].sort()),
        consumers: Object.freeze([...group.consumers]), keepF32Weights: false });
      continue;
    }
    const q4 = ['Q4_K', 'Q4_K_M'].includes(location.dtype);
    const complete = group.consumers.length > 0 && !group.unresolved;
    const dtypes = [...group.dtypes].sort();
    // Unknown/non-matmul consumers keep the existing materialization contract.
    // They never acquire a packed-only proof from a different tensor's role.
    const packed = q4 && (dtypes.includes('q4k') || (!complete && defaults.useFusedQ4K));
    const dense = !q4 || dtypes.some(dtype => dtype !== 'q4k') || !complete;
    const denseDtypes = dtypes.filter(dtype => dtype !== 'q4k');
    if (q4 && denseDtypes.some(dtype => !['f16', 'f32'].includes(dtype))) {
      throw new Error(`Tensor "${name}" requires unsupported Q4 dense representations: ${denseDtypes.join(', ')}.`);
    }
    if (packed && defaults.q4kLayout !== 'row') throw new Error(`Tensor "${name}" requires row-packed Q4 storage.`);
    result[name] = Object.freeze({ mode: packed ? (dense ? 'both' : 'packed-only') : 'dense-only',
      primaryDtype: group.consumers.some(c => c.declaredWeightDtype === 'q4k' && c.weightDtype === 'f16') ? 'f16' : null,
      complete, dtypes: Object.freeze(dtypes), aliases: Object.freeze([...group.names].sort()),
      consumers: Object.freeze([...group.consumers]),
      keepF32Weights: manifest.quantizationInfo?.preserveSourceTensors === true && location.dtype === 'F32'
        ? true : complete && denseDtypes.length > 0 ? denseDtypes.includes('f32') : defaults.keepF32Weights });
  }
  return Object.freeze(result);
}
