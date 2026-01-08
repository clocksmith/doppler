/**
 * LoRA adapter support for runtime weight deltas.
 *
 * Defines adapter structures and helper lookups for layer modules.
 *
 * @module inference/pipeline/lora
 */

export { LORA_MODULE_ALIASES } from './lora-types.js';

/**
 * @param {import('./lora-types.js').LoRAAdapter | null | undefined} adapter
 * @param {number} layerIdx
 * @param {import('./lora-types.js').LoRAModuleName} moduleName
 * @returns {import('./lora-types.js').LoRAModuleWeights | null}
 */
export const getLoRAModule = (adapter, layerIdx, moduleName) => {
  if (!adapter) return null;
  if (adapter.targetModules && !adapter.targetModules.includes(moduleName)) return null;
  const layer = adapter.layers.get(layerIdx);
  if (!layer) return null;
  return layer[moduleName] || null;
};
