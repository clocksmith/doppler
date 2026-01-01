/**
 * LoRA adapter support for runtime weight deltas.
 *
 * Defines adapter structures and helper lookups for layer modules.
 *
 * @module inference/pipeline/lora
 */

import type { LoRAAdapter, LoRAModuleName, LoRAModuleWeights } from './lora-types.js';

export type { LoRAAdapter, LoRAModuleName, LoRAModuleWeights, LoRALayerMap } from './lora-types.js';
export { LORA_MODULE_ALIASES } from './lora-types.js';

export const getLoRAModule = (
  adapter: LoRAAdapter | null | undefined,
  layerIdx: number,
  moduleName: LoRAModuleName
): LoRAModuleWeights | null => {
  if (!adapter) return null;
  if (adapter.targetModules && !adapter.targetModules.includes(moduleName)) return null;
  const layer = adapter.layers.get(layerIdx);
  if (!layer) return null;
  return layer[moduleName] || null;
};
