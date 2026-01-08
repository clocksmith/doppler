/**
 * LoRA adapter support for runtime weight deltas.
 *
 * Defines adapter structures and helper lookups for layer modules.
 *
 * @module inference/pipeline/lora
 */

export type { LoRAAdapter, LoRAModuleName, LoRAModuleWeights, LoRALayerMap } from './lora-types.js';
export { LORA_MODULE_ALIASES } from './lora-types.js';

import type { LoRAAdapter, LoRAModuleName, LoRAModuleWeights } from './lora-types.js';

export function getLoRAModule(
  adapter: LoRAAdapter | null | undefined,
  layerIdx: number,
  moduleName: LoRAModuleName
): LoRAModuleWeights | null;
