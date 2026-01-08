/**
 * FFN Module Types
 *
 * Shared interfaces and type definitions for FFN operations.
 *
 * @module inference/pipeline/ffn/types
 */

import type { ParsedModelConfig } from '../config.js';
import type { LayerWeights } from '../types.js';

/**
 * Checks if a layer uses MoE (Mixture of Experts) FFN.
 * Inlined to avoid circular dependency with layer.ts.
 */
export function isMoELayerLocal(
  layerIdx: number,
  config: ParsedModelConfig,
  layerWeights?: LayerWeights | null
): boolean {
  if (!config.useMoE) return false;
  if (layerWeights?.routerWeight) return true;
  const layerTypes = config.layerTypes;
  if (Array.isArray(layerTypes) && layerIdx < layerTypes.length) {
    return layerTypes[layerIdx] === 'moe';
  }
  return true;
}

// Track if we've logged one-time messages
let loggedFusedDownNorm = false;

/**
 * Check if fused down+norm has been logged (for one-time trace messages).
 */
export function hasLoggedFusedDownNorm(): boolean {
  return loggedFusedDownNorm;
}

/**
 * Mark fused down+norm as logged.
 */
export function setLoggedFusedDownNorm(value: boolean): void {
  loggedFusedDownNorm = value;
}
