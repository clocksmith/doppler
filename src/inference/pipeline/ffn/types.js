/**
 * FFN Module Types
 *
 * Shared interfaces and type definitions for FFN operations.
 *
 * @module inference/pipeline/ffn/types
 */

/**
 * Checks if a layer uses MoE (Mixture of Experts) FFN.
 * Inlined to avoid circular dependency with layer.ts.
 * @param {number} layerIdx
 * @param {import('../config.js').ParsedModelConfig} config
 * @param {import('../types.js').LayerWeights | null} [layerWeights]
 * @returns {boolean}
 */
export function isMoELayerLocal(layerIdx, config, layerWeights) {
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
 * @returns {boolean}
 */
export function hasLoggedFusedDownNorm() {
  return loggedFusedDownNorm;
}

/**
 * Mark fused down+norm as logged.
 * @param {boolean} value
 * @returns {void}
 */
export function setLoggedFusedDownNorm(value) {
  loggedFusedDownNorm = value;
}
