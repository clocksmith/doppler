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
export declare function isMoELayerLocal(
  layerIdx: number,
  config: ParsedModelConfig,
  layerWeights?: LayerWeights | null
): boolean;

/**
 * Check if fused down+norm has been logged (for one-time trace messages).
 */
export declare function hasLoggedFusedDownNorm(): boolean;

/**
 * Mark fused down+norm as logged.
 */
export declare function setLoggedFusedDownNorm(value: boolean): void;
