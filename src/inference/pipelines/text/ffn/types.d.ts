/**
 * FFN Module Types
 *
 * Shared interfaces and type definitions for FFN operations.
 *
 * @module inference/pipelines/text/ffn/types
 */

import type { ParsedModelConfig } from '../config.js';

/**
 * The single MoE-layer test: every layer of a useMoE model, unless the manifest's
 * layerTypes declares explicit 'moe' entries.
 */
export declare function isMoELayer(
  layerIdx: number,
  config: Pick<ParsedModelConfig, 'useMoE' | 'layerTypes'>
): boolean;

/**
 * Check if fused down+norm has been logged (for one-time trace messages).
 */
export declare function hasLoggedFusedDownNorm(): boolean;

/**
 * Mark fused down+norm as logged.
 */
export declare function setLoggedFusedDownNorm(value: boolean): void;
