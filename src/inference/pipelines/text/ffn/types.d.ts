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

export interface DenseGateUpInputs {
  layerWeights: import('../types.js').LayerWeights;
  inputTensor: import('../../../../gpu/tensor.js').Tensor;
  phase: 'prefill' | 'decode';
  layerIdx: number;
  kernelPath: import('../types.js').LayerContext['kernelPath'];
  ffnStepPrecision: NonNullable<Parameters<typeof import('./dense-plan.js').resolveDenseFFNMatmulStepDtype>[0]>['ffnStepPrecision'];
  numTokens: number;
  intermediateSize: number;
  hiddenSize: number;
  context: import('../types.js').LayerContext;
  recorder: import('../../../../gpu/command-recorder.js').CommandRecorder | null | undefined;
  lora: import('../types.js').LayerContext['lora'];
  hiddenActivation: ParsedModelConfig['hiddenActivation'];
  swigluLimit: ParsedModelConfig['swigluLimit'];
}
