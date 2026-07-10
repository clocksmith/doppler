import type { WeightBuffer } from '../../../gpu/weight-buffer.js';
import type { LayerWeights } from './types.js';
import type { ProbeConfigSchema } from '../../../config/schema/index.js';

export interface LinearAttentionABProjectionOptions {
  phase: 'decode' | 'prefill';
  numTokens: number;
  hiddenSize: number;
  numVHeads: number;
  layerIdx: number;
  debugProbes?: ProbeConfigSchema[] | null;
  operatorDiagnostics?: { enabled?: boolean } | null;
}

export interface LinearAttentionQKVZProjectionOptions extends LinearAttentionABProjectionOptions {
  convDim: number;
  valueDim: number;
}

export interface LinearAttentionABProjection {
  weight: WeightBuffer;
  outDim: number;
  bProjOffsetElements: number;
}

export interface LinearAttentionQKVZProjection {
  weight: WeightBuffer;
  outDim: number;
}

export declare function resolveLinearAttentionABProjection(
  layerWeights: LayerWeights,
  options: LinearAttentionABProjectionOptions
): LinearAttentionABProjection | null;

export declare function resolveLinearAttentionQKVZProjection(
  layerWeights: LayerWeights,
  options: LinearAttentionQKVZProjectionOptions
): LinearAttentionQKVZProjection | null;
