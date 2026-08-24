import type { LayerContext, LayerWeights } from './types.js';
import type { ParsedModelConfig } from './config.js';

export function resolveActivationDtype(dtype: string | null | undefined): string;
export function getConvLayerState(
  convLayerStates: Map<number, Record<string, unknown>> | null | undefined,
  layerIdx: number
): Record<string, unknown>;
export function isSlidingLayerType(layerType: string | null | undefined): boolean;
export function resolveAttentionRotaryDim(
  config: ParsedModelConfig,
  layerType: string | null | undefined
): number;
export function resolveAttentionFrequencyBaseDim(
  config: ParsedModelConfig,
  layerType: string | null | undefined
): number;
export function resolveAttentionHeadDim(
  config: ParsedModelConfig,
  layerType: string | null | undefined
): number;
export function resolveAttentionNumKVHeads(
  config: ParsedModelConfig,
  layerType: string | null | undefined,
  layerWeights: LayerWeights | null | undefined,
  headDim: number
): number;
export function resolveLayerScalarValue(layerScalar: Float32Array | null | undefined): number;
export function applyLayerScalar(
  layerIdx: number,
  tensor: unknown,
  size: number,
  context: LayerContext,
  layerWeights: LayerWeights | null
): Promise<unknown>;
