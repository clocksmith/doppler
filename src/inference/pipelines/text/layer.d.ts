/**
 * Transformer layer processing (attention + FFN).
 *
 * @module inference/pipelines/text/layer
 */

import type { ParsedModelConfig } from './config.js';
import type { LayerWeights, LayerContext, SandwichNormInfo } from './types.js';

export {
  applyLayerScalar,
  getConvLayerState,
  isSlidingLayerType,
  resolveActivationDtype,
  resolveAttentionFrequencyBaseDim,
  resolveAttentionHeadDim,
  resolveAttentionNumKVHeads,
  resolveAttentionRotaryDim,
  resolveLayerScalarValue,
} from './layer-execution-contract.js';

/**
 * Detect sandwich norm architecture (Gemma 3).
 */
export function detectSandwichNorm(config: ParsedModelConfig | null): SandwichNormInfo;

/**
 * Check if a layer is a MoE layer.
 */
export function isMoELayer(
  layerIdx: number,
  config: ParsedModelConfig,
  layerWeights?: LayerWeights | null
): boolean;

/**
 * Process a single transformer layer.
 */
export function processLayer(
  layerIdx: number,
  hiddenStates: GPUBuffer | Float32Array,
  numTokens: number,
  isPrefill: boolean,
  context: LayerContext
): Promise<GPUBuffer | Float32Array>;

/**
 * GPU-native layer processing (no CPU readbacks).
 */
export function processLayerGPU(
  layerIdx: number,
  inputBuffer: GPUBuffer,
  numTokens: number,
  isPrefill: boolean,
  size: number,
  context: LayerContext
): Promise<GPUBuffer>;

/** True when any entry in `layerTypes` indicates a conv-hybrid layer. */
export declare function hasConvLayers(layerTypes: Array<string> | null | undefined): boolean;

/** Per-layer KV sharing resolver. */
export declare function resolveAttentionKVSharing(
  config: ParsedModelConfig,
  layerIdx: number,
  layerType: string | null | undefined
): Record<string, unknown> | null;

/** True when the model config declares per-layer input blocks. */
export declare function hasPerLayerInputBlock(config: ParsedModelConfig | null): boolean;

/** Apply the per-layer input block transform for `layerIdx`. */
export declare function applyPerLayerInputBlock(
  layerIdx: number,
  hiddenTensor: unknown,
  numTokens: number,
  size: number,
  context: LayerContext,
  layerWeights: LayerWeights | null
): Promise<unknown>;
