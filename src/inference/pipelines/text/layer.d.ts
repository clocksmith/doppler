/**
 * Transformer layer processing (attention + FFN).
 *
 * @module inference/pipelines/text/layer
 */

import type { ParsedModelConfig } from './config.js';
import type { LayerWeights, LayerContext, SandwichNormInfo } from './types.js';

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

/** Resolve the activation dtype for a layer (normalizes aliases). */
export declare function resolveActivationDtype(dtype: string | null | undefined): string;

/** Fetch or lazily create the per-layer convolution state record. */
export declare function getConvLayerState(
  convLayerStates: Record<string, unknown>,
  layerIdx: number
): Record<string, unknown>;

/** True when `layerType` is a sliding-window attention variant. */
export declare function isSlidingLayerType(layerType: string | null | undefined): boolean;

/** True when any entry in `layerTypes` indicates a conv-hybrid layer. */
export declare function hasConvLayers(layerTypes: Array<string> | null | undefined): boolean;

/** Per-layer rotary dimension resolver (handles layerType-driven overrides). */
export declare function resolveAttentionRotaryDim(
  config: ParsedModelConfig,
  layerType: string | null | undefined
): number;

/** Per-layer attention head dimension resolver. */
export declare function resolveAttentionHeadDim(
  config: ParsedModelConfig,
  layerType: string | null | undefined
): number;

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

