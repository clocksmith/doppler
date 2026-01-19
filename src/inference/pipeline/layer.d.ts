/**
 * Transformer layer processing (attention + FFN).
 *
 * @module inference/pipeline/layer
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

/**
 * CPU fallback layer processing.
 */
export function processLayerCPU(
  layerIdx: number,
  hiddenStates: Float32Array,
  numTokens: number,
  isPrefill: boolean,
  context: LayerContext
): Promise<Float32Array>;
