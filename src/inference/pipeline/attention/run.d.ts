/**
 * Attention Run - Immediate GPU submission path
 *
 * Contains runLayerAttentionGPU which executes attention operations
 * with immediate GPU submission (each kernel submits independently).
 *
 * @module inference/pipeline/attention/run
 */

import type { LayerWeights } from '../types.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../../gpu/weight-buffer.js';
import type { Tensor } from '../../../gpu/tensor.js';
import type { LoRAAdapter } from '../lora.js';
import type { AttentionConfig, AttentionState, AttentionResult, AttentionDebugFlags } from './types.js';

/**
 * Run attention for a single layer (GPU path).
 *
 * @param input - Input hidden states tensor
 * @param layerWeights - Weights for this layer
 * @param config - Attention configuration
 * @param state - Shared state (RoPE freqs, KV cache)
 * @param debug - Debug mode flag
 * @param debugFlags - Mutable debug flags to prevent repeated logging
 * @returns Output tensor after attention
 */
export function runLayerAttentionGPU(
  input: Tensor,
  layerWeights: LayerWeights | null,
  config: AttentionConfig,
  state: AttentionState,
  debug?: boolean,
  debugFlags?: AttentionDebugFlags,
  getWeightBuffer?: (weight: GPUBuffer | WeightBuffer | Float32Array | ArrayBuffer | CpuWeightBuffer, label: string) => GPUBuffer | WeightBuffer,
  getNormWeightBuffer?: (weight: GPUBuffer | Float32Array | ArrayBuffer | CpuWeightBuffer, label: string) => GPUBuffer,
  debugCheckBuffer?: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>,
  lora?: LoRAAdapter | null
): Promise<AttentionResult>;
