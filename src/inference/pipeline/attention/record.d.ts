/**
 * Attention Record - Batched GPU recording path
 *
 * Contains recordLayerAttentionGPU which records attention operations
 * to a shared command encoder without submitting. All operations are
 * batched and submitted together at the end of the forward pass.
 *
 * @module inference/pipeline/attention/record
 */

import type { LayerWeights } from '../types.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../../gpu/weight-buffer.js';
import type { CommandRecorder } from '../../../gpu/kernel-selector.js';
import type { Tensor } from '../../../gpu/tensor.js';
import type { LoRAAdapter } from '../lora.js';
import type { AttentionConfig, AttentionState, AttentionResult, AttentionDebugFlags } from './types.js';

/**
 * Record attention for a single layer (batched, no submit).
 *
 * Uses record* kernel variants to batch all GPU operations into a shared
 * command encoder. No GPU submits happen here - submit once at end of forward pass.
 */
export function recordLayerAttentionGPU(
  recorder: CommandRecorder,
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
