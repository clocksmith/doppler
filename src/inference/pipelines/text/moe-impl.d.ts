/**
 * Mixture of Experts (MoE) feed-forward implementation.
 *
 * This module handles:
 * - Token routing to experts via softmax + top-k
 * - Expert weight loading (on-demand)
 * - Parallel expert execution on GPU
 * - Scatter-add combination of expert outputs
 *
 * Supports multiple MoE architectures:
 * - Mixtral-style (gate/up/down per expert)
 * - GPT-OSS style (MXFP4 quantized fused gate_up + bias)
 *
 * @module inference/pipelines/text/moe-impl
 */

import type { Tensor, TensorDtype } from '../../../gpu/tensor.js';
import type { MoERouter } from '../../moe-router.js';
import type { ExpertWeights } from './types.js';
import type { KernelPathSchema } from '../../../config/schema/index.js';

/**
 * Clear the dequantization cache (call on model unload).
 */
export function clearDequantCache(): void;

/**
 * Get cache stats for debugging.
 */
export function getDequantCacheStats(): {
  hits: number;
  misses: number;
  size: number;
  maxEntries: number;
};

/**
 * Configure dequant cache max entries at runtime.
 */
export function setDequantCacheMaxEntries(maxEntries: number): void;

/**
 * Configuration for MoE feed-forward.
 */
export interface MoEConfig {
  modelType?: string | null;
  hiddenSize: number;
  intermediateSize: number;
  numExperts: number;
  moeTopK: number;
  expertFormat: 'mixtral' | 'gpt-oss';
  hiddenActivation: string;
  swigluLimit: number | null;
  activationDtype?: TensorDtype;
  kernelPath?: KernelPathSchema | null;
}

/**
 * Expert weights with optional GPT-OSS quantized format.
 */
export interface MoEExpertWeights extends ExpertWeights {
  expertFormat: 'mixtral' | 'gpt-oss';
  numExperts?: number;
  gateUpBlocks?: GPUBuffer;
  gateUpScales?: GPUBuffer;
  gateUpBias?: GPUBuffer;
  downBlocks?: GPUBuffer;
  downScales?: GPUBuffer;
  downBias?: GPUBuffer;
}

/**
 * Layer router weights (for models with per-layer routers like GPT-OSS).
 */
export interface LayerRouterWeights {
  weight: Float32Array | GPUBuffer;
  bias: Float32Array | GPUBuffer | null;
}

/**
 * Expert weight loader interface.
 */
export interface ExpertLoader {
  loadExpert(layerIdx: number, expertIdx: number): Promise<MoEExpertWeights | null>;
}

/**
 * MoE feed-forward with CPU routing.
 */
export function moeFeedForwardCPU(
  hiddenStates: Float32Array,
  numTokens: number,
  config: MoEConfig,
  moeRouter: MoERouter,
  expertWeights: Map<string, MoEExpertWeights>,
  expertLoader: ExpertLoader,
  layerIdx: number
): Promise<Float32Array>;

/**
 * MoE feed-forward fully on GPU.
 */
export function moeFeedForwardGPU(
  inputBuffer: GPUBuffer,
  numTokens: number,
  config: MoEConfig,
  moeRouter: MoERouter,
  expertWeights: Map<string, MoEExpertWeights>,
  expertLoader: ExpertLoader,
  layerIdx: number,
  layerRouterWeights?: Map<number, LayerRouterWeights>
): Promise<GPUBuffer>;

/**
 * Check if layer is MoE layer (some models have dense layers too).
 */
export function isMoELayer(layerIdx: number): boolean;
