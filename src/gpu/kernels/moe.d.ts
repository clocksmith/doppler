/**
 * Mixture of Experts (MoE) Kernels
 *
 * Provides kernels for MoE routing and token distribution:
 * - Top-K expert selection
 * - MoE token gathering (dispatching tokens to experts)
 * - Scatter-add (collecting expert outputs back to tokens)
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

/** MoE kernel options */
export interface MoEOptions extends OutputBufferOptions {
  normalize?: boolean;
  maxTokensPerExpert?: number;
  weightsDtype?: 'f16' | 'f32';
}

/** MoE gather result */
export interface MoEGatherResult {
  gathered: Tensor;
  tokenCounts: GPUBuffer;
  tokenMap: GPUBuffer;
  maxTokensPerExpert: number;
}

/**
 * Run top-K expert selection
 */
export declare function runTopK(
  probs: GPUBuffer,
  numTokens: number,
  numExperts: number,
  topK: number,
  options?: MoEOptions
): Promise<{ indices: GPUBuffer; weights: GPUBuffer }>;

/**
 * Record top-K expert selection (batched, no submit)
 */
export declare function recordTopK(
  recorder: CommandRecorder,
  probs: GPUBuffer,
  numTokens: number,
  numExperts: number,
  topK: number,
  options?: MoEOptions
): Promise<{ indices: GPUBuffer; weights: GPUBuffer }>;

/**
 * Run MoE gather (dispatch tokens to experts)
 */
export declare function runMoEGather(
  hiddenStates: Tensor,
  expertIndices: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  numExperts: number,
  topK: number,
  options?: MoEOptions
): Promise<MoEGatherResult>;

/**
 * Record MoE gather (batched, no submit)
 */
export declare function recordMoEGather(
  recorder: CommandRecorder,
  hiddenStates: Tensor,
  expertIndices: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  numExperts: number,
  topK: number,
  options?: MoEOptions
): Promise<MoEGatherResult>;

/**
 * Run scatter-add (collect expert outputs back to tokens)
 */
export declare function runScatterAdd(
  expertOutputs: Tensor,
  indices: GPUBuffer,
  weights: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  numExperts: number,
  topK: number,
  options?: MoEOptions
): Promise<Tensor>;

/**
 * Record scatter-add (batched, no submit)
 */
export declare function recordScatterAdd(
  recorder: CommandRecorder,
  expertOutputs: Tensor,
  indices: GPUBuffer,
  weights: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  numExperts: number,
  topK: number,
  options?: MoEOptions
): Promise<Tensor>;

/**
 * Run dynamic scatter-add with token offsets
 */
export declare function runScatterAddDynamic(
  expertOutputs: Tensor,
  indices: GPUBuffer,
  weights: GPUBuffer,
  tokenOffsets: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  topK: number,
  options?: MoEOptions
): Promise<Tensor>;

/**
 * Record dynamic scatter-add (batched, no submit)
 */
export declare function recordScatterAddDynamic(
  recorder: CommandRecorder,
  expertOutputs: Tensor,
  indices: GPUBuffer,
  weights: GPUBuffer,
  tokenOffsets: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  topK: number,
  options?: MoEOptions
): Promise<Tensor>;
