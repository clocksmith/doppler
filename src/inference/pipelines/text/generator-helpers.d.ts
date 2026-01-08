/**
 * Generator helper utilities.
 *
 * @module inference/pipelines/text/generator-helpers
 */

import type { PipelineState } from './state.js';
import type { LayerContext } from './types.js';
import type { LogitsConfig, LogitsWeights } from './logits.js';
import type { WeightBufferConfig } from './weights.js';
import type { ExecutionSessionPlan } from './execution-plan.js';

export declare function debugCheckBuffer(
  state: PipelineState,
  buffer: GPUBuffer,
  label: string,
  numTokens: number,
  expectedDim?: number
): Promise<void>;

export declare function buildLayerContext(
  state: PipelineState,
  recorder: unknown,
  isDecodeMode: boolean,
  debugLayers: number[] | null | undefined,
  debugCheckBufferFn?: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>,
  executionPlan?: ExecutionSessionPlan | null
): LayerContext;

export declare function getWeightBufferConfig(state: PipelineState): WeightBufferConfig;

export declare function getLogitsWeights(state: PipelineState): LogitsWeights;

export declare function getLogitsConfig(state: PipelineState): LogitsConfig;
