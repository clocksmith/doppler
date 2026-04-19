/**
 * Generator helper utilities.
 *
 * @module inference/pipelines/text/generator-helpers
 */

import type { PipelineState } from './state.js';
import type { LayerContext } from './types.js';
import type { LogitsConfig, LogitsWeights } from './logits/index.js';
import type { WeightBufferConfig } from './weights.js';
import type { ExecutionSessionPlan } from './execution-plan.js';
import type { ExecutionV1PerLayerInputsSessionSchema } from '../../../config/schema/execution-v1.schema.js';

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

export declare function resolvePerLayerInputsSession(
  manifestSession: ExecutionV1PerLayerInputsSessionSchema | null | undefined,
  runtimeSession: ExecutionV1PerLayerInputsSessionSchema | Record<string, unknown> | null | undefined
): ExecutionV1PerLayerInputsSessionSchema | null;

export declare function releaseSharedAttentionState(
  sharedAttentionState: LayerContext['sharedAttentionState'],
  recorder?: {
    trackTemporaryBuffer(buffer: GPUBuffer): void;
  } | null
): void;

export declare function getLogitsWeights(state: PipelineState): LogitsWeights;

export declare function getLogitsConfig(state: PipelineState): LogitsConfig;
