import type { ExecutionV1PerLayerInputsSessionSchema } from '../../../../config/schema/execution-v1.schema.js';
import type { ExecutionSessionPlan } from '../execution-plan.js';
import type { PipelineState } from '../state.js';
import type { LayerContext } from '../types.js';

export declare function resolvePerLayerInputsSession(
  manifestSession: ExecutionV1PerLayerInputsSessionSchema | null | undefined,
  runtimeSession: ExecutionV1PerLayerInputsSessionSchema | Record<string, unknown> | null | undefined
): ExecutionV1PerLayerInputsSessionSchema | null;

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
  debugCheckBufferFn?: (
    buffer: GPUBuffer,
    label: string,
    numTokens: number,
    expectedDim?: number
  ) => Promise<void>,
  executionPlan?: ExecutionSessionPlan | null
): LayerContext;
