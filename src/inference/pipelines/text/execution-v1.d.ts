import type {
  ExecutionV1GraphSchema,
  ExecutionV1SessionSchema,
  ExecutionV1PoliciesSchema,
  ExecutionV1ExpandedStepSchema,
} from '../../../config/schema/execution-v1.schema.js';

export interface ExecutionV1CompiledState {
  session: ExecutionV1SessionSchema;
  policies: ExecutionV1PoliciesSchema;
  resolvedSteps: {
    prefill: ExecutionV1ExpandedStepSchema[];
    decode: ExecutionV1ExpandedStepSchema[];
    all: ExecutionV1ExpandedStepSchema[];
  };
  runtimeInferencePatch: Record<string, unknown>;
}

export declare function hasExecutionV1(
  manifestInference: { schema?: string | null; execution?: unknown }
): boolean;

export declare function compileExecutionV1(options?: {
  manifestInference: {
    schema: string;
    execution: ExecutionV1GraphSchema;
    session: ExecutionV1SessionSchema;
  };
  modelId?: string;
  numLayers?: number;
  runtimeSession?: ExecutionV1SessionSchema | null;
}): ExecutionV1CompiledState;

export declare function applyExecutionV1RuntimeConfig(options?: {
  runtimeConfig: Record<string, unknown>;
  manifest: {
    inference?: { schema?: string; execution?: ExecutionV1GraphSchema; session?: ExecutionV1SessionSchema };
    modelId?: string;
    architecture?: { numLayers?: number };
  };
  modelId?: string;
  numLayers?: number;
}): {
  runtimeConfig: Record<string, unknown>;
  executionV1State: ExecutionV1CompiledState | null;
};
