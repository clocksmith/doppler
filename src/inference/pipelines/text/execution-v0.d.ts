import type {
  ExecutionV0ConfigSchema,
  ExecutionV0FieldSourceMap,
  ExecutionV0PatchSchema,
  ExecutionV0PoliciesSchema,
  ExecutionV0SessionDefaultsSchema,
  ResolvedExecutionV0StepSchema,
} from '../../../config/schema/execution-v0.schema.js';

export interface CompileExecutionV0Options {
  manifestInference?: {
    schema?: string | null;
    execution?: ExecutionV0ConfigSchema['execution'] | null;
    sessionDefaults?: ExecutionV0SessionDefaultsSchema | null;
    model?: Record<string, unknown> | null;
  } | null;
  runtimeInference?: {
    session?: ExecutionV0SessionDefaultsSchema | null;
    executionPatch?: ExecutionV0PatchSchema | null;
  } | null;
  modelId?: string;
  numLayers?: number;
}

export interface CompiledExecutionV0RuntimeState {
  sessionDefaults: ExecutionV0SessionDefaultsSchema;
  policies: ExecutionV0PoliciesSchema;
  resolvedSteps: {
    prefill: ResolvedExecutionV0StepSchema[];
    decode: ResolvedExecutionV0StepSchema[];
    all: ResolvedExecutionV0StepSchema[];
  };
  resolvedSources: {
    session: ExecutionV0FieldSourceMap;
    steps: Record<string, ExecutionV0FieldSourceMap>;
  };
  runtimeInferencePatch: Record<string, unknown>;
}

export interface ApplyExecutionV0RuntimeConfigOptions {
  runtimeConfig?: {
    inference?: Record<string, unknown>;
  } | null;
  manifest?: {
    modelId?: string | null;
    architecture?: {
      numLayers?: number | null;
    } | null;
    inference?: CompileExecutionV0Options['manifestInference'];
  } | null;
  modelId?: string;
  numLayers?: number;
}

export declare function hasExecutionV0(
  manifestInference: CompileExecutionV0Options['manifestInference']
): boolean;

export declare function compileExecutionV0(
  options?: CompileExecutionV0Options
): CompiledExecutionV0RuntimeState | null;

export declare function applyExecutionV0RuntimeConfig(options?: ApplyExecutionV0RuntimeConfigOptions): {
  runtimeConfig: ApplyExecutionV0RuntimeConfigOptions['runtimeConfig'];
  executionV0State: CompiledExecutionV0RuntimeState | null;
};
