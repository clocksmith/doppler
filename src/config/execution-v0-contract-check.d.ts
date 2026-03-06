import type {
  ExecutionV0ComputeDefaultsSchema,
  ExecutionV0KernelProfileSchema,
  ExecutionV0KernelRefSchema,
  ExecutionV0KVIO,
  ExecutionV0PrecisionSchema,
  ExecutionV0SessionDefaultsSchema,
  ExecutionV0StepSchema,
} from './schema/execution-v0.schema.js';

export interface ExecutionV0ContractCheckResult {
  id: string;
  ok: boolean;
}

export interface ExecutionV0ContractPerStep {
  precision: {
    inputDtype: string | null;
    mathDtype: string | null;
    accumDtype: string | null;
    outputDtype: string | null;
  };
  precisionSources: {
    inputDtype: 'manifest' | 'kernelProfile' | 'sessionDefault' | 'derived';
    mathDtype: 'manifest' | 'kernelProfile' | 'sessionDefault' | 'derived';
    accumDtype: 'manifest' | 'kernelProfile' | 'sessionDefault' | 'derived';
    outputDtype: 'manifest' | 'kernelProfile' | 'sessionDefault' | 'derived';
  };
  resolvedPrecision?: {
    inputDtype: 'f16' | 'f32' | null;
    mathDtype: 'f16' | 'f32';
    accumDtype: 'f16' | 'f32';
    outputDtype: 'f16' | 'f32';
  };
  kvIO?: ExecutionV0KVIO;
  kvIOSource?: 'manifest' | 'kernelProfile' | 'sessionDefault';
}

export interface ExecutionV0ContractArtifact {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: ExecutionV0ContractCheckResult[];
  errors: string[];
  stats: {
    kernelProfiles: number;
    pinnedSteps: number;
  };
  perStep: Record<string, ExecutionV0ContractPerStep>;
}

export declare function normalizeExecutionV0Dtype(value: unknown, label: string): 'f16' | 'f32';
export declare function buildExecutionV0KernelProfileKey(
  kernelRef: ExecutionV0KernelRefSchema | null | undefined
): string | null;
export declare function indexExecutionV0KernelProfiles(
  sessionDefaults?: Partial<ExecutionV0SessionDefaultsSchema> | null
): Map<string, ExecutionV0KernelProfileSchema>;
export declare function resolveExecutionV0KernelProfile(
  profileIndex: Map<string, ExecutionV0KernelProfileSchema>,
  step: Partial<ExecutionV0StepSchema>
): ExecutionV0KernelProfileSchema | null;
export declare function resolveExecutionV0Precision(
  step: Partial<ExecutionV0StepSchema>,
  profile: ExecutionV0KernelProfileSchema | null,
  sessionDefaults?: Partial<ExecutionV0SessionDefaultsSchema> | null
): {
  precision: {
    inputDtype: 'f16' | 'f32' | null;
    mathDtype: 'f16' | 'f32';
    accumDtype: 'f16' | 'f32';
    outputDtype: 'f16' | 'f32';
  };
  sources: {
    inputDtype: 'manifest' | 'kernelProfile' | 'sessionDefault' | 'derived';
    mathDtype: 'manifest' | 'kernelProfile' | 'sessionDefault' | 'derived';
    accumDtype: 'manifest' | 'kernelProfile' | 'sessionDefault' | 'derived';
    outputDtype: 'manifest' | 'kernelProfile' | 'sessionDefault' | 'derived';
  };
};
export declare function resolveExecutionV0KVIO(
  step: Partial<ExecutionV0StepSchema>,
  profile: ExecutionV0KernelProfileSchema | null,
  sessionDefaults?: Partial<ExecutionV0SessionDefaultsSchema> | null
): {
  value: ExecutionV0KVIO;
  source: 'manifest' | 'kernelProfile' | 'sessionDefault';
};
export declare function buildExecutionV0ContractArtifact(
  manifestInference: Record<string, unknown> | null | undefined,
  options?: {
    modelId?: string;
  }
): ExecutionV0ContractArtifact | null;
