export const INITIAL_EXECUTION_IDENTITY_SCHEMA_ID: 'doppler.initial-execution-identity/v1';

export interface InitialExecutionIdentity {
  schema: 'doppler.initial-execution-identity/v1';
  executionGraphHash: `sha256:${string}`;
  resolvedGraphHash: `sha256:${string}`;
  kernelClosure: Array<{ moduleId: string; file: string; entry: string; digest: `sha256:${string}` }>;
  kernelClosureHash: `sha256:${string}`;
  dtypeLane: Record<string, unknown>;
  fusionSet: unknown[];
  fusionSetHash: `sha256:${string}`;
  kvLayout: Record<string, unknown>;
  kvLayoutHash: `sha256:${string}`;
  memoryPolicy: Record<string, unknown>;
  memoryPolicyHash: `sha256:${string}`;
  executionPlanDigest: `sha256:${string}`;
  runtimeEngine: Record<string, unknown>;
  runtimeEngineDigest: `sha256:${string}`;
  digest: `sha256:${string}`;
}

export declare function validateInitialExecutionIdentity(identity: unknown): { ok: boolean; errors: string[] };
export declare function createInitialExecutionIdentity(fields: Omit<InitialExecutionIdentity, 'schema' | 'kernelClosureHash' | 'fusionSetHash' | 'kvLayoutHash' | 'memoryPolicyHash' | 'runtimeEngineDigest' | 'digest'>): InitialExecutionIdentity;
export declare function observeInitialExecutionIdentity(resolved: Record<string, unknown>): InitialExecutionIdentity;
export declare function assertInitialExecutionIdentity(expected: InitialExecutionIdentity, observed: InitialExecutionIdentity): true;
