export const INITIAL_EXECUTION_IDENTITY_SCHEMA_ID: 'doppler.initial-execution-identity/v1';
export const INITIAL_EXECUTION_IDENTITY_V2_SCHEMA_ID: 'doppler.initial-execution-identity/v2';
export const PROGRAM_LOAD_POLICY_SCHEMA_ID: 'doppler.pack-program-load-policy/v1';

export interface InitialExecutionIdentityV1 {
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

export interface InitialExecutionIdentityV2 extends Omit<InitialExecutionIdentityV1, 'schema' | 'digest'> {
  schema: 'doppler.initial-execution-identity/v2';
  programLoadPolicy: {
    schema: 'doppler.pack-program-load-policy/v1';
    runtimeConfig: {
      inference: {
        session: Record<string, unknown>;
        compute: Record<string, unknown>;
      };
    };
  };
  programLoadPolicyHash: `sha256:${string}`;
  digest: `sha256:${string}`;
}

export type InitialExecutionIdentity = InitialExecutionIdentityV1 | InitialExecutionIdentityV2;

export declare function validateInitialExecutionIdentity(identity: unknown): { ok: boolean; errors: string[] };
export declare function createInitialExecutionIdentity(fields: Omit<InitialExecutionIdentityV1, 'schema' | 'kernelClosureHash' | 'fusionSetHash' | 'kvLayoutHash' | 'memoryPolicyHash' | 'runtimeEngineDigest' | 'digest'>): InitialExecutionIdentityV1;
export declare function createInitialExecutionIdentityV2(fields: Omit<InitialExecutionIdentityV2, 'schema' | 'kernelClosureHash' | 'fusionSetHash' | 'kvLayoutHash' | 'memoryPolicyHash' | 'runtimeEngineDigest' | 'programLoadPolicyHash' | 'digest'>): InitialExecutionIdentityV2;
export declare function observeInitialExecutionIdentity(resolved: Record<string, unknown>): InitialExecutionIdentityV2;
export declare function resolveProgramLoadRuntimeConfig(identity: InitialExecutionIdentity): Record<string, unknown> | null;
export declare function assertInitialExecutionIdentity(expected: InitialExecutionIdentity, observed: InitialExecutionIdentity): true;
