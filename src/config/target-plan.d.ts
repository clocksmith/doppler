import type { InitialExecutionIdentity } from './initial-execution-identity.js';

export const TARGET_PLAN_SCHEMA_ID: 'doppler.target-plan/v1';
export const TARGET_PLAN_SCHEMA_VERSION: 1;
export const TARGET_PLAN_V2_SCHEMA_ID: 'doppler.target-plan/v2';
export const TARGET_PLAN_V2_SCHEMA_VERSION: 2;

export interface TargetPlanMemoryExpression {
  op: 'constant' | 'affine';
  bytes?: number;
  constantBytes?: number;
  terms?: Record<string, number>;
  alignment?: number;
  minimumBytes?: number;
}

export interface TargetPlanMemoryLayout {
  kvCacheLayout: string;
  bufferSlots: Array<{
    slotId: string;
    role: string;
    scope: 'static' | 'layer-recycled' | 'transient' | 'session';
    owner: 'runtime' | 'program';
    usage?: string[];
    usageBits?: number;
    size: TargetPlanMemoryExpression;
  }>;
}

export interface TargetPlanV1 {
  schema: 'doppler.target-plan/v1';
  schemaVersion: 1;
  targetId: string;
  modelId: string;
  modelIRHash: `sha256:${string}`;
  executionGraphHash: `sha256:${string}`;
  programBundleHash: `sha256:${string}`;
  capabilityPredicate: {
    requiresF16: boolean;
    requiresSubgroups: boolean;
    minBufferSize: number;
    supportedVendors?: string[];
  };
  dtypes: { activation: string; kv: string; weight: string };
  fusions: string[];
  kernelClosure: Array<{ moduleId: string; digest: `sha256:${string}`; sourceHash: `sha256:${string}` }>;
  memoryLayout: TargetPlanMemoryLayout;
  phases: Record<string, Array<Record<string, unknown>>>;
  qualification: Array<{
    surface: string;
    status: 'passed';
    evidenceArtifactId: string;
    evidenceHash: `sha256:${string}`;
    transcriptHash?: `sha256:${string}`;
    generatedTokens: number;
  }>;
}

export interface TargetPlanV2 extends Omit<TargetPlanV1, 'schema' | 'schemaVersion'> {
  schema: 'doppler.target-plan/v2';
  schemaVersion: 2;
  initialExecutionIdentity: InitialExecutionIdentity;
}

export type TargetPlan = TargetPlanV1 | TargetPlanV2;

export interface TargetPlanDeviceProfile {
  surface: string;
  hasF16?: boolean;
  hasSubgroups?: boolean;
  maxBufferSize?: number;
  adapter?: {
    vendor?: string | null;
    architecture?: string | null;
    device?: string | null;
    description?: string | null;
  };
}

export declare function validateTargetPlan(plan: unknown): { ok: boolean; errors: string[] };
export declare function hashTargetPlan(plan: unknown): `sha256:${string}`;
export declare function matchesDeviceCapability(targetPlan: TargetPlan, deviceProfile: Record<string, unknown>): boolean;
export declare function selectQualifiedTargetPlan(
  targetPlans: TargetPlan[],
  deviceProfile: TargetPlanDeviceProfile
): TargetPlan;
export declare function createTargetPlan(params: Omit<TargetPlanV1, 'schema' | 'schemaVersion'>): TargetPlanV1;
export declare function createTargetPlanV2(params: Omit<TargetPlanV2, 'schema' | 'schemaVersion'>): TargetPlanV2;
