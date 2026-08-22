export const TARGET_PLAN_SCHEMA_ID = 'doppler.target-plan/v1';
export const TARGET_PLAN_SCHEMA_VERSION = 1;

export interface TargetPlanCapabilityPredicate {
  requiresF16: boolean;
  requiresSubgroups: boolean;
  minBufferSize: number;
  supportedVendors?: string[] | null;
}

export interface TargetPlanDtypes {
  activation: 'f32' | 'f16' | 'f16-subgroups';
  kv: 'f32' | 'f16' | 'q4k' | 'q8_0';
  weight: string;
}

export interface TargetPlanKernelModule {
  id: string;
  file: string;
  entry: string;
  digest: `sha256:${string}`;
}

export interface TargetPlanMemoryLayout {
  kvCacheLayout: 'paged' | 'contiguous' | 'sliding-window' | 'tiered';
  estimatedPeakBytes: number;
  bufferSlots?: Array<{
    slotId: string;
    role: string;
    scope: 'static' | 'layer-recycled' | 'transient' | 'session';
  }>;
}

export interface TargetPlan {
  schema: 'doppler.target-plan/v1';
  schemaVersion: 1;
  targetId: string;
  modelId: string;
  capabilityPredicate: TargetPlanCapabilityPredicate;
  dtypes: TargetPlanDtypes;
  fusions?: string[];
  kernelClosure: TargetPlanKernelModule[];
  memoryLayout: TargetPlanMemoryLayout;
  phases: {
    prefill?: unknown[];
    decode?: unknown[];
    encode?: unknown[];
  };
}

export declare function validateTargetPlan(plan: unknown): { ok: boolean; errors: string[] };
export declare function hashTargetPlan(plan: unknown): `sha256:${string}`;
export declare function matchesDeviceCapability(
  targetPlan: TargetPlan,
  deviceProfile: {
    hasF16?: boolean;
    hasSubgroups?: boolean;
    maxBufferSize?: number;
    adapter?: { vendor?: string | null };
  }
): boolean;
export declare function createTargetPlan(params: Partial<TargetPlan>): TargetPlan;
