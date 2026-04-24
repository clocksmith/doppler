export interface DistributedPlanCompatibility {
  artifactIdentityHash?: string | null;
  manifestHash?: string | null;
  executionGraphDigest?: string | null;
  integrityExtensionsHash?: string | null;
}

export interface DistributedPlanValidationOptions {
  expectedCompatibility?: DistributedPlanCompatibility | null;
  expectedPlanId?: string | null;
  expectedTopologyHash?: string | null;
}

export interface DistributedPlanValidationResult {
  valid: boolean;
  errors: string[];
  code: string | null;
}

export declare function validateDistributedPlan(
  plan: Record<string, unknown>,
  options?: DistributedPlanValidationOptions
): DistributedPlanValidationResult;
