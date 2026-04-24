export interface DistributedPlanValidationResult {
  valid: boolean;
  errors: string[];
  code: string | null;
}

export declare function parseDistributedPlan(
  jsonString: string,
  options?: {
    expectedCompatibility?: {
      artifactIdentityHash?: string | null;
      manifestHash?: string | null;
      executionGraphDigest?: string | null;
      integrityExtensionsHash?: string | null;
    };
    expectedPlanId?: string | null;
    expectedTopologyHash?: string | null;
  }
): Record<string, unknown>;

export declare function validateDistributedPlan(
  plan: Record<string, unknown>,
  options?: {
    expectedCompatibility?: {
      artifactIdentityHash?: string | null;
      manifestHash?: string | null;
      executionGraphDigest?: string | null;
      integrityExtensionsHash?: string | null;
    };
    expectedPlanId?: string | null;
    expectedTopologyHash?: string | null;
  }
): DistributedPlanValidationResult;

