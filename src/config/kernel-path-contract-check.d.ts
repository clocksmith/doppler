export interface KernelPathContractEntryFacts {
  id: string;
  aliasOf: string | null;
  hasFile: boolean;
}

export interface KernelPathFallbackMappingFacts {
  primaryKernelPathId: string;
  fallbackKernelPathId: string;
  primaryActivationDtype: 'f16' | 'f32';
  fallbackActivationDtype: 'f16' | 'f32' | null;
}

export interface KernelPathContractFacts {
  registryId: string;
  entries: KernelPathContractEntryFacts[];
  fallbackMappings: KernelPathFallbackMappingFacts[];
  fallbackRules?: Array<{
    matchKernelPathId: string | null;
    value: string | null;
    isDefault: boolean;
  }>;
  autoSelectRules?: Array<{
    matchKernelPathRef: string | null;
    allowCapabilityAutoSelection: boolean | null;
    hasSubgroups: boolean | null;
    valueKind: 'string' | 'context';
    value: string;
    isDefault: boolean;
  }>;
}

export interface KernelPathContractCheckResult {
  id: string;
  ok: boolean;
}

export interface KernelPathContractValidationResult {
  ok: boolean;
  errors: string[];
  checks: KernelPathContractCheckResult[];
}

export interface KernelPathContractArtifact {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: KernelPathContractCheckResult[];
  errors: string[];
  stats: {
    totalEntries: number;
    aliasEntries: number;
    canonicalEntries: number;
    fallbackMappings: number;
    fallbackRules: number;
    autoSelectRules: number;
  };
}

export declare function extractKernelPathContractFacts(
  input: Record<string, unknown> | unknown[],
  options?: {
    registryId?: string;
  }
): KernelPathContractFacts;

export declare function validateKernelPathContractFacts(
  facts: KernelPathContractFacts
): KernelPathContractValidationResult;

export declare function buildKernelPathContractArtifact(
  input: Record<string, unknown> | unknown[],
  options?: {
    registryId?: string;
  }
): KernelPathContractArtifact;
