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
