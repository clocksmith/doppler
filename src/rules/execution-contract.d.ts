
export interface ExecutionContractStepFacts {
  id: string;
  phase: 'prefill' | 'decode' | 'both';
  opClass: 'attention' | 'embed' | 'norm' | 'projection' | 'residual' | 'sample' | 'other';
}

export interface ExecutionContractSessionFacts {
  layout: 'contiguous' | 'paged' | 'tiered' | 'bdpa';
  disableCommandBatching: boolean;
  decodeBatchSize: number;
  headDim: number;
  kvLen: number;
  coldQuantMode: 'none' | 'int8' | 'int4' | 'turboquant' | 'turboquant_prod';
  contiguousQuantMode: 'none' | 'turboquant' | 'turboquant_prod';
}

export interface ExecutionContractFacts {
  modelId: string;
  session: ExecutionContractSessionFacts;
  steps: ExecutionContractStepFacts[];
}

export interface ExecutionContractCheckResult {
  id: string;
  ok: boolean;
}

export interface ExecutionContractValidationResult {
  ok: boolean;
  errors: string[];
  checks: ExecutionContractCheckResult[];
}

export interface ManifestExecutionContractValidationResult extends ExecutionContractValidationResult {
  facts: ExecutionContractFacts;
}

export interface ExecutionContractArtifact {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: ExecutionContractCheckResult[];
  errors: string[];
  session: ExecutionContractSessionFacts | null;
  steps: {
    total: number;
    attention: number;
    attentionPhases: {
      prefill: number;
      decode: number;
      both: number;
    };
  } | null;
}

export declare function sanitizeLeanModuleName(value: unknown): string;

export declare function extractExecutionContractFacts(
  manifest: Record<string, unknown>
): ExecutionContractFacts;

export declare function validateExecutionContractFacts(
  facts: ExecutionContractFacts
): ExecutionContractValidationResult;

export declare function validateManifestExecutionContract(
  manifest: Record<string, unknown>
): ManifestExecutionContractValidationResult;

export declare function buildExecutionContractArtifact(
  manifest: Record<string, unknown>
): ExecutionContractArtifact | null;
