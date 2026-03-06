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
  coldQuantMode: 'none' | 'int8' | 'int4';
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
