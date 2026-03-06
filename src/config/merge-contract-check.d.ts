export interface MergeContractCheckResult {
  id: string;
  ok: boolean;
  detail: string;
  mode: 'actual' | 'modeled';
}

export interface MergeContractArtifact {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: MergeContractCheckResult[];
  errors: string[];
}

export declare function buildMergeContractArtifact(): MergeContractArtifact;
