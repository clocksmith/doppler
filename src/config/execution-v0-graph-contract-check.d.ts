export interface ExecutionV0GraphContractCheckResult {
  id: string;
  ok: boolean;
}

export interface ExecutionV0GraphContractArtifact {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: ExecutionV0GraphContractCheckResult[];
  errors: string[];
  stats: {
    prefillSteps: number;
    decodeSteps: number;
  };
}

export declare function buildExecutionV0GraphContractArtifact(
  options?: Record<string, unknown>
): ExecutionV0GraphContractArtifact | null;
