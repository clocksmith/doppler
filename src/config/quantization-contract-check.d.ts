export interface QuantizationContractArtifact {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: Array<{ id: string; ok: boolean }>;
  errors: string[];
  stats: {
    sampledSizes: number;
  };
}

export declare function buildQuantizationContractArtifact(): QuantizationContractArtifact;
