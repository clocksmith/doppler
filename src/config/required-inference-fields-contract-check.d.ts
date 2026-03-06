export interface RequiredInferenceFieldsContractArtifact {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: Array<{ id: string; ok: boolean }>;
  errors: string[];
  stats: {
    fieldCases: number;
    nullableCases: number;
    nonNullableCases: number;
  };
}

export declare function buildRequiredInferenceFieldsContractArtifact(): RequiredInferenceFieldsContractArtifact;

export interface ManifestRequiredInferenceFieldsArtifact extends RequiredInferenceFieldsContractArtifact {
  scope: 'manifest';
  label: string;
}

export declare function buildManifestRequiredInferenceFieldsArtifact(
  inference: Record<string, unknown> | null | undefined,
  label?: string
): ManifestRequiredInferenceFieldsArtifact;
