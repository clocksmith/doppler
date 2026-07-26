export declare const SOURCE_INTAKE_SCHEMA: 'doppler.source-intake/v1';
export declare const SOURCE_INTAKE_CONVERSION_SKELETON_SCHEMA:
  'doppler.source-intake-conversion-skeleton/v1';
export declare const SOURCE_INTAKE_CONTRACT_TEST_SCHEMA:
  'doppler.source-intake-contract-test/v1';

export interface SourceIntakePolicy {
  schema: 'doppler.source-intake-policy/v1';
  facts: Array<{
    factId: string;
    sourcePointers: string[];
    owner: string;
    verificationBoundary?: string;
  }>;
  requiredFacts: string[];
  architectureKeyPattern: string;
}

export interface SourceIntakeFact {
  factId: string;
  source: { file: string | null; jsonPointer: string | null; value: unknown };
  owner: string;
  proposal: unknown;
  confidence: 'direct' | 'derived' | 'family-inferred' | 'ambiguous' | 'unsupported';
  status: 'accepted' | 'unresolved';
  verification: { kind: 'boundary-pack' | 'contract-test'; boundary: string | null };
  note?: string;
}

export interface SourceIntakeReport {
  schema: 'doppler.source-intake/v1';
  ok: boolean;
  source: Record<string, unknown>;
  facts: SourceIntakeFact[];
  summary: Record<string, unknown>;
  digest: string;
}

export declare function inspectSourceModel(options: {
  sourceDir: string;
  policy: SourceIntakePolicy;
  familyIntake?: SourceIntakeReport | null;
}): Promise<{
  report: SourceIntakeReport;
  artifacts: {
    conversion: Record<string, unknown>;
    contractTests: Record<string, unknown>;
  };
}>;
