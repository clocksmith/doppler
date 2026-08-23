export const MANIFEST_CONVERSION_POSTFLIGHT_SCHEMA_ID: 'doppler.manifest-conversion-postflight/v1';

export interface ManifestConversionFileObservation {
  role?: string;
  index?: number;
  path?: string;
  filename?: string;
  size: number;
  digest: `sha256:${string}`;
}

export interface ManifestConversionPostflightReceipt {
  schema: 'doppler.manifest-conversion-postflight-receipt/v1';
  modelId: string;
  entryPointId: string;
  author: { kind: 'human' | 'ai' | 'tool'; actor: string };
  preflightEvidence: {
    receiptDigest: `sha256:${string}`;
    conversionConfigDigest: `sha256:${string}`;
    expectedTensorCount: number;
  };
  conversionEvidence: {
    reportDigest: `sha256:${string}`;
    startedAtUtc: string;
    completedAtUtc: string;
    durationMs: number;
    modelType: string;
    tensorCount: number;
    shardCount: number;
    totalBytes: number;
  };
  manifestEvidence: {
    digest: `sha256:${string}`;
    inferenceDigest: `sha256:${string}`;
    executionDigest: `sha256:${string}`;
    artifactIdentity: Record<string, unknown>;
  };
  physicalClosure: {
    digest: `sha256:${string}`;
    shardCount: number;
    shardBytes: number;
    artifactCount: number;
    artifacts: Array<{
      role: string;
      path: string;
      size: number;
      digest: `sha256:${string}`;
    }>;
  };
  dispositions: {
    conversionExecuted: true;
    physicalShardClosureVerified: true;
    qualificationStarted: false;
    packEligible: false;
  };
  receiptDigest: `sha256:${string}`;
}

export declare function createManifestConversionPostflightReceipt(options: {
  conversionConfig: Record<string, unknown>;
  conversionReport: Record<string, unknown>;
  conversionReportDigest: `sha256:${string}`;
  manifest: Record<string, unknown>;
  manifestDigest: `sha256:${string}`;
  preflightReceipt: Record<string, unknown>;
  shardObservations: ManifestConversionFileObservation[];
  artifactObservations: ManifestConversionFileObservation[];
  policy: Record<string, unknown>;
}): Readonly<ManifestConversionPostflightReceipt>;
