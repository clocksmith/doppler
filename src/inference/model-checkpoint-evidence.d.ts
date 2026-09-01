export declare const MODEL_CHECKPOINT_EVIDENCE_SCHEMA: 'doppler.model-checkpoint-evidence/v1';

export type ModelCheckpointStage =
  | 'embedding'
  | 'rmsnorm'
  | 'qkv'
  | 'rope'
  | 'attention'
  | 'kv'
  | 'mlp'
  | 'logits';

export interface ModelCheckpointRecord {
  recordIndex: number;
  opId: string | null;
  stageName: string | null;
  opType: string | null;
  layerIndex: number | null;
  dtype: string | null;
  shapeSignature: string | null;
  fullTensorDigest: `sha256:${string}`;
  sample: number[] | null;
  sampleCoordinates: unknown[] | null;
  data: number[] | null;
  stats: Record<string, unknown> | null;
  hasNaN: boolean;
  hasInf: boolean;
}

export interface ModelCheckpointDigest {
  stage: ModelCheckpointStage;
  phase: 'prefill' | 'decode';
  stepIndex: number;
  digest: `sha256:${string}`;
  recordCount: number;
  records: Array<ModelCheckpointRecord | Record<string, unknown>>;
  layout?: string | null;
  kvDtype?: string | null;
}

export interface ModelCheckpointStep {
  stepIndex: number;
  phase: 'prefill' | 'decode';
  checkpoints: Partial<Record<ModelCheckpointStage, ModelCheckpointDigest>>;
  missing: ModelCheckpointStage[];
  pass: boolean;
}

export interface ModelCheckpointEvidence {
  schema: typeof MODEL_CHECKPOINT_EVIDENCE_SCHEMA;
  status: 'complete' | 'blocked';
  timelineRecordCount: number;
  expectedStepCount: number | null;
  stepCount: number;
  decodeStepCount: number;
  requiredStages: ModelCheckpointStage[];
  capturedStages: ModelCheckpointStage[];
  missingStages: ModelCheckpointStage[];
  blockers: string[];
  steps: ModelCheckpointStep[];
  kv: ModelCheckpointDigest | null;
}

export declare const MODEL_CHECKPOINT_STAGES: readonly ModelCheckpointStage[];

export declare function buildModelCheckpointEvidence(options: {
  operatorDiagnostics: { timeline?: Array<Record<string, unknown>> } | null;
  kvCacheByteProof?: Record<string, unknown> | null;
  expectedStepCount?: number | null;
  minimumDecodeSteps?: number | null;
}): ModelCheckpointEvidence | null;

export declare function flattenModelCheckpointDigests(
  evidence: ModelCheckpointEvidence | null
): ModelCheckpointDigest[];
