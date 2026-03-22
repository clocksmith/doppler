/**
 * Operator execution record emission for operator-level differential debugging.
 */

import type { StageName, OperatorClass } from './stage-names.js';

export type CaptureLevel = 'none' | 'metadata' | 'slice' | 'full';
export type InferencePhase = 'prefill' | 'decode' | 'both';

export interface OperatorExecutionRecord {
  // Model identity
  modelHash: string | null;
  adapterHash: string | null;
  runtimeConfigHash: string | null;
  executionPlanHash: string | null;

  // Semantic context (Doppler-owned)
  phase: InferencePhase | null;
  tokenIndex: number | null;
  layerIndex: number | null;
  opId: string;
  opType: OperatorClass | null;
  stageName: StageName;
  shapeSignature: string | null;
  dtype: string | null;
  quantizationMode: string | null;
  inputTensorIds: string[];
  outputTensorIds: string[];

  // Capture and drift policy
  capturePolicy: CaptureLevel;
  driftPolicyId: string | null;

  // Execution truth (Doe-owned)
  kernelDigest: string | null;
  wgslHash: string | null;
  pipelineHash: string | null;
  backend: string | null;
  adapterVendor: string | null;
  adapterArchitecture: string | null;
  driverVersion: string | null;
  workgroupGeometry: number[] | null;
  dispatchCount: number | null;
  timing: { wallMs: number; gpuMs: number | null } | null;
  captureArtifactIds: string[];
}

export interface ExecutionFacts {
  kernelDigest?: string | null;
  wgslHash?: string | null;
  pipelineHash?: string | null;
  backend?: string | null;
  adapterVendor?: string | null;
  adapterArchitecture?: string | null;
  driverVersion?: string | null;
  workgroupGeometry?: number[] | null;
  dispatchCount?: number | null;
  gpuMs?: number | null;
  captureArtifactIds?: string[];
}

export interface DivergenceResult {
  type: 'sequence_mismatch' | 'drift_check_needed' | 'length_mismatch';
  index: number;
  opId?: string;
  opType?: OperatorClass | null;
  tolerance?: number;
  baseline?: OperatorExecutionRecord;
  observed?: OperatorExecutionRecord;
  baselineLength?: number;
  observedLength?: number;
  message?: string;
}

export declare function createOperatorExecutionRecord(
  options: Partial<OperatorExecutionRecord> & { stageName: StageName }
): OperatorExecutionRecord;

export declare class OperatorEventEmitter {
  constructor(options?: {
    modelHash?: string | null;
    adapterHash?: string | null;
    runtimeConfigHash?: string | null;
    executionPlanHash?: string | null;
    enabled?: boolean;
  });

  get enabled(): boolean;
  enable(): void;
  disable(): void;

  beginOp(
    stageName: StageName,
    options?: Partial<OperatorExecutionRecord>
  ): string | null;

  endOp(executionFacts?: ExecutionFacts | null): OperatorExecutionRecord | null;

  emitRecord(
    stageName: StageName,
    options?: Partial<OperatorExecutionRecord>
  ): OperatorExecutionRecord | null;

  getTimeline(): OperatorExecutionRecord[];
  getRecordsByLayer(layerIdx: number): OperatorExecutionRecord[];
  getRecordsByPhase(phase: InferencePhase): OperatorExecutionRecord[];
  getRecordsByOpType(opType: OperatorClass): OperatorExecutionRecord[];
  getRecordByOpId(opId: string): OperatorExecutionRecord | null;
  get length(): number;
  clear(): void;
  toJSON(): {
    modelHash: string | null;
    adapterHash: string | null;
    runtimeConfigHash: string | null;
    executionPlanHash: string | null;
    recordCount: number;
    records: OperatorExecutionRecord[];
  };
}

export declare function findFirstDivergence(
  baselineTimeline: OperatorExecutionRecord[],
  observedTimeline: OperatorExecutionRecord[],
  getDriftTolerance?: (opType: OperatorClass | null, dtype: string | null) => number | null
): DivergenceResult | null;
