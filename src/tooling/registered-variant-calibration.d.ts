import type { RuntimeOptimizationReceipt } from './runtime-optimization.js';

export type RegisteredCalibrationDigest = `sha256:${string}`;
export type RegisteredCalibrationPhase = 'prefill' | 'decode';
export type RegisteredCalibrationTailClass = 'full-block' | 'tail';

export interface RegisteredVariantReference {
  operation: string;
  variantId: string;
  descriptorDigest: RegisteredCalibrationDigest;
  kernelDigest: RegisteredCalibrationDigest;
}

export interface RegisteredVariantCalibrationIdentity {
  artifactDigest: RegisteredCalibrationDigest;
  manifestDigest: RegisteredCalibrationDigest;
  executionGraphDigest: RegisteredCalibrationDigest;
  executionEngineDigest: RegisteredCalibrationDigest;
  browserDigest: RegisteredCalibrationDigest;
  adapterDigest: RegisteredCalibrationDigest;
  wrapperDigest: RegisteredCalibrationDigest;
  capabilities: string[];
}

export interface RegisteredVariantCalibrationShape {
  shapeId: string;
  phase: RegisteredCalibrationPhase;
  sequenceLength: number;
  batch: number;
  heads: {
    query: number;
    kv: number;
    dim: number;
  };
  tailClass: RegisteredCalibrationTailClass;
  layouts: {
    input: string;
    weight: string;
    output: string;
    kv: string;
  };
  dtypes: {
    storage: string;
    materialization: string;
    accumulation: string;
  };
  fusionRole: string;
  quantizationFormat: string;
}

export interface RegisteredVariantCalibrationPlan {
  schema: 'doppler.registered-variant-calibration-plan/v1';
  identity: RegisteredVariantCalibrationIdentity;
  baseline: RegisteredVariantReference;
  candidates: RegisteredVariantReference[];
  shapeSuite: RegisteredVariantCalibrationShape[];
  outputKind?: 'registered-kernel-variant' | 'registered-execution-graph-patch';
}

export interface ResolvedRegisteredVariant {
  reference: RegisteredVariantReference;
  descriptor: Record<string, unknown>;
  descriptorDigest: RegisteredCalibrationDigest;
  compatible: boolean;
  missingCapabilities: string[];
}

export interface ResolvedRegisteredVariantCalibrationPlan
  extends Omit<RegisteredVariantCalibrationPlan, 'baseline'> {
  baseline: ResolvedRegisteredVariant;
  resolvedCandidates: ResolvedRegisteredVariant[];
}

export interface RegisteredCalibrationCorrectnessInput {
  mode: 'operator-reference' | 'boundary-pack' | 'token-parity';
  identity: RegisteredVariantCalibrationIdentity;
  baseline: ResolvedRegisteredVariant;
  candidate: ResolvedRegisteredVariant;
  shape?: RegisteredVariantCalibrationShape;
  shapeSuite?: RegisteredVariantCalibrationShape[];
  minimumTokenCount?: number;
}

export interface RegisteredCalibrationCorrectnessResult {
  passed?: boolean;
  kernelDigest?: RegisteredCalibrationDigest;
  schema?: string;
  promotionGate?: {
    boundaryCompatible?: boolean;
    sourcePrecisionControlPassed?: boolean;
  };
  exact?: boolean;
  tokenCount?: number;
  [key: string]: unknown;
}

export interface RegisteredCalibrationTypedCandidate {
  kind: 'registered-kernel-variant';
  reference: RegisteredVariantReference;
  scope: {
    artifactDigest: RegisteredCalibrationDigest;
    executionGraphDigest: RegisteredCalibrationDigest;
    executionEngineDigest: RegisteredCalibrationDigest;
    browserDigest: RegisteredCalibrationDigest;
    adapterDigest: RegisteredCalibrationDigest;
    shapeSignatures: RegisteredVariantCalibrationShape[];
  };
}

export interface RegisteredCalibrationPerformanceInput {
  typedCandidate: RegisteredCalibrationTypedCandidate;
  baseline: ResolvedRegisteredVariant;
  candidate: ResolvedRegisteredVariant;
  shapeSuite: RegisteredVariantCalibrationShape[];
  identity: RegisteredVariantCalibrationIdentity;
}

export interface RegisteredVariantCalibrationProposal {
  kind: 'registered-kernel-variant' | 'registered-execution-graph-patch';
  activation: 'manual-promotion-required';
  candidate: RegisteredCalibrationTypedCandidate;
  selectionPolicy: {
    precisionPreference: 'f16' | 'best-proven';
    afterPromotion:
      | 'required-on-compatible-hardware'
      | 'selected-for-matching-evidence-scope';
    requiredCapabilities: string[];
    fallback: string;
  };
}

export interface RegisteredVariantCalibrationResult {
  candidate: RegisteredVariantReference;
  decision: 'incompatible' | 'rejected' | 'proposed';
  missingCapabilities?: string[];
  correctness?: Record<string, unknown>;
  performance?: RuntimeOptimizationReceipt;
  proposal?: RegisteredVariantCalibrationProposal | null;
  reason?: string | null;
}

export interface RegisteredVariantCalibrationReceipt {
  schema: 'doppler.registered-variant-calibration-receipt/v1';
  planDigest: RegisteredCalibrationDigest;
  identity: RegisteredVariantCalibrationIdentity;
  results: RegisteredVariantCalibrationResult[];
  proposedSelections: RegisteredVariantCalibrationProposal[];
  precisionSelectionPolicy: {
    policy: 'prefer-proven-f16';
    rule: string;
    gates: string[];
  };
  runtimeMutationApplied: false;
  digest: RegisteredCalibrationDigest;
}

export declare const REGISTERED_VARIANT_CALIBRATION_PLAN_SCHEMA:
  'doppler.registered-variant-calibration-plan/v1';
export declare const REGISTERED_VARIANT_CALIBRATION_RECEIPT_SCHEMA:
  'doppler.registered-variant-calibration-receipt/v1';

export declare function digestRegisteredVariantDescriptor(
  operation: string,
  variantId: string,
  descriptor: Record<string, unknown>
): RegisteredCalibrationDigest;

export declare function validateRegisteredVariantCalibrationPlan(
  plan: RegisteredVariantCalibrationPlan,
  registry: Record<string, unknown>
): ResolvedRegisteredVariantCalibrationPlan;

export declare function calibrateRegisteredVariants(
  plan: RegisteredVariantCalibrationPlan,
  options: {
    registry: Record<string, unknown>;
    runCorrectness(
      input: RegisteredCalibrationCorrectnessInput
    ): Promise<RegisteredCalibrationCorrectnessResult>;
    evaluatePerformance(
      input: RegisteredCalibrationPerformanceInput
    ): Promise<RuntimeOptimizationReceipt>;
  }
): Promise<RegisteredVariantCalibrationReceipt>;
