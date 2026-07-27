import type {
  RegisteredCalibrationDigest,
  RegisteredVariantCalibrationPlan,
  RegisteredVariantCalibrationReceipt,
} from './registered-variant-calibration.js';
import type { RuntimeOptimizationContract } from './runtime-optimization.js';

export interface RegisteredVariantCorrectnessEvidence {
  binding: {
    artifactDigest: RegisteredCalibrationDigest;
    executionGraphDigest: RegisteredCalibrationDigest;
    descriptorDigest: RegisteredCalibrationDigest;
    kernelDigest: RegisteredCalibrationDigest;
    executionEngineDigest: RegisteredCalibrationDigest;
    browserDigest: RegisteredCalibrationDigest;
    adapterDigest: RegisteredCalibrationDigest;
  };
  operatorReference: Record<string, {
    passed: boolean;
    kernelDigest: RegisteredCalibrationDigest;
    [key: string]: unknown;
  }>;
  boundaryPack: {
    schema: 'doppler.boundary-comparison-receipt/v1';
    promotionGate: {
      boundaryCompatible: boolean;
      sourcePrecisionControlPassed: boolean;
      [key: string]: unknown;
    };
    [key: string]: unknown;
  };
  tokenParity: {
    exact: boolean;
    tokenCount: number;
    [key: string]: unknown;
  };
}

export interface RegisteredOptimizationCandidateRegistryEntry {
  registryId: string;
  kind: 'registered-kernel-variant' | 'registered-execution-graph-patch';
  digest: RegisteredCalibrationDigest;
  runtimeInputs: {
    runtimeProfile: string | null;
    runtimeConfig: Record<string, unknown>;
  };
  evidenceScope: {
    artifactDigest: RegisteredCalibrationDigest;
    executionGraphDigest: RegisteredCalibrationDigest;
    descriptorDigest: RegisteredCalibrationDigest;
    kernelDigest: RegisteredCalibrationDigest;
    executionEngineDigest: RegisteredCalibrationDigest;
    browserDigest: RegisteredCalibrationDigest;
    adapterDigest: RegisteredCalibrationDigest;
  };
  checkedInPath: string;
}

export interface RegisteredVariantCalibrationJob {
  schema: 'doppler.registered-variant-calibration-job/v1';
  surface: 'node' | 'browser';
  plan: RegisteredVariantCalibrationPlan;
  correctnessEvidence: Record<string, RegisteredVariantCorrectnessEvidence>;
  candidateRegistry: {
    schema: 'doppler.runtime-optimization-candidate-registry/v1';
    entries: Record<string, RegisteredOptimizationCandidateRegistryEntry>;
  };
  performance: Record<string, {
    registryId: string;
    contract: RuntimeOptimizationContract;
  }>;
  commandOptions?: Record<string, unknown>;
}

export interface RegisteredVariantCalibrationJobReceipt
  extends Omit<RegisteredVariantCalibrationReceipt, 'digest'> {
  jobDigest: RegisteredCalibrationDigest;
  executionSurface: 'node' | 'browser';
  executionEngine: string;
  digest: RegisteredCalibrationDigest;
}

export declare const REGISTERED_VARIANT_CALIBRATION_JOB_SCHEMA:
  'doppler.registered-variant-calibration-job/v1';

export declare function runRegisteredVariantCalibrationJob(
  job: RegisteredVariantCalibrationJob,
  options: {
    registry: Record<string, unknown>;
    kernelDigests: Readonly<Record<string, string>>;
    runCommand(
      request: Record<string, unknown>,
      options?: Record<string, unknown>
    ): Promise<Record<string, unknown>>;
    executionEngine: string;
    onEvent?(event: Record<string, unknown>): void;
  }
): Promise<RegisteredVariantCalibrationJobReceipt>;
