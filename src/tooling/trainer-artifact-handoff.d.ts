import type { RuntimeConfigSchema } from '../config/schema/index.js';
import type {
  TrainerArtifactBridgeDescriptor,
  TrainerArtifactImportPlan,
  TrainerArtifactVerificationReceipt,
} from '../experimental/bridge/trainer-artifact-bridge.js';
import type { NodeSourceRuntimeBundle } from './node-source-runtime.js';

export declare const TRAINER_ARTIFACT_HANDOFF_VERIFICATION_SCHEMA_ID:
  'doppler.trainer-artifact-handoff-verification/v1';
export declare const TRAINER_ARTIFACT_IMPORT_RECEIPT_SCHEMA_ID:
  'doppler.trainer-artifact-import-receipt/v1';

export interface VerifyTrainerArtifactHandoffOptions {
  contract?: Record<string, unknown> | TrainerArtifactBridgeDescriptor;
  contractPath?: string;
  repositoryRoots: Record<string, string>;
  verifiedAt?: string;
}

export interface VerifyTrainerArtifactHandoffResult {
  descriptor: TrainerArtifactBridgeDescriptor;
  receipt: TrainerArtifactVerificationReceipt & {
    artifactKind: string;
    artifactRole: string;
    verifiedAt: string;
    selection: TrainerArtifactBridgeDescriptor['selection'];
    admission: Record<string, boolean>;
    checks: Array<Record<string, unknown>>;
    files: Array<Record<string, unknown>>;
    architecture: Record<string, unknown>;
  };
  repositoryRoots: Record<string, string>;
}

export interface ImportTrainerArtifactHandoffOptions extends VerifyTrainerArtifactHandoffOptions {
  runtimeConfig?: RuntimeConfigSchema | null;
  runtimeResolver?: (options: Record<string, unknown>) => Promise<NodeSourceRuntimeBundle | null>;
  adapterLoader?: (manifest: Record<string, unknown>, options: Record<string, unknown>) => Promise<unknown>;
}

export declare function loadTrainerArtifactHandoffContract(
  contractPath: string
): Promise<Record<string, unknown>>;

export declare function resolveTrainerArtifactHandoffDescriptor(
  contract: Record<string, unknown> | TrainerArtifactBridgeDescriptor
): TrainerArtifactBridgeDescriptor;

export declare function verifyTrainerArtifactHandoff(
  options: VerifyTrainerArtifactHandoffOptions
): Promise<VerifyTrainerArtifactHandoffResult>;

export declare function importTrainerArtifactHandoff(
  options: ImportTrainerArtifactHandoffOptions
): Promise<{
  descriptor: TrainerArtifactBridgeDescriptor;
  verification: VerifyTrainerArtifactHandoffResult['receipt'];
  plan: TrainerArtifactImportPlan;
  imported: Record<string, unknown>;
  receipt: Record<string, unknown> & { receiptHash: string };
}>;
