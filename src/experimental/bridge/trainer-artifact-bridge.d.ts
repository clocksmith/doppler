export declare const TRAINER_ARTIFACT_BRIDGE_SCHEMA_ID: 'doppler.trainer-artifact-bridge/v1';
export declare const TRAINER_ARTIFACT_IMPORT_PLAN_SCHEMA_ID: 'doppler.trainer-artifact-import-plan/v1';
export declare const TRAINER_ARTIFACT_PARITY_EVIDENCE_SCHEMA_ID: 'doppler.trainer-artifact-parity-evidence/v1';
export declare const TRAINER_ARTIFACT_PARITY_RECEIPT_SCHEMA_ID: 'doppler.trainer-artifact-parity-receipt/v1';
export declare const TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT: 'full_checkpoint';
export declare const TRAINER_ARTIFACT_KIND_PEFT_ADAPTER: 'peft_adapter';

export declare const TRANSLATION_FULL_CHECKPOINT_PARITY_CHECKS: readonly string[];
export declare const COLUMBO_QWEN_ADAPTER_PARITY_CHECKS: readonly string[];

export interface TrainerArtifactFileIdentity {
  id: string;
  role: string;
  repository: string;
  rootPath: string;
  path: string;
  sha256: string;
  bytes: number;
}

export interface TrainerArtifactArchitecture {
  architectures: string[];
  modelType: string;
  hiddenSize: number;
  intermediateSize: number;
  layers: number;
  attentionHeads: number;
  keyValueHeads: number;
  headDim: number;
  vocabularySize: number;
}

export interface TrainerArtifactBridgeDescriptor {
  schema: typeof TRAINER_ARTIFACT_BRIDGE_SCHEMA_ID;
  bridgeId: string;
  sourceContractId: string;
  artifact: {
    kind: typeof TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT | typeof TRAINER_ARTIFACT_KIND_PEFT_ADAPTER;
    role: 'diagnostic_baseline' | 'diagnostic_candidate' | 'selected_candidate';
    format: string;
    repository: string;
    rootPath: string;
    files: TrainerArtifactFileIdentity[];
  };
  baseModel: {
    modelId: string;
    checkpointSha256: string;
    tokenizer: {
      files: TrainerArtifactFileIdentity[];
      promptContract: TrainerArtifactFileIdentity;
    };
    architecture: TrainerArtifactArchitecture;
  };
  conversion: {
    owner: string;
    sourceArtifactSha256: string;
    config: TrainerArtifactFileIdentity | null;
    runtimeArtifact: Record<string, unknown> | null;
  };
  evaluation: {
    populationRole: string;
    files: TrainerArtifactFileIdentity[];
  };
  selection: {
    authority: string;
    status: 'not_selected' | 'selected';
    receipt: string | null;
  };
  parity: {
    profile: 'translation_full_checkpoint' | 'columbo_qwen_adapter';
    requiredChecks: string[];
  };
  claimBoundary: string;
}

export interface TrainerArtifactVerificationReceipt {
  schema: string;
  ok: boolean;
  bridgeId: string;
  artifactIdentitySha256: string;
  receiptHash: string;
  [key: string]: unknown;
}

export interface TrainerArtifactImportPlan {
  schema: typeof TRAINER_ARTIFACT_IMPORT_PLAN_SCHEMA_ID;
  bridgeId: string;
  artifactKind: string;
  entrypoint: 'resolveNodeSourceRuntimeBundle' | 'loadLoRAWeights';
  source: Record<string, unknown>;
  baseModel: TrainerArtifactBridgeDescriptor['baseModel'];
  conversion: TrainerArtifactBridgeDescriptor['conversion'];
  verificationReceiptHash: string | null;
  admission: {
    identityVerificationRequired: true;
    parityExecutionAllowed: true;
    candidateCompetitionAllowed: boolean;
    promotionAllowed: false;
    selectionAuthority: string;
    selectionReceipt: string | null;
  };
  planHash: string;
}

export interface TrainerArtifactParityEvidenceCheck {
  id: string;
  status: 'pending' | 'pass' | 'fail';
  evidenceHash: string | null;
  artifactIdentitySha256: string;
  upstreamDecision: 'pass' | 'block' | null;
}

export interface TrainerArtifactParityEvidence {
  schema: typeof TRAINER_ARTIFACT_PARITY_EVIDENCE_SCHEMA_ID;
  bridgeId: string;
  profile: string;
  artifactIdentitySha256: string;
  checks: TrainerArtifactParityEvidenceCheck[];
}

export declare function normalizeGammaTrainerArtifactHandoff(
  contract: Record<string, unknown>
): TrainerArtifactBridgeDescriptor;

export declare function validateTrainerArtifactBridgeDescriptor(
  descriptor: unknown
): {
  valid: boolean;
  descriptor: TrainerArtifactBridgeDescriptor | null;
  errors: string[];
};

export declare function assertTrainerArtifactCandidateEntry(
  descriptor: TrainerArtifactBridgeDescriptor
): TrainerArtifactBridgeDescriptor;

export declare function buildTrainerArtifactImportPlan(
  descriptor: TrainerArtifactBridgeDescriptor,
  verificationReceipt?: TrainerArtifactVerificationReceipt | null
): TrainerArtifactImportPlan;

export declare function buildTrainerArtifactParityTemplate(
  descriptor: TrainerArtifactBridgeDescriptor,
  verificationReceipt: TrainerArtifactVerificationReceipt
): TrainerArtifactParityEvidence;

export declare function verifyTrainerArtifactParityEvidence(
  descriptor: TrainerArtifactBridgeDescriptor,
  verificationReceipt: TrainerArtifactVerificationReceipt,
  evidence: TrainerArtifactParityEvidence
): {
  schema: typeof TRAINER_ARTIFACT_PARITY_RECEIPT_SCHEMA_ID;
  bridgeId: string;
  profile: string;
  artifactIdentitySha256: string;
  identityReceiptHash: string;
  decision: 'pass' | 'block';
  blockers: string[];
  checkEvidenceHashes: Record<string, string | null>;
  receiptHash: string;
};
