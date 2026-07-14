import { isPlainObject } from '../../utils/plain-object.js';
import { sha256Hex } from '../../utils/sha256.js';
import { stableSortObject } from '../../utils/stable-sort-object.js';

export const TRAINER_ARTIFACT_BRIDGE_SCHEMA_ID = 'doppler.trainer-artifact-bridge/v1';
export const TRAINER_ARTIFACT_IMPORT_PLAN_SCHEMA_ID = 'doppler.trainer-artifact-import-plan/v1';
export const TRAINER_ARTIFACT_PARITY_EVIDENCE_SCHEMA_ID = 'doppler.trainer-artifact-parity-evidence/v1';
export const TRAINER_ARTIFACT_PARITY_RECEIPT_SCHEMA_ID = 'doppler.trainer-artifact-parity-receipt/v1';

export const TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT = 'full_checkpoint';
export const TRAINER_ARTIFACT_KIND_PEFT_ADAPTER = 'peft_adapter';

export const TRANSLATION_FULL_CHECKPOINT_PARITY_CHECKS = Object.freeze([
  'source_artifact_byte_identity',
  'tokenizer_and_prompt_identity',
  'architecture_and_conversion_lineage',
  'evaluation_input_identity',
  'first_token_logits',
  'selected_tokens',
  'completions',
  'exact_browser_artifact_capability',
]);

export const COLUMBO_QWEN_ADAPTER_PARITY_CHECKS = Object.freeze([
  'source_base_vs_converted_base',
  'reference_converted_base_vs_doppler_base',
  'source_adapter_vs_converted_adapter',
  'reference_converted_adapter_vs_doppler_adapter',
  'browser_capability_evaluation',
]);

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const FILE_ROLES_BY_KIND = Object.freeze({
  [TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT]: Object.freeze([
    'weights',
    'config',
    'generation_config',
  ]),
  [TRAINER_ARTIFACT_KIND_PEFT_ADAPTER]: Object.freeze([
    'adapter_weights',
    'adapter_config',
    'doppler_adapter_manifest',
    'runtime_adapter_manifest',
    'training_report',
  ]),
});
const PARITY_CHECKS_BY_PROFILE = Object.freeze({
  translation_full_checkpoint: TRANSLATION_FULL_CHECKPOINT_PARITY_CHECKS,
  columbo_qwen_adapter: COLUMBO_QWEN_ADAPTER_PARITY_CHECKS,
});

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function hashStableJson(value) {
  return sha256Hex(stableJson(value));
}

function requireText(value, label) {
  const text = typeof value === 'string' ? value.trim() : '';
  if (!text) {
    throw new Error(`trainer artifact bridge: ${label} is required.`);
  }
  return text;
}

function requireSha256(value, label) {
  const digest = requireText(value, label).replace(/^sha256:/, '').toLowerCase();
  if (!SHA256_PATTERN.test(digest)) {
    throw new Error(`trainer artifact bridge: ${label} must be a SHA-256 digest.`);
  }
  return digest;
}

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`trainer artifact bridge: ${label} must be a positive integer.`);
  }
  return value;
}

function assertRepoRelativePath(value, label, allowEmpty = false) {
  const path = typeof value === 'string' ? value.trim().replaceAll('\\', '/') : '';
  if (!path && allowEmpty) return '';
  if (!path || path.startsWith('/') || path.split('/').includes('..')) {
    throw new Error(`trainer artifact bridge: ${label} must be a repository-relative path.`);
  }
  return path.replace(/^\.\//, '').replace(/\/$/, '');
}

function normalizeFileIdentity(file, label) {
  if (!isPlainObject(file)) {
    throw new Error(`trainer artifact bridge: ${label} must be an object.`);
  }
  return {
    id: requireText(file.id, `${label}.id`),
    role: requireText(file.role, `${label}.role`),
    repository: requireText(file.repository, `${label}.repository`),
    rootPath: assertRepoRelativePath(file.rootPath, `${label}.rootPath`, true),
    path: assertRepoRelativePath(file.path, `${label}.path`),
    sha256: requireSha256(file.sha256, `${label}.sha256`),
    bytes: requirePositiveInteger(file.bytes, `${label}.bytes`),
  };
}

function normalizeFileList(files, label) {
  if (!Array.isArray(files) || files.length === 0) {
    throw new Error(`trainer artifact bridge: ${label} must be a non-empty array.`);
  }
  const normalized = files.map((file, index) => normalizeFileIdentity(file, `${label}[${index}]`));
  const ids = new Set();
  for (const file of normalized) {
    if (ids.has(file.id)) {
      throw new Error(`trainer artifact bridge: duplicate file id "${file.id}".`);
    }
    ids.add(file.id);
  }
  return normalized;
}

function normalizeArchitecture(architecture) {
  if (!isPlainObject(architecture)) {
    throw new Error('trainer artifact bridge: baseModel.architecture must be an object.');
  }
  const architectures = Array.isArray(architecture.architectures)
    ? architecture.architectures.map((entry, index) => requireText(entry, `baseModel.architecture.architectures[${index}]`))
    : [];
  if (architectures.length === 0) {
    throw new Error('trainer artifact bridge: baseModel.architecture.architectures must be non-empty.');
  }
  return {
    architectures,
    modelType: requireText(architecture.modelType, 'baseModel.architecture.modelType'),
    hiddenSize: requirePositiveInteger(architecture.hiddenSize, 'baseModel.architecture.hiddenSize'),
    intermediateSize: requirePositiveInteger(
      architecture.intermediateSize,
      'baseModel.architecture.intermediateSize'
    ),
    layers: requirePositiveInteger(architecture.layers, 'baseModel.architecture.layers'),
    attentionHeads: requirePositiveInteger(
      architecture.attentionHeads,
      'baseModel.architecture.attentionHeads'
    ),
    keyValueHeads: requirePositiveInteger(
      architecture.keyValueHeads,
      'baseModel.architecture.keyValueHeads'
    ),
    headDim: requirePositiveInteger(architecture.headDim, 'baseModel.architecture.headDim'),
    vocabularySize: requirePositiveInteger(
      architecture.vocabularySize,
      'baseModel.architecture.vocabularySize'
    ),
  };
}

function normalizeSelection(selection) {
  if (!isPlainObject(selection)) {
    throw new Error('trainer artifact bridge: selection must be an object.');
  }
  const status = requireText(selection.status, 'selection.status');
  if (status !== 'not_selected' && status !== 'selected') {
    throw new Error('trainer artifact bridge: selection.status must be "not_selected" or "selected".');
  }
  const receipt = selection.receipt == null ? null : requireText(selection.receipt, 'selection.receipt');
  if (status === 'not_selected' && receipt !== null) {
    throw new Error('trainer artifact bridge: an unselected artifact cannot carry a selection receipt.');
  }
  if (status === 'selected' && receipt === null) {
    throw new Error('trainer artifact bridge: a selected artifact requires a selection receipt.');
  }
  return {
    authority: requireText(selection.authority, 'selection.authority'),
    status,
    receipt,
  };
}

function normalizeParity(parity, artifactKind) {
  if (!isPlainObject(parity)) {
    throw new Error('trainer artifact bridge: parity must be an object.');
  }
  const profile = requireText(parity.profile, 'parity.profile');
  const canonicalChecks = PARITY_CHECKS_BY_PROFILE[profile];
  if (!canonicalChecks) {
    throw new Error(`trainer artifact bridge: unsupported parity profile "${profile}".`);
  }
  const expectedProfile = artifactKind === TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT
    ? 'translation_full_checkpoint'
    : 'columbo_qwen_adapter';
  if (profile !== expectedProfile) {
    throw new Error(
      `trainer artifact bridge: artifact kind "${artifactKind}" requires parity profile "${expectedProfile}".`
    );
  }
  const requiredChecks = Array.isArray(parity.requiredChecks) ? parity.requiredChecks : [];
  if (stableJson(requiredChecks) !== stableJson(canonicalChecks)) {
    throw new Error(`trainer artifact bridge: parity.requiredChecks must match profile "${profile}".`);
  }
  return { profile, requiredChecks: [...canonicalChecks] };
}

function assertArtifactRoles(artifact) {
  const expectedRoles = FILE_ROLES_BY_KIND[artifact.kind];
  const roles = new Set(artifact.files.map((file) => file.role));
  for (const role of expectedRoles) {
    if (!roles.has(role)) {
      throw new Error(`trainer artifact bridge: artifact file role "${role}" is required.`);
    }
  }
}

function normalizeDescriptor(descriptor) {
  if (!isPlainObject(descriptor)) {
    throw new Error('trainer artifact bridge: descriptor must be an object.');
  }
  if (descriptor.schema !== TRAINER_ARTIFACT_BRIDGE_SCHEMA_ID) {
    throw new Error(`trainer artifact bridge: schema must be "${TRAINER_ARTIFACT_BRIDGE_SCHEMA_ID}".`);
  }
  const kind = requireText(descriptor.artifact?.kind, 'artifact.kind');
  if (!FILE_ROLES_BY_KIND[kind]) {
    throw new Error(`trainer artifact bridge: unsupported artifact kind "${kind}".`);
  }
  const role = requireText(descriptor.artifact?.role, 'artifact.role');
  if (
    role !== 'diagnostic_baseline'
    && role !== 'diagnostic_candidate'
    && role !== 'selected_candidate'
  ) {
    throw new Error('trainer artifact bridge: artifact.role is unsupported.');
  }
  const artifact = {
    kind,
    role,
    format: requireText(descriptor.artifact?.format, 'artifact.format'),
    repository: requireText(descriptor.artifact?.repository, 'artifact.repository'),
    rootPath: assertRepoRelativePath(descriptor.artifact?.rootPath, 'artifact.rootPath', true),
    files: normalizeFileList(descriptor.artifact?.files, 'artifact.files'),
  };
  assertArtifactRoles(artifact);

  const tokenizerFiles = normalizeFileList(
    descriptor.baseModel?.tokenizer?.files,
    'baseModel.tokenizer.files'
  );
  const promptContract = normalizeFileIdentity(
    descriptor.baseModel?.tokenizer?.promptContract,
    'baseModel.tokenizer.promptContract'
  );
  const selection = normalizeSelection(descriptor.selection);
  if (
    (role === 'diagnostic_baseline' || role === 'diagnostic_candidate')
    && selection.status !== 'not_selected'
  ) {
    throw new Error('trainer artifact bridge: a diagnostic artifact cannot be selected.');
  }
  if (role === 'selected_candidate' && selection.status !== 'selected') {
    throw new Error('trainer artifact bridge: a selected candidate requires selected status.');
  }

  const conversionConfig = descriptor.conversion?.config == null
    ? null
    : normalizeFileIdentity(descriptor.conversion.config, 'conversion.config');
  const evaluationFiles = normalizeFileList(descriptor.evaluation?.files, 'evaluation.files');
  const normalized = {
    schema: TRAINER_ARTIFACT_BRIDGE_SCHEMA_ID,
    bridgeId: requireText(descriptor.bridgeId, 'bridgeId'),
    sourceContractId: requireText(descriptor.sourceContractId, 'sourceContractId'),
    artifact,
    baseModel: {
      modelId: requireText(descriptor.baseModel?.modelId, 'baseModel.modelId'),
      checkpointSha256: requireSha256(
        descriptor.baseModel?.checkpointSha256,
        'baseModel.checkpointSha256'
      ),
      tokenizer: { files: tokenizerFiles, promptContract },
      architecture: normalizeArchitecture(descriptor.baseModel?.architecture),
    },
    conversion: {
      owner: requireText(descriptor.conversion?.owner, 'conversion.owner'),
      sourceArtifactSha256: requireSha256(
        descriptor.conversion?.sourceArtifactSha256,
        'conversion.sourceArtifactSha256'
      ),
      config: conversionConfig,
      runtimeArtifact: isPlainObject(descriptor.conversion?.runtimeArtifact)
        ? stableSortObject(descriptor.conversion.runtimeArtifact)
        : null,
    },
    evaluation: {
      populationRole: requireText(descriptor.evaluation?.populationRole, 'evaluation.populationRole'),
      files: evaluationFiles,
    },
    selection,
    parity: normalizeParity(descriptor.parity, kind),
    claimBoundary: requireText(descriptor.claimBoundary, 'claimBoundary'),
  };
  if (normalized.conversion.sourceArtifactSha256 !== normalized.baseModel.checkpointSha256) {
    throw new Error('trainer artifact bridge: conversion source hash must match the base checkpoint hash.');
  }
  return normalized;
}

function toFileIdentity(id, role, repository, rootPath, identity) {
  return {
    id,
    role,
    repository,
    rootPath,
    path: identity.path,
    sha256: identity.sha256,
    bytes: identity.bytes,
  };
}

function splitQualifiedPath(value, fallbackRepository) {
  const text = requireText(value, 'qualified file path');
  const delimiter = text.indexOf(':');
  if (delimiter < 0) {
    return { repository: fallbackRepository, path: text };
  }
  return {
    repository: text.slice(0, delimiter),
    path: text.slice(delimiter + 1),
  };
}

export function normalizeGammaTrainerArtifactHandoff(contract) {
  if (!isPlainObject(contract)) {
    throw new Error('trainer artifact bridge: Gamma handoff contract must be an object.');
  }
  const checkpoint = contract.sourceCheckpoint;
  const repository = requireText(checkpoint?.repository, 'sourceCheckpoint.repository');
  const checkpointRoot = assertRepoRelativePath(checkpoint?.path, 'sourceCheckpoint.path');
  const conversionConfig = contract.conversionLineage?.conversionConfig;
  const artifactFiles = [
    toFileIdentity('checkpoint.weights', 'weights', repository, checkpointRoot, checkpoint.weights),
    toFileIdentity('checkpoint.config', 'config', repository, checkpointRoot, checkpoint.config),
    toFileIdentity(
      'checkpoint.generation_config',
      'generation_config',
      repository,
      checkpointRoot,
      checkpoint.generationConfig
    ),
  ];
  const tokenizerFiles = contract.tokenizer.files.map((identity, index) => toFileIdentity(
    `tokenizer.${index}`,
    'tokenizer_asset',
    repository,
    checkpointRoot,
    identity
  ));
  const promptContract = toFileIdentity(
    'tokenizer.prompt_contract',
    'prompt_contract',
    repository,
    checkpointRoot,
    contract.tokenizer.promptContract
  );
  const evaluationFiles = [
    ['evaluation.population', 'population', contract.evaluationInputs.population],
    ['evaluation.summary', 'summary', contract.evaluationInputs.summary],
    ['evaluation.predictions', 'predictions', contract.evaluationInputs.predictions],
  ].map(([id, role, identity]) => {
    const qualified = splitQualifiedPath(identity.path, repository);
    return toFileIdentity(id, role, qualified.repository, '', { ...identity, path: qualified.path });
  });
  const selectionStatus = contract.selection?.status === 'gamma_selected' ? 'selected' : 'not_selected';

  return normalizeDescriptor({
    schema: TRAINER_ARTIFACT_BRIDGE_SCHEMA_ID,
    bridgeId: `bridge.${contract.contractId}`,
    sourceContractId: contract.contractId,
    artifact: {
      kind: TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT,
      role: contract.artifactRole,
      format: checkpoint.format,
      repository,
      rootPath: checkpointRoot,
      files: artifactFiles,
    },
    baseModel: {
      modelId: contract.conversionLineage.diagnosticHostedArtifact.modelId,
      checkpointSha256: checkpoint.weights.sha256,
      tokenizer: { files: tokenizerFiles, promptContract },
      architecture: contract.architecture,
    },
    conversion: {
      owner: contract.authorities.conversionAndParity,
      sourceArtifactSha256: contract.conversionLineage.sourceWeightsSha256,
      config: toFileIdentity(
        'conversion.config',
        'conversion_config',
        'clocksmith/doppler',
        '',
        conversionConfig
      ),
      runtimeArtifact: contract.conversionLineage.diagnosticHostedArtifact,
    },
    evaluation: {
      populationRole: contract.evaluationInputs.populationRole,
      files: evaluationFiles,
    },
    selection: {
      authority: contract.authorities.bf16Selection,
      status: selectionStatus,
      receipt: contract.selection.receipt,
    },
    parity: {
      profile: 'translation_full_checkpoint',
      requiredChecks: contract.parity.requiredChecks,
    },
    claimBoundary: contract.claimBoundary,
  });
}

export function validateTrainerArtifactBridgeDescriptor(descriptor) {
  try {
    return { valid: true, descriptor: normalizeDescriptor(descriptor), errors: [] };
  } catch (error) {
    return {
      valid: false,
      descriptor: null,
      errors: [error instanceof Error ? error.message : String(error)],
    };
  }
}

export function assertTrainerArtifactCandidateEntry(descriptor) {
  const normalized = normalizeDescriptor(descriptor);
  if (normalized.artifact.role !== 'selected_candidate') {
    throw new Error('trainer artifact bridge: candidate entry denied; artifact is not a selected candidate.');
  }
  if (normalized.selection.status !== 'selected' || !normalized.selection.receipt) {
    throw new Error('trainer artifact bridge: candidate entry denied; selection receipt is required.');
  }
  const expectedAuthority = normalized.artifact.kind === TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT
    ? 'clocksmith/gamma'
    : 'clocksmith/columbo';
  if (normalized.selection.authority !== expectedAuthority) {
    throw new Error(
      `trainer artifact bridge: candidate entry denied; selection authority must be "${expectedAuthority}".`
    );
  }
  return normalized;
}

export function buildTrainerArtifactImportPlan(descriptor, verificationReceipt = null) {
  const normalized = normalizeDescriptor(descriptor);
  if (verificationReceipt && verificationReceipt.ok !== true) {
    throw new Error('trainer artifact bridge: import plan requires a passing verification receipt.');
  }
  const candidateCompetitionAllowed = normalized.artifact.role === 'selected_candidate'
    && normalized.selection.status === 'selected'
    && Boolean(normalized.selection.receipt);
  const requiredRoles = FILE_ROLES_BY_KIND[normalized.artifact.kind];
  const filesByRole = Object.fromEntries(
    normalized.artifact.files.map((file) => [file.role, file.id])
  );
  const plan = {
    schema: TRAINER_ARTIFACT_IMPORT_PLAN_SCHEMA_ID,
    bridgeId: normalized.bridgeId,
    artifactKind: normalized.artifact.kind,
    entrypoint: normalized.artifact.kind === TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT
      ? 'resolveNodeSourceRuntimeBundle'
      : 'loadLoRAWeights',
    source: {
      repository: normalized.artifact.repository,
      rootPath: normalized.artifact.rootPath,
      format: normalized.artifact.format,
      requiredRoles,
      filesByRole,
    },
    baseModel: normalized.baseModel,
    conversion: normalized.conversion,
    verificationReceiptHash: verificationReceipt?.receiptHash ?? null,
    admission: {
      identityVerificationRequired: true,
      parityExecutionAllowed: true,
      candidateCompetitionAllowed,
      promotionAllowed: false,
      selectionAuthority: normalized.selection.authority,
      selectionReceipt: normalized.selection.receipt,
    },
  };
  return { ...plan, planHash: hashStableJson(plan) };
}

export function buildTrainerArtifactParityTemplate(descriptor, verificationReceipt) {
  const normalized = normalizeDescriptor(descriptor);
  if (!verificationReceipt || verificationReceipt.ok !== true) {
    throw new Error('trainer artifact bridge: parity template requires a passing verification receipt.');
  }
  const identityCheckIds = new Set([
    'source_artifact_byte_identity',
    'tokenizer_and_prompt_identity',
    'architecture_and_conversion_lineage',
    'evaluation_input_identity',
  ]);
  return {
    schema: TRAINER_ARTIFACT_PARITY_EVIDENCE_SCHEMA_ID,
    bridgeId: normalized.bridgeId,
    profile: normalized.parity.profile,
    artifactIdentitySha256: verificationReceipt.artifactIdentitySha256,
    checks: normalized.parity.requiredChecks.map((id) => ({
      id,
      status: identityCheckIds.has(id) ? 'pass' : 'pending',
      evidenceHash: identityCheckIds.has(id) ? verificationReceipt.receiptHash : null,
      artifactIdentitySha256: verificationReceipt.artifactIdentitySha256,
      upstreamDecision: identityCheckIds.has(id) ? 'pass' : null,
    })),
  };
}

export function verifyTrainerArtifactParityEvidence(descriptor, verificationReceipt, evidence) {
  const normalized = normalizeDescriptor(descriptor);
  if (!verificationReceipt || verificationReceipt.ok !== true) {
    throw new Error('trainer artifact bridge: parity verification requires a passing identity receipt.');
  }
  const blockers = [];
  if (!isPlainObject(evidence) || evidence.schema !== TRAINER_ARTIFACT_PARITY_EVIDENCE_SCHEMA_ID) {
    blockers.push('invalid_parity_evidence_schema');
  }
  if (evidence?.bridgeId !== normalized.bridgeId) blockers.push('parity_bridge_id_mismatch');
  if (evidence?.profile !== normalized.parity.profile) blockers.push('parity_profile_mismatch');
  if (evidence?.artifactIdentitySha256 !== verificationReceipt.artifactIdentitySha256) {
    blockers.push('parity_artifact_identity_mismatch');
  }
  const checks = Array.isArray(evidence?.checks) ? evidence.checks : [];
  const checkById = new Map();
  for (const check of checks) {
    const id = typeof check?.id === 'string' ? check.id : '';
    if (!id) {
      blockers.push('parity_check_missing_id');
      continue;
    }
    if (checkById.has(id)) blockers.push(`duplicate_parity_check:${id}`);
    checkById.set(id, check);
  }
  for (const id of normalized.parity.requiredChecks) {
    const check = checkById.get(id);
    if (!check) {
      blockers.push(`missing_parity_check:${id}`);
      continue;
    }
    if (check.status !== 'pass') blockers.push(`parity_check_not_passed:${id}`);
    if (check.upstreamDecision !== 'pass') blockers.push(`parity_upstream_not_passed:${id}`);
    if (check.artifactIdentitySha256 !== verificationReceipt.artifactIdentitySha256) {
      blockers.push(`parity_check_identity_mismatch:${id}`);
    }
    const evidenceHash = typeof check.evidenceHash === 'string'
      ? check.evidenceHash.replace(/^sha256:/, '').toLowerCase()
      : '';
    if (!SHA256_PATTERN.test(evidenceHash)) blockers.push(`invalid_parity_evidence_hash:${id}`);
  }
  for (const id of checkById.keys()) {
    if (!normalized.parity.requiredChecks.includes(id)) blockers.push(`unexpected_parity_check:${id}`);
  }
  const core = {
    schema: TRAINER_ARTIFACT_PARITY_RECEIPT_SCHEMA_ID,
    bridgeId: normalized.bridgeId,
    profile: normalized.parity.profile,
    artifactIdentitySha256: verificationReceipt.artifactIdentitySha256,
    identityReceiptHash: verificationReceipt.receiptHash,
    decision: blockers.length === 0 ? 'pass' : 'block',
    blockers: [...new Set(blockers)].sort(),
    checkEvidenceHashes: Object.fromEntries(
      normalized.parity.requiredChecks.map((id) => [id, checkById.get(id)?.evidenceHash ?? null])
    ),
  };
  return { ...core, receiptHash: hashStableJson(core) };
}
