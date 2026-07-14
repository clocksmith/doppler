import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';

import {
  TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT,
  TRAINER_ARTIFACT_KIND_PEFT_ADAPTER,
  buildTrainerArtifactImportPlan,
  normalizeGammaTrainerArtifactHandoff,
  validateTrainerArtifactBridgeDescriptor,
} from '../experimental/bridge/trainer-artifact-bridge.js';
import { loadLoRAWeights } from '../experimental/adapters/lora-loader.js';
import { resolveNodeSourceRuntimeBundle } from './node-source-runtime.js';
import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const TRAINER_ARTIFACT_HANDOFF_VERIFICATION_SCHEMA_ID =
  'doppler.trainer-artifact-handoff-verification/v1';
export const TRAINER_ARTIFACT_IMPORT_RECEIPT_SCHEMA_ID =
  'doppler.trainer-artifact-import-receipt/v1';

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function hashStableJson(value) {
  return sha256Hex(stableJson(value));
}

function normalizeRepositoryRoots(repositoryRoots) {
  if (!repositoryRoots || typeof repositoryRoots !== 'object' || Array.isArray(repositoryRoots)) {
    throw new Error('trainer artifact handoff: repositoryRoots must be an object.');
  }
  const normalized = {};
  for (const [repository, root] of Object.entries(repositoryRoots)) {
    const repositoryId = String(repository || '').trim();
    const rootPath = String(root || '').trim();
    if (!repositoryId || !rootPath) {
      throw new Error('trainer artifact handoff: repositoryRoots entries require repository and path.');
    }
    normalized[repositoryId] = path.resolve(rootPath);
  }
  return normalized;
}

function resolveWithinRoot(root, relativePath, label) {
  const resolved = path.resolve(root, relativePath);
  const boundary = path.relative(root, resolved);
  if (boundary === '..' || boundary.startsWith(`..${path.sep}`) || path.isAbsolute(boundary)) {
    throw new Error(`trainer artifact handoff: ${label} escapes its repository root.`);
  }
  return resolved;
}

function resolveFileIdentityPath(file, repositoryRoots) {
  const repositoryRoot = repositoryRoots[file.repository];
  if (!repositoryRoot) {
    throw new Error(
      `trainer artifact handoff: repository root is required for "${file.repository}" (${file.id}).`
    );
  }
  const artifactRoot = resolveWithinRoot(repositoryRoot, file.rootPath || '.', `${file.id}.rootPath`);
  return resolveWithinRoot(artifactRoot, file.path, `${file.id}.path`);
}

async function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(filePath);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

async function verifyFileIdentity(file, repositoryRoots) {
  const observed = { bytes: null, sha256: null };
  try {
    const filePath = resolveFileIdentityPath(file, repositoryRoots);
    const stat = await fs.stat(filePath);
    if (!stat.isFile()) {
      throw new Error('path is not a file');
    }
    observed.bytes = stat.size;
    observed.sha256 = await sha256File(filePath);
    const errors = [];
    if (observed.bytes !== file.bytes) errors.push('byte_length_mismatch');
    if (observed.sha256 !== file.sha256) errors.push('sha256_mismatch');
    return {
      id: file.id,
      role: file.role,
      repository: file.repository,
      rootPath: file.rootPath,
      path: file.path,
      expected: { bytes: file.bytes, sha256: file.sha256 },
      observed,
      ok: errors.length === 0,
      errors,
    };
  } catch (error) {
    return {
      id: file.id,
      role: file.role,
      repository: file.repository,
      rootPath: file.rootPath,
      path: file.path,
      expected: { bytes: file.bytes, sha256: file.sha256 },
      observed,
      ok: false,
      errors: [error instanceof Error ? error.message : String(error)],
    };
  }
}

function collectDescriptorFiles(descriptor, recordedImportReceipt = null) {
  return [
    ...descriptor.artifact.files,
    ...descriptor.baseModel.tokenizer.files,
    descriptor.baseModel.tokenizer.promptContract,
    ...(descriptor.conversion.config ? [descriptor.conversion.config] : []),
    ...descriptor.evaluation.files,
    ...(recordedImportReceipt ? [recordedImportReceipt] : []),
  ];
}

function normalizeRecordedImportReceipt(contract) {
  const verification = contract?.importVerification;
  const receipt = verification?.receipt;
  if (verification?.status !== 'verified' || !receipt) return null;
  return {
    id: 'handoff.recorded_import_receipt',
    role: 'recorded_import_receipt',
    repository: 'clocksmith/doppler',
    rootPath: '',
    path: receipt.path,
    sha256: receipt.sha256,
    bytes: receipt.bytes,
    expectedImportReceiptHash: verification.importReceiptHash,
  };
}

async function verifyRecordedImportReceiptBinding(file, repositoryRoots, fileResults) {
  if (!file) return null;
  const fileResult = fileResults.find((entry) => entry.id === file.id);
  const errors = [];
  if (fileResult?.ok !== true) {
    errors.push('recorded_import_receipt_file_identity_failed');
  } else {
    try {
      const receiptPath = resolveFileIdentityPath(file, repositoryRoots);
      const receipt = JSON.parse(await fs.readFile(receiptPath, 'utf8'));
      if (receipt?.importReceipt?.receiptHash !== file.expectedImportReceiptHash) {
        errors.push('recorded_import_receipt_hash_binding_mismatch');
      }
      if (receipt?.importReceipt?.candidateCompetitionAllowed !== false) {
        errors.push('recorded_import_receipt_candidate_gate_mismatch');
      }
      if (receipt?.importReceipt?.promotionAllowed !== false) {
        errors.push('recorded_import_receipt_promotion_gate_mismatch');
      }
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
    }
  }
  return {
    id: 'recorded_import_receipt_identity',
    ok: errors.length === 0,
    fileIds: [file.id],
    errors,
  };
}

function architectureFieldRows(expected, config) {
  return [
    ['architectures', expected.architectures, config.architectures],
    ['modelType', expected.modelType, config.model_type],
    ['hiddenSize', expected.hiddenSize, config.hidden_size],
    ['intermediateSize', expected.intermediateSize, config.intermediate_size],
    ['layers', expected.layers, config.num_hidden_layers],
    ['attentionHeads', expected.attentionHeads, config.num_attention_heads],
    ['keyValueHeads', expected.keyValueHeads, config.num_key_value_heads],
    ['headDim', expected.headDim, config.head_dim],
    ['vocabularySize', expected.vocabularySize, config.vocab_size],
  ];
}

function valuesEqual(left, right) {
  return stableJson(left) === stableJson(right);
}

async function verifyArchitecture(descriptor, repositoryRoots, fileResults) {
  const configFile = descriptor.artifact.files.find((file) => file.role === 'config');
  if (!configFile) {
    return {
      ok: descriptor.artifact.kind === TRAINER_ARTIFACT_KIND_PEFT_ADAPTER,
      sourceFileId: null,
      fields: [],
      errors: descriptor.artifact.kind === TRAINER_ARTIFACT_KIND_PEFT_ADAPTER
        ? []
        : ['config_file_role_missing'],
    };
  }
  if (fileResults.find((entry) => entry.id === configFile.id)?.ok !== true) {
    return {
      ok: false,
      sourceFileId: configFile.id,
      fields: [],
      errors: ['config_file_identity_failed'],
    };
  }
  try {
    const filePath = resolveFileIdentityPath(configFile, repositoryRoots);
    const config = JSON.parse(await fs.readFile(filePath, 'utf8'));
    const fields = architectureFieldRows(descriptor.baseModel.architecture, config).map(
      ([field, expected, observed]) => ({
        field,
        expected,
        observed: observed ?? null,
        ok: valuesEqual(expected, observed),
      })
    );
    return {
      ok: fields.every((entry) => entry.ok),
      sourceFileId: configFile.id,
      fields,
      errors: fields.filter((entry) => !entry.ok).map((entry) => `architecture_mismatch:${entry.field}`),
    };
  } catch (error) {
    return {
      ok: false,
      sourceFileId: configFile.id,
      fields: [],
      errors: [error instanceof Error ? error.message : String(error)],
    };
  }
}

function groupCheck(id, fileResults, fileIds, extraErrors = []) {
  const idSet = new Set(fileIds);
  const relevant = fileResults.filter((entry) => idSet.has(entry.id));
  const errors = [
    ...relevant.flatMap((entry) => entry.errors.map((error) => `${entry.id}:${error}`)),
    ...extraErrors,
  ];
  return {
    id,
    ok: errors.length === 0 && relevant.length === fileIds.length,
    fileIds,
    errors,
  };
}

function resolveDescriptor(input) {
  const candidate = input?.schema === 'doppler.trainer-artifact-bridge/v1'
    ? input
    : normalizeGammaTrainerArtifactHandoff(input);
  const validation = validateTrainerArtifactBridgeDescriptor(candidate);
  if (!validation.valid || !validation.descriptor) {
    throw new Error(validation.errors.join('; '));
  }
  return validation.descriptor;
}

function artifactIdentity(descriptor) {
  return {
    bridgeId: descriptor.bridgeId,
    artifactKind: descriptor.artifact.kind,
    artifactFiles: descriptor.artifact.files.map((file) => ({
      id: file.id,
      sha256: file.sha256,
      bytes: file.bytes,
    })),
    tokenizerFiles: descriptor.baseModel.tokenizer.files.map((file) => ({
      id: file.id,
      sha256: file.sha256,
      bytes: file.bytes,
    })),
    promptContract: {
      sha256: descriptor.baseModel.tokenizer.promptContract.sha256,
      bytes: descriptor.baseModel.tokenizer.promptContract.bytes,
    },
    architecture: descriptor.baseModel.architecture,
    conversion: {
      sourceArtifactSha256: descriptor.conversion.sourceArtifactSha256,
      configSha256: descriptor.conversion.config?.sha256 ?? null,
      runtimeArtifact: descriptor.conversion.runtimeArtifact,
    },
    evaluationFiles: descriptor.evaluation.files.map((file) => ({
      id: file.id,
      sha256: file.sha256,
      bytes: file.bytes,
    })),
  };
}

export async function loadTrainerArtifactHandoffContract(contractPath) {
  const resolvedPath = path.resolve(String(contractPath || ''));
  if (!String(contractPath || '').trim()) {
    throw new Error('trainer artifact handoff: contractPath is required.');
  }
  return JSON.parse(await fs.readFile(resolvedPath, 'utf8'));
}

export function resolveTrainerArtifactHandoffDescriptor(contract) {
  return resolveDescriptor(contract);
}

export async function verifyTrainerArtifactHandoff(options = {}) {
  const contract = options.contract ?? await loadTrainerArtifactHandoffContract(options.contractPath);
  const descriptor = resolveDescriptor(contract);
  const repositoryRoots = normalizeRepositoryRoots(options.repositoryRoots);
  const recordedImportReceipt = normalizeRecordedImportReceipt(contract);
  const files = collectDescriptorFiles(descriptor, recordedImportReceipt);
  const fileResults = [];
  for (const file of files) {
    fileResults.push(await verifyFileIdentity(file, repositoryRoots));
  }
  const architecture = await verifyArchitecture(descriptor, repositoryRoots, fileResults);
  const artifactIds = descriptor.artifact.files.map((file) => file.id);
  const tokenizerIds = [
    ...descriptor.baseModel.tokenizer.files.map((file) => file.id),
    descriptor.baseModel.tokenizer.promptContract.id,
  ];
  const conversionIds = descriptor.conversion.config ? [descriptor.conversion.config.id] : [];
  const evaluationIds = descriptor.evaluation.files.map((file) => file.id);
  const conversionErrors = [...architecture.errors];
  if (descriptor.conversion.sourceArtifactSha256 !== descriptor.baseModel.checkpointSha256) {
    conversionErrors.push('conversion_source_checkpoint_hash_mismatch');
  }
  const recordedImportCheck = await verifyRecordedImportReceiptBinding(
    recordedImportReceipt,
    repositoryRoots,
    fileResults
  );
  const checks = [
    groupCheck('source_artifact_byte_identity', fileResults, artifactIds),
    groupCheck('tokenizer_and_prompt_identity', fileResults, tokenizerIds),
    groupCheck(
      'architecture_and_conversion_lineage',
      fileResults,
      conversionIds,
      conversionErrors
    ),
    groupCheck('evaluation_input_identity', fileResults, evaluationIds),
    ...(recordedImportCheck ? [recordedImportCheck] : []),
  ];
  const identity = artifactIdentity(descriptor);
  const core = {
    schema: TRAINER_ARTIFACT_HANDOFF_VERIFICATION_SCHEMA_ID,
    bridgeId: descriptor.bridgeId,
    sourceContractId: descriptor.sourceContractId,
    artifactKind: descriptor.artifact.kind,
    artifactRole: descriptor.artifact.role,
    ok: checks.every((check) => check.ok),
    verifiedAt: options.verifiedAt ?? new Date().toISOString(),
    artifactIdentitySha256: hashStableJson(identity),
    selection: descriptor.selection,
    admission: {
      baselineImportAllowed: true,
      parityExecutionAllowed: true,
      candidateCompetitionAllowed: descriptor.artifact.role === 'selected_candidate'
        && descriptor.selection.status === 'selected'
        && Boolean(descriptor.selection.receipt),
      promotionAllowed: false,
    },
    checks,
    files: fileResults,
    architecture,
  };
  return {
    descriptor,
    receipt: { ...core, receiptHash: hashStableJson(core) },
    repositoryRoots,
  };
}

function resolveArtifactRoot(descriptor, repositoryRoots) {
  const repositoryRoot = repositoryRoots[descriptor.artifact.repository];
  if (!repositoryRoot) {
    throw new Error(
      `trainer artifact handoff: repository root is required for "${descriptor.artifact.repository}".`
    );
  }
  return resolveWithinRoot(repositoryRoot, descriptor.artifact.rootPath || '.', 'artifact.rootPath');
}

function fileByRole(descriptor, role) {
  const file = descriptor.artifact.files.find((entry) => entry.role === role);
  if (!file) {
    throw new Error(`trainer artifact handoff: artifact role "${role}" is required.`);
  }
  return file;
}

export async function importTrainerArtifactHandoff(options = {}) {
  const verification = await verifyTrainerArtifactHandoff(options);
  if (!verification.receipt.ok) {
    throw new Error('trainer artifact handoff: import denied because identity verification failed.');
  }
  const { descriptor, repositoryRoots } = verification;
  const plan = buildTrainerArtifactImportPlan(descriptor, verification.receipt);
  const artifactRoot = resolveArtifactRoot(descriptor, repositoryRoots);
  let imported;
  if (descriptor.artifact.kind === TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT) {
    const runtimeResolver = options.runtimeResolver ?? resolveNodeSourceRuntimeBundle;
    const runtimeBundle = await runtimeResolver({
      inputPath: artifactRoot,
      modelId: descriptor.baseModel.modelId,
      verifyHashes: true,
      runtimeConfig: options.runtimeConfig ?? null,
    });
    if (!runtimeBundle || runtimeBundle.sourceKind !== 'safetensors') {
      throw new Error('trainer artifact handoff: full checkpoint did not resolve as a SafeTensors runtime bundle.');
    }
    imported = {
      kind: TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT,
      sourceKind: runtimeBundle.sourceKind,
      sourceRoot: runtimeBundle.sourceRoot,
      modelId: runtimeBundle.model?.modelId ?? descriptor.baseModel.modelId,
      runtimeBundle,
    };
  } else if (descriptor.artifact.kind === TRAINER_ARTIFACT_KIND_PEFT_ADAPTER) {
    const manifestFile = fileByRole(descriptor, 'doppler_adapter_manifest');
    const manifestPath = resolveFileIdentityPath(manifestFile, repositoryRoots);
    const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
    const adapterLoader = options.adapterLoader ?? loadLoRAWeights;
    const loaded = await adapterLoader(manifest, {
      basePath: path.dirname(manifestPath),
      resolvePath(relativePath) {
        return resolveWithinRoot(path.dirname(manifestPath), relativePath, 'adapter weightsPath');
      },
      async readFile(filePath) {
        const bytes = await fs.readFile(filePath);
        return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      },
    });
    imported = {
      kind: TRAINER_ARTIFACT_KIND_PEFT_ADAPTER,
      manifestPath,
      adapterId: manifest.id,
      baseModel: manifest.baseModel,
      loaded,
    };
  } else {
    throw new Error(`trainer artifact handoff: unsupported import kind "${descriptor.artifact.kind}".`);
  }
  const receiptCore = {
    schema: TRAINER_ARTIFACT_IMPORT_RECEIPT_SCHEMA_ID,
    bridgeId: descriptor.bridgeId,
    artifactKind: descriptor.artifact.kind,
    identityReceiptHash: verification.receipt.receiptHash,
    importPlanHash: plan.planHash,
    importedIdentity: descriptor.artifact.kind === TRAINER_ARTIFACT_KIND_FULL_CHECKPOINT
      ? { modelId: imported.modelId, sourceKind: imported.sourceKind }
      : { adapterId: imported.adapterId, baseModel: imported.baseModel },
    candidateCompetitionAllowed: plan.admission.candidateCompetitionAllowed,
    promotionAllowed: false,
  };
  return {
    descriptor,
    verification: verification.receipt,
    plan,
    imported,
    receipt: { ...receiptCore, receiptHash: hashStableJson(receiptCore) },
  };
}
