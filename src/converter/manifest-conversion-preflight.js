import { expandExecutionV1 } from '../config/schema/index.js';
import { resolveTensorRole } from '../formats/rdrr/index.js';
import { sha256Hex } from '../formats/sha256.js';
import { stableSortObject } from '../formats/stable-sort-object.js';
import { normalizeQuantTag } from './quantization-info.js';
import { resolveConversionPlan } from './conversion-plan.js';
import { validateSafetensorsIndexEvidence } from './safetensors-header-evidence.js';
import { createTensorRoleClosureReceipt } from './tensor-role-closure.js';

export const MANIFEST_CONVERSION_PREFLIGHT_SCHEMA_ID = 'doppler.manifest-conversion-preflight/v1';

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function digest(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

function requireObject(value, label) {
  if (!isObject(value)) throw new Error(`${label} must be an object.`);
  return value;
}

function requireAuthor(author) {
  if (!isObject(author) || !['human', 'ai', 'tool'].includes(author.kind)
    || typeof author.actor !== 'string' || !author.actor.trim()) {
    throw new Error('Manifest conversion preflight requires attributable authorship.');
  }
}

function requireEqual(observed, expected, label) {
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error(`${label} does not match its promoted evidence.`);
  }
}

function selectScopedTensors(headers, tensorPolicy) {
  const roots = new Set(tensorPolicy.rootBindings.map((binding) => binding.name));
  return Object.entries(headers.tensors)
    .filter(([name]) => roots.has(name) || tensorPolicy.scopePrefixes.some((prefix) => name.startsWith(prefix)))
    .map(([name, descriptor]) => ({ name, ...descriptor }))
    .sort((left, right) => left.name.localeCompare(right.name));
}

function countBy(values, select) {
  const counts = {};
  for (const value of values) {
    const key = select(value);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)));
}

function jsonClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function validateSourceAcquisition(receipt, headers) {
  requireObject(receipt, 'Source acquisition receipt');
  if (receipt.schema !== 'doppler.source-acquisition-receipt/v1' || receipt.complete !== true) {
    throw new Error('Manifest conversion preflight requires complete source acquisition evidence.');
  }
  for (const field of ['checkpointId', 'repository', 'revision']) {
    if (receipt[field] !== headers[field]) {
      throw new Error(`Source acquisition ${field} does not match header evidence.`);
    }
  }
  if (!Array.isArray(receipt.files) || receipt.files.some((file) => file.verified !== true)) {
    throw new Error('Source acquisition receipt contains unverified files.');
  }
  const expectedShards = [
    headers.sourceFile,
    ...headers.additionalSourceHeaders.map((header) => header.sourceFile),
  ].sort();
  const acquiredShards = receipt.files
    .filter((file) => file.role === 'weight-shard')
    .map((file) => file.path)
    .sort();
  requireEqual(acquiredShards, expectedShards, 'Acquired weight-shard closure');
  return receipt;
}

export function createManifestConversionPreflightReceipt({
  rawConfig,
  conversionConfig,
  semanticReceipt,
  headers,
  weightIndex,
  tensorPolicy,
  tensorClosureReceipt,
  sourceAcquisitionReceipt,
  policy,
}) {
  requireObject(policy, 'Manifest conversion preflight policy');
  if (policy.schema !== MANIFEST_CONVERSION_PREFLIGHT_SCHEMA_ID) {
    throw new Error(`Manifest conversion preflight requires policy "${MANIFEST_CONVERSION_PREFLIGHT_SCHEMA_ID}".`);
  }
  requireAuthor(policy.author);
  requireObject(semanticReceipt, 'Semantic lowering receipt');
  if (semanticReceipt.schema !== 'doppler.semantic-manifest-lowering-receipt/v1') {
    throw new Error('Manifest conversion preflight requires a semantic lowering receipt.');
  }
  requireEqual(conversionConfig, semanticReceipt.conversionConfig, 'Conversion config');
  if (digest(conversionConfig) !== semanticReceipt.conversionConfigDigest) {
    throw new Error('Conversion config digest does not match semantic lowering evidence.');
  }
  const recreatedClosure = createTensorRoleClosureReceipt({
    modelIR: semanticReceipt.modelIR,
    headers,
    policy: tensorPolicy,
  });
  requireEqual(tensorClosureReceipt, recreatedClosure, 'Tensor-role closure receipt');
  if (tensorClosureReceipt.complete !== true
    || tensorClosureReceipt.missingTensors.length > 0
    || tensorClosureReceipt.unexpectedTensors.length > 0) {
    throw new Error('Manifest conversion preflight requires complete tensor-role closure.');
  }
  const acquisition = validateSourceAcquisition(sourceAcquisitionReceipt, headers);
  const indexEvidence = validateSafetensorsIndexEvidence(headers, weightIndex);

  const tensors = selectScopedTensors(headers, tensorPolicy);
  if (tensors.length !== tensorClosureReceipt.expectedTensorCount) {
    throw new Error('Scoped tensor count does not match tensor-role closure.');
  }
  const sourceDtypes = [...new Set(tensors.map((tensor) => tensor.dtype))].sort();
  if (sourceDtypes.length !== 1) {
    throw new Error(`Manifest conversion preflight requires one source dtype; got ${sourceDtypes.join(', ')}.`);
  }
  const sourceQuantization = normalizeQuantTag(sourceDtypes[0]);
  const plan = resolveConversionPlan({
    rawConfig,
    tensors,
    converterConfig: conversionConfig,
    sourceQuantization,
  });
  if (plan.executionVersion !== 'v1') throw new Error('Manifest conversion preflight requires execution v1.');
  if (plan.modelType !== conversionConfig.modelType) throw new Error('Conversion-plan model type drifted.');

  const commands = expandExecutionV1(plan.manifestInference.execution);
  const tensorRoles = countBy(tensors, (tensor) => resolveTensorRole(tensor));
  if (Object.hasOwn(tensorRoles, 'other')) {
    throw new Error('Manifest conversion preflight found unclassified source tensors.');
  }
  const core = {
    schema: 'doppler.manifest-conversion-preflight-receipt/v1',
    modelId: semanticReceipt.modelId,
    entryPointId: semanticReceipt.entryPointId,
    author: structuredClone(policy.author),
    semanticEvidence: {
      sourceModelIRHash: semanticReceipt.sourceModelIRHash,
      modelIRHash: semanticReceipt.modelIRHash,
      conversionConfigDigest: semanticReceipt.conversionConfigDigest,
    },
    tensorClosureEvidence: {
      receiptDigest: tensorClosureReceipt.receiptDigest,
      expectedTensorCount: tensorClosureReceipt.expectedTensorCount,
      observedTensorCount: tensorClosureReceipt.observedTensorCount,
    },
    sourceAcquisitionEvidence: {
      receiptDigest: acquisition.receiptDigest,
      fileCount: acquisition.fileCount,
      totalBytes: acquisition.totalBytes,
      weightShardCount: acquisition.files.filter((file) => file.role === 'weight-shard').length,
    },
    sourceEvidence: {
      checkpointId: headers.checkpointId,
      repository: headers.repository,
      revision: headers.revision,
      headerEvidenceDigest: digest(headers),
      sourceDtypes,
      sourceQuantization,
      scopedTensorCount: tensors.length,
      tensorDescriptorDigest: digest(tensors),
      tensorRoles,
      weightIndex: indexEvidence,
    },
    conversionPlan: {
      executionVersion: plan.executionVersion,
      modelType: plan.modelType,
      sourceQuantization: plan.sourceQuantization,
      manifestQuantization: plan.manifestQuantization,
      quantizationInfo: jsonClone(plan.quantizationInfo),
      inferenceDigest: digest(plan.manifestInference),
      executionDigest: digest(plan.manifestInference.execution),
      commandCount: commands.length,
      commandsByPhase: countBy(commands, (command) => command.phase),
      commandsBySection: countBy(commands, (command) => command.section),
      kernelClosure: [...new Set(commands.map((command) => `${command.kernel}#${command.entry}`))].sort(),
    },
    dispositions: {
      headerPreflightPassed: true,
      weightBodiesRequired: true,
      weightBodiesPresent: true,
      conversionExecuted: false,
      qualificationStarted: false,
      packEligible: false,
    },
  };
  return Object.freeze({ ...core, receiptDigest: digest(core) });
}
