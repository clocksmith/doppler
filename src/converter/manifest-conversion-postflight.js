import { validateConversionReport } from '../config/schema/conversion-report.schema.js';
import { createConverterConfig } from '../config/schema/converter.schema.js';
import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';
import { sanitizeModelId } from './core.js';

export const MANIFEST_CONVERSION_POSTFLIGHT_SCHEMA_ID = 'doppler.manifest-conversion-postflight/v1';

const SHA256_DIGEST = /^sha256:[0-9a-f]{64}$/;

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function requireObject(value, label) {
  if (!isObject(value)) throw new Error(`${label} must be an object.`);
  return value;
}

function requireString(value, label) {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be a non-empty string.`);
  return value.trim();
}

function requireDigest(value, label) {
  const digest = requireString(value, label);
  if (!SHA256_DIGEST.test(digest)) throw new Error(`${label} must be a SHA-256 digest.`);
  return digest;
}

function digest(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

function requireEqual(observed, expected, label) {
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error(`${label} does not match converted evidence.`);
  }
}

function requireAuthor(author) {
  if (!isObject(author) || !['human', 'ai', 'tool'].includes(author.kind)
    || typeof author.actor !== 'string' || !author.actor.trim()) {
    throw new Error('Manifest conversion postflight requires attributable authorship.');
  }
}

function requirePassedArtifact(value, label) {
  if (!isObject(value) || value.ok !== true) {
    throw new Error(`Manifest conversion postflight requires a passing ${label}.`);
  }
}

function receiptCore(receipt) {
  const { receiptDigest: ignored, ...core } = receipt;
  void ignored;
  return core;
}

export function validateManifestConversionPostflightReceipt(receipt) {
  const errors = [];
  try {
    requireObject(receipt, 'Manifest conversion postflight receipt');
    if (receipt.schema !== 'doppler.manifest-conversion-postflight-receipt/v1') {
      throw new Error('Manifest conversion postflight receipt schema is invalid.');
    }
    requireString(receipt.modelId, 'receipt.modelId');
    requireString(receipt.entryPointId, 'receipt.entryPointId');
    requireAuthor(receipt.author);
    requireObject(receipt.preflightEvidence, 'receipt.preflightEvidence');
    requireDigest(receipt.preflightEvidence.receiptDigest, 'receipt.preflightEvidence.receiptDigest');
    requireDigest(receipt.preflightEvidence.conversionConfigDigest, 'receipt.preflightEvidence.conversionConfigDigest');
    requireObject(receipt.conversionEvidence, 'receipt.conversionEvidence');
    requireDigest(receipt.conversionEvidence.reportDigest, 'receipt.conversionEvidence.reportDigest');
    if (!Number.isFinite(Date.parse(receipt.conversionEvidence.startedAtUtc))
      || !Number.isFinite(Date.parse(receipt.conversionEvidence.completedAtUtc))
      || Date.parse(receipt.conversionEvidence.completedAtUtc) < Date.parse(receipt.conversionEvidence.startedAtUtc)
      || !Number.isFinite(receipt.conversionEvidence.durationMs)
      || receipt.conversionEvidence.durationMs < 0) {
      throw new Error('receipt.conversionEvidence physical timing is invalid.');
    }
    requireObject(receipt.manifestEvidence, 'receipt.manifestEvidence');
    for (const field of ['digest', 'inferenceDigest', 'executionDigest']) {
      requireDigest(receipt.manifestEvidence[field], `receipt.manifestEvidence.${field}`);
    }
    requireObject(receipt.manifestEvidence.artifactIdentity, 'receipt.manifestEvidence.artifactIdentity');
    requireObject(receipt.physicalClosure, 'receipt.physicalClosure');
    requireDigest(receipt.physicalClosure.digest, 'receipt.physicalClosure.digest');
    if (!Number.isInteger(receipt.physicalClosure.shardCount) || receipt.physicalClosure.shardCount < 1
      || !Number.isInteger(receipt.physicalClosure.shardBytes) || receipt.physicalClosure.shardBytes < 1
      || !Number.isInteger(receipt.physicalClosure.artifactCount) || receipt.physicalClosure.artifactCount < 1
      || !Array.isArray(receipt.physicalClosure.artifacts)
      || receipt.physicalClosure.artifacts.length !== receipt.physicalClosure.artifactCount) {
      throw new Error('receipt.physicalClosure counts are invalid.');
    }
    requireEqual(receipt.dispositions, {
      conversionExecuted: true,
      physicalShardClosureVerified: true,
      qualificationStarted: false,
      packEligible: false,
    }, 'Receipt dispositions');
    requireEqual(requireDigest(receipt.receiptDigest, 'receipt.receiptDigest'), digest(receiptCore(receipt)), 'Receipt digest');
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
  }
  return { ok: errors.length === 0, errors };
}

function normalizeManifestShard(shard) {
  const hash = requireString(shard?.hash, `manifest shard ${shard?.index} hash`).replace(/^sha256:/, '');
  if (!/^[0-9a-f]{64}$/.test(hash)) throw new Error(`Manifest shard ${shard?.index} hash is invalid.`);
  return {
    index: shard.index,
    filename: shard.filename,
    size: shard.size,
    hash,
    offset: shard.offset,
  };
}

function validateShardClosure(manifest, observations) {
  if (manifest.hashAlgorithm !== 'sha256') {
    throw new Error('Manifest conversion postflight currently requires hashAlgorithm "sha256".');
  }
  if (!Array.isArray(manifest.shards) || manifest.shards.length < 1) {
    throw new Error('Manifest conversion postflight requires manifest shards.');
  }
  if (!Array.isArray(observations) || observations.length !== manifest.shards.length) {
    throw new Error('Observed shard closure does not match manifest shard count.');
  }
  const shards = manifest.shards.map(normalizeManifestShard);
  let totalBytes = 0;
  for (const [index, shard] of shards.entries()) {
    const observed = requireObject(observations[index], `shardObservations[${index}]`);
    requireEqual(observed.index, shard.index, `Shard ${index} index`);
    requireEqual(observed.filename, shard.filename, `Shard ${index} filename`);
    requireEqual(observed.size, shard.size, `Shard ${index} size`);
    requireEqual(requireDigest(observed.digest, `shardObservations[${index}].digest`), `sha256:${shard.hash}`, `Shard ${index} digest`);
    totalBytes += shard.size;
  }
  requireEqual(totalBytes, manifest.totalSize, 'Physical shard bytes');
  return { shards, totalBytes };
}

function validateArtifactClosure(manifest, observations) {
  if (!Array.isArray(observations) || observations.length < 1) {
    throw new Error('Manifest conversion postflight requires non-shard artifact observations.');
  }
  const artifacts = observations.map((entry, index) => {
    requireObject(entry, `artifactObservations[${index}]`);
    if (!Number.isInteger(entry.size) || entry.size < 0) {
      throw new Error(`artifactObservations[${index}].size must be a non-negative integer.`);
    }
    return {
      role: requireString(entry.role, `artifactObservations[${index}].role`),
      path: requireString(entry.path, `artifactObservations[${index}].path`),
      size: entry.size,
      digest: requireDigest(entry.digest, `artifactObservations[${index}].digest`),
    };
  });
  const tokenizerPath = manifest.tokenizer?.file ?? manifest.tokenizer?.sentencepieceModel;
  if (typeof tokenizerPath === 'string' && tokenizerPath.trim()) {
    const tokenizer = artifacts.find((entry) => entry.role === 'tokenizer');
    if (!tokenizer || tokenizer.path !== tokenizerPath) {
      throw new Error('Tokenizer artifact observation does not match the manifest tokenizer path.');
    }
  }
  const paths = artifacts.map((entry) => entry.path);
  if (new Set(paths).size !== paths.length) throw new Error('Artifact observation paths must be unique.');
  return artifacts;
}

function validateArtifactIdentity(manifest, conversionConfig, preflight) {
  const identity = requireObject(manifest.artifactIdentity, 'manifest.artifactIdentity');
  const resolvedConfig = createConverterConfig(conversionConfig);
  const explicitIdentity = isObject(resolvedConfig.manifest.artifactIdentity)
    ? resolvedConfig.manifest.artifactIdentity
    : {};
  const source = preflight.sourceEvidence;
  requireEqual(identity.sourceCheckpointId, source.checkpointId, 'Source checkpoint identity');
  requireEqual(identity.sourceRepo, source.repository, 'Source repository identity');
  requireEqual(identity.sourceRevision, source.revision, 'Source revision identity');
  requireEqual(identity.sourceFormat, 'safetensors', 'Source format identity');
  requireEqual(identity.artifactCompleteness, 'complete', 'Artifact completeness');
  requireEqual(identity.conversionConfigDigest, digest(conversionConfig), 'Conversion config digest');
  requireEqual(preflight.semanticEvidence.conversionConfigDigest, digest(conversionConfig), 'Semantic conversion config digest');

  const shardSetHash = digest({
    hashAlgorithm: manifest.hashAlgorithm,
    shards: manifest.shards.map(normalizeManifestShard),
  });
  requireEqual(identity.shardSetHash, shardSetHash, 'Shard-set identity');
  const weightPackHash = digest({
    sourceCheckpointId: identity.sourceCheckpointId,
    sourceFormat: identity.sourceFormat,
    modelType: manifest.modelType,
    modalitySet: identity.modalitySet,
    quantizationInfo: manifest.quantizationInfo,
    materializationProfile: identity.materializationProfile,
    shardSetHash,
    sharding: { shardSizeBytes: resolvedConfig.sharding.shardSizeBytes },
    output: { textOnly: resolvedConfig.output.textOnly === true },
  });
  requireEqual(identity.weightPackHash, weightPackHash, 'Weight-pack identity');
  const modelIdPrefix = sanitizeModelId(manifest.modelId) ?? 'model';
  const weightPackId = explicitIdentity.weightPackId
    ?? `${modelIdPrefix}-wp-${weightPackHash.slice(7, 19)}`;
  requireEqual(identity.weightPackId, weightPackId, 'Weight-pack ID');
  const manifestVariantHash = digest({
    weightPackId: identity.weightPackId,
    modelType: manifest.modelType,
    inference: manifest.inference,
    config: resolvedConfig.manifest,
  });
  const manifestVariantId = explicitIdentity.manifestVariantId
    ?? `${modelIdPrefix}-mv-${manifestVariantHash.slice(7, 19)}`;
  requireEqual(identity.manifestVariantId, manifestVariantId, 'Manifest-variant ID');
  return structuredClone(identity);
}

export function createManifestConversionPostflightReceipt({
  conversionConfig,
  conversionReport,
  conversionReportDigest,
  manifest,
  manifestDigest,
  preflightReceipt,
  shardObservations,
  artifactObservations,
  policy,
}) {
  requireObject(policy, 'Manifest conversion postflight policy');
  if (policy.schema !== MANIFEST_CONVERSION_POSTFLIGHT_SCHEMA_ID) {
    throw new Error(`Manifest conversion postflight requires policy "${MANIFEST_CONVERSION_POSTFLIGHT_SCHEMA_ID}".`);
  }
  requireAuthor(policy.author);
  requireObject(conversionConfig, 'Conversion config');
  requireObject(manifest, 'Converted manifest');
  requireObject(preflightReceipt, 'Manifest conversion preflight receipt');
  if (preflightReceipt.schema !== 'doppler.manifest-conversion-preflight-receipt/v1'
    || preflightReceipt.dispositions?.headerPreflightPassed !== true
    || preflightReceipt.dispositions?.conversionExecuted !== false) {
    throw new Error('Manifest conversion postflight requires a passing, unexecuted preflight receipt.');
  }
  validateConversionReport(conversionReport);
  if (conversionReport.startedAtUtc === undefined || conversionReport.completedAtUtc === undefined
    || conversionReport.durationMs === undefined) {
    throw new Error('Manifest conversion postflight requires measured physical conversion timing.');
  }
  requirePassedArtifact(conversionReport.executionContractArtifact, 'execution contract artifact');
  requirePassedArtifact(conversionReport.layerPatternContractArtifact, 'layer-pattern contract artifact');
  requirePassedArtifact(conversionReport.requiredInferenceFieldsArtifact, 'required-inference-fields artifact');

  requireEqual(manifest.modelId, preflightReceipt.modelId, 'Manifest model ID');
  requireEqual(conversionReport.modelId, manifest.modelId, 'Conversion report model ID');
  requireEqual(conversionReport.result.modelType, manifest.modelType, 'Conversion report model type');
  requireEqual(conversionReport.result.shardCount, manifest.shards?.length, 'Conversion report shard count');
  requireEqual(conversionReport.result.tensorCount, Object.keys(manifest.tensors ?? {}).length, 'Conversion report tensor count');
  requireEqual(conversionReport.result.totalSize, manifest.totalSize, 'Conversion report total size');
  requireEqual(conversionReport.timestamp, manifest.metadata?.convertedAt, 'Conversion identity timestamp');
  requireEqual(digest(manifest.inference), preflightReceipt.conversionPlan.inferenceDigest, 'Manifest inference digest');
  requireEqual(digest(manifest.inference?.execution), preflightReceipt.conversionPlan.executionDigest, 'Manifest execution digest');
  requireEqual(Object.keys(manifest.tensors ?? {}).length, preflightReceipt.tensorClosureEvidence.expectedTensorCount, 'Converted tensor count');

  const identity = validateArtifactIdentity(manifest, conversionConfig, preflightReceipt);
  const shardClosure = validateShardClosure(manifest, shardObservations);
  const artifacts = validateArtifactClosure(manifest, artifactObservations);
  const normalizedManifestDigest = requireDigest(manifestDigest, 'manifestDigest');
  const normalizedReportDigest = requireDigest(conversionReportDigest, 'conversionReportDigest');
  const physicalClosure = {
    manifestDigest: normalizedManifestDigest,
    conversionReportDigest: normalizedReportDigest,
    shards: shardObservations.map((entry) => structuredClone(entry)),
    artifacts,
  };
  const core = {
    schema: 'doppler.manifest-conversion-postflight-receipt/v1',
    modelId: manifest.modelId,
    entryPointId: preflightReceipt.entryPointId,
    author: structuredClone(policy.author),
    preflightEvidence: {
      receiptDigest: preflightReceipt.receiptDigest,
      conversionConfigDigest: preflightReceipt.semanticEvidence.conversionConfigDigest,
      expectedTensorCount: preflightReceipt.tensorClosureEvidence.expectedTensorCount,
    },
    conversionEvidence: {
      reportDigest: normalizedReportDigest,
      startedAtUtc: conversionReport.startedAtUtc,
      completedAtUtc: conversionReport.completedAtUtc,
      durationMs: conversionReport.durationMs,
      modelType: manifest.modelType,
      tensorCount: conversionReport.result.tensorCount,
      shardCount: conversionReport.result.shardCount,
      totalBytes: conversionReport.result.totalSize,
    },
    manifestEvidence: {
      digest: normalizedManifestDigest,
      inferenceDigest: preflightReceipt.conversionPlan.inferenceDigest,
      executionDigest: preflightReceipt.conversionPlan.executionDigest,
      artifactIdentity: identity,
    },
    physicalClosure: {
      digest: digest(physicalClosure),
      shardCount: shardClosure.shards.length,
      shardBytes: shardClosure.totalBytes,
      artifactCount: artifacts.length,
      artifacts,
    },
    dispositions: {
      conversionExecuted: true,
      physicalShardClosureVerified: true,
      qualificationStarted: false,
      packEligible: false,
    },
  };
  const receipt = { ...core, receiptDigest: digest(core) };
  const validation = validateManifestConversionPostflightReceipt(receipt);
  if (!validation.ok) throw new Error(`Invalid manifest conversion postflight receipt: ${validation.errors.join('; ')}`);
  return Object.freeze(receipt);
}
