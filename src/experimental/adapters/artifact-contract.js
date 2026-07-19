export const ADAPTER_ARTIFACT_SCHEMA = 'doppler.adapter-artifact/v1';
export const ADAPTER_ARTIFACT_LIFECYCLES = Object.freeze([
  'preserved',
  'candidate',
  'qualified',
  'promoted',
  'revoked',
]);
export const ADAPTER_ARTIFACT_ACCESS = Object.freeze(['public', 'gated', 'private']);
export const ADAPTER_ARTIFACT_ORIGIN_PROVIDERS = Object.freeze([
  'huggingface',
  'gcs',
  'https-preservation',
]);

const SHA256_PATTERN = /^sha256:[a-f0-9]{64}$/;
const HF_REVISION_PATTERN = /^[a-f0-9]{40,64}$/;
const GCS_GENERATION_PATTERN = /^[1-9][0-9]*$/;
const GCS_BUCKET_PATTERN = /^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$/;

const asText = (value) => String(value || '').trim();
const hasTraversal = (value) => asText(value).split('/').includes('..');

const requireText = (errors, value, field) => {
  if (!asText(value)) errors.push({ field, message: `${field} is required`, value });
};

const requireSha256 = (errors, value, field) => {
  if (!SHA256_PATTERN.test(asText(value))) {
    errors.push({ field, message: `${field} must be a sha256: content identity`, value });
  }
};

const validateArtifactPath = (errors, value, field) => {
  const path = asText(value);
  requireText(errors, path, field);
  if (path.startsWith('/') || hasTraversal(path)) {
    errors.push({ field, message: `${field} must be an artifact-relative path`, value });
  }
};

export function validateAdapterArtifactOrigin(origin, { allowPreservation = false } = {}) {
  const errors = [];
  if (!origin || typeof origin !== 'object' || Array.isArray(origin)) {
    return {
      valid: false,
      errors: [{ field: 'origin', message: 'origin must be an object', value: origin }],
    };
  }

  if (!ADAPTER_ARTIFACT_ORIGIN_PROVIDERS.includes(origin.provider)) {
    errors.push({ field: 'origin.provider', message: 'origin.provider is unsupported', value: origin.provider });
  } else if (origin.provider === 'huggingface') {
    requireText(errors, origin.repoId, 'origin.repoId');
    if (!HF_REVISION_PATTERN.test(asText(origin.revision))) {
      errors.push({
        field: 'origin.revision',
        message: 'origin.revision must be a full immutable Hugging Face commit hash',
        value: origin.revision,
      });
    }
    validateArtifactPath(errors, origin.path, 'origin.path');
  } else if (origin.provider === 'gcs') {
    if (!GCS_BUCKET_PATTERN.test(asText(origin.bucket))) {
      errors.push({ field: 'origin.bucket', message: 'origin.bucket is invalid', value: origin.bucket });
    }
    validateArtifactPath(errors, origin.object, 'origin.object');
    if (!GCS_GENERATION_PATTERN.test(asText(origin.generation))) {
      errors.push({
        field: 'origin.generation',
        message: 'origin.generation must pin an immutable GCS object generation',
        value: origin.generation,
      });
    }
  } else if (origin.provider === 'https-preservation') {
    if (!allowPreservation) {
      errors.push({
        field: 'origin.provider',
        message: 'https-preservation is allowed only as a preservation mirror',
        value: origin.provider,
      });
    }
    let url = null;
    try {
      url = new URL(asText(origin.url));
    } catch {
      errors.push({ field: 'origin.url', message: 'origin.url must be an absolute HTTPS URL', value: origin.url });
    }
    if (url && (url.protocol !== 'https:' || url.username || url.password || url.search)) {
      errors.push({
        field: 'origin.url',
        message: 'origin.url must be HTTPS, immutable, credential-free, and query-free',
        value: origin.url,
      });
    }
    if (!HF_REVISION_PATTERN.test(asText(origin.revision))) {
      errors.push({
        field: 'origin.revision',
        message: 'preservation origins must declare their full immutable revision',
        value: origin.revision,
      });
    }
  }

  return { valid: errors.length === 0, errors };
}

export function validateAdapterArtifactRecord(record) {
  const errors = [];
  if (!record || typeof record !== 'object' || Array.isArray(record)) {
    return {
      valid: false,
      errors: [{ field: 'record', message: 'adapter artifact record must be an object', value: record }],
    };
  }

  if (record.schema !== ADAPTER_ARTIFACT_SCHEMA) {
    errors.push({ field: 'schema', message: `schema must be ${ADAPTER_ARTIFACT_SCHEMA}`, value: record.schema });
  }
  requireText(errors, record.artifactId, 'artifactId');
  if (!ADAPTER_ARTIFACT_LIFECYCLES.includes(record.lifecycle)) {
    errors.push({ field: 'lifecycle', message: 'lifecycle is unsupported', value: record.lifecycle });
  }
  if (!ADAPTER_ARTIFACT_ACCESS.includes(record.access)) {
    errors.push({ field: 'access', message: 'access is unsupported', value: record.access });
  }

  const weights = record.weights || {};
  requireSha256(errors, weights.sha256, 'weights.sha256');
  if (!Number.isInteger(weights.bytes) || weights.bytes <= 0) {
    errors.push({ field: 'weights.bytes', message: 'weights.bytes must be a positive integer', value: weights.bytes });
  }
  if (weights.format !== 'safetensors') {
    errors.push({ field: 'weights.format', message: 'weights.format must be safetensors', value: weights.format });
  }

  const manifest = record.adapterManifest || {};
  requireText(errors, manifest.id, 'adapterManifest.id');
  requireText(errors, manifest.baseModel, 'adapterManifest.baseModel');
  if (!Number.isInteger(manifest.rank) || manifest.rank <= 0) {
    errors.push({ field: 'adapterManifest.rank', message: 'adapterManifest.rank must be a positive integer', value: manifest.rank });
  }
  if (!Number.isFinite(manifest.alpha) || manifest.alpha <= 0) {
    errors.push({ field: 'adapterManifest.alpha', message: 'adapterManifest.alpha must be positive', value: manifest.alpha });
  }
  if (!Array.isArray(manifest.targetModules) || manifest.targetModules.length === 0) {
    errors.push({ field: 'adapterManifest.targetModules', message: 'adapterManifest.targetModules is required', value: manifest.targetModules });
  }
  if (asText(manifest.checksumAlgorithm) !== 'sha256') {
    errors.push({ field: 'adapterManifest.checksumAlgorithm', message: 'adapterManifest.checksumAlgorithm must be sha256', value: manifest.checksumAlgorithm });
  }
  const manifestChecksum = asText(manifest.checksum).replace(/^sha256:/, '');
  if (`sha256:${manifestChecksum}` !== asText(weights.sha256)) {
    errors.push({ field: 'adapterManifest.checksum', message: 'adapter manifest checksum must match weights.sha256', value: manifest.checksum });
  }
  if (Number(manifest.weightsSize) !== weights.bytes) {
    errors.push({ field: 'adapterManifest.weightsSize', message: 'adapter manifest weightsSize must match weights.bytes', value: manifest.weightsSize });
  }

  const trainingBase = record.trainingBase || {};
  requireText(errors, trainingBase.repoId, 'trainingBase.repoId');
  if (!HF_REVISION_PATTERN.test(asText(trainingBase.revision))) {
    errors.push({ field: 'trainingBase.revision', message: 'trainingBase.revision must be a full immutable source revision', value: trainingBase.revision });
  }

  const runtimeBase = record.runtimeBase || {};
  requireText(errors, runtimeBase.modelId, 'runtimeBase.modelId');
  requireText(errors, runtimeBase.weightPackId, 'runtimeBase.weightPackId');
  requireText(errors, runtimeBase.manifestVariantId, 'runtimeBase.manifestVariantId');
  for (const field of [
    'modelSha256',
    'manifestSha256',
    'tokenizerSha256',
    'weightPackSha256',
    'conversionConfigSha256',
  ]) {
    requireSha256(errors, runtimeBase[field], `runtimeBase.${field}`);
  }
  if (manifest.baseModel && runtimeBase.modelId && manifest.baseModel !== runtimeBase.modelId) {
    errors.push({
      field: 'runtimeBase.modelId',
      message: 'runtimeBase.modelId must match adapterManifest.baseModel',
      value: runtimeBase.modelId,
    });
  }

  if (record.primaryOrigin == null) {
    if (record.lifecycle === 'qualified' || record.lifecycle === 'promoted') {
      errors.push({ field: 'primaryOrigin', message: `${record.lifecycle} artifacts require a primary Hugging Face or GCS origin`, value: record.primaryOrigin });
    }
  } else {
    const originValidation = validateAdapterArtifactOrigin(record.primaryOrigin);
    errors.push(...originValidation.errors.map((error) => ({ ...error, field: `primaryOrigin.${error.field.replace(/^origin\.?/, '')}` })));
  }

  if (!Array.isArray(record.preservationMirrors)) {
    errors.push({ field: 'preservationMirrors', message: 'preservationMirrors must be an array', value: record.preservationMirrors });
  } else {
    record.preservationMirrors.forEach((origin, index) => {
      const validation = validateAdapterArtifactOrigin(origin, { allowPreservation: true });
      errors.push(...validation.errors.map((error) => ({
        ...error,
        field: `preservationMirrors[${index}].${error.field.replace(/^origin\.?/, '')}`,
      })));
    });
  }

  if (!Array.isArray(record.evidence) || record.evidence.length === 0) {
    errors.push({ field: 'evidence', message: 'evidence must contain at least one receipt identity', value: record.evidence });
  } else {
    record.evidence.forEach((evidence, index) => {
      requireText(errors, evidence.kind, `evidence[${index}].kind`);
      validateArtifactPath(errors, evidence.path, `evidence[${index}].path`);
      requireSha256(errors, evidence.sha256, `evidence[${index}].sha256`);
    });
  }

  return { valid: errors.length === 0, errors };
}

export function assertAdapterArtifactRecord(record) {
  const validation = validateAdapterArtifactRecord(record);
  if (!validation.valid) {
    const message = validation.errors.map((error) => `${error.field}: ${error.message}`).join('; ');
    throw new Error(`Adapter artifact contract invalid: ${message}`);
  }
  return record;
}

export function buildImmutableArtifactUrl(origin) {
  const validation = validateAdapterArtifactOrigin(origin);
  if (!validation.valid) {
    const message = validation.errors.map((error) => `${error.field}: ${error.message}`).join('; ');
    throw new Error(`Adapter artifact origin invalid: ${message}`);
  }
  if (origin.provider === 'huggingface') {
    return `https://huggingface.co/${origin.repoId}/resolve/${origin.revision}/${origin.path}`;
  }
  return `https://storage.googleapis.com/${origin.bucket}/${origin.object}?generation=${origin.generation}`;
}

export function adapterArtifactCacheKey(record) {
  assertAdapterArtifactRecord(record);
  return record.weights.sha256;
}
