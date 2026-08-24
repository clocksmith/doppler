import { loadJson } from '../formats/load-json.js';

const TARGET_FIELDS = [
  'logicalModelIds',
  'modelIds',
  'sourceCheckpointIds',
  'weightPackIds',
  'manifestVariantIds',
  'artifactVariantIds',
  'adapterIds',
  'adapterDigests',
];
const IDENTITY_FIELDS = {
  logicalModelIds: 'logicalModelId',
  modelIds: 'modelId',
  sourceCheckpointIds: 'sourceCheckpointId',
  weightPackIds: 'weightPackId',
  manifestVariantIds: 'manifestVariantId',
  artifactVariantIds: 'artifactVariantId',
  adapterIds: 'adapterId',
  adapterDigests: 'adapterDigest',
};
const SEVERITIES = new Set(['correctness', 'security', 'reliability', 'provenance', 'policy']);
const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const HASH_PATTERN = /^sha256:[0-9a-f]{64}$/;
const REGISTRY = 'revocation registry';

let registryPromise = null;
let bundledRegistryValue = null;
let liveRegistryValue = null;
let assertLiveRegistryCurrent = null;
let policyRevision = 0;
const authorizedAdapters = new WeakMap();

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function deepFreeze(value) {
  for (const child of Object.values(value)) {
    if (child && typeof child === 'object') deepFreeze(child);
  }
  return Object.freeze(value);
}

function assertExactKeys(value, fields, label) {
  if (!isPlainObject(value)) {
    throw new Error(`${label} must be an object.`);
  }
  const expected = new Set(fields);
  for (const field of Object.keys(value)) {
    if (!expected.has(field)) throw new Error(`${label}.${field} is not supported.`);
  }
  for (const field of fields) {
    if (!Object.prototype.hasOwnProperty.call(value, field)) {
      throw new Error(`${label}.${field} is required.`);
    }
  }
}

function normalizeText(value, label) {
  const normalized = typeof value === 'string' ? value.trim() : '';
  if (!normalized) throw new Error(`${label} must be non-empty.`);
  return normalized;
}

function normalizeInstant(value, label) {
  const normalized = normalizeText(value, label);
  const instant = new Date(normalized);
  if (!Number.isFinite(instant.getTime()) || instant.toISOString() !== normalized) {
    throw new Error(`${label} must be an ISO instant.`);
  }
  return normalized;
}

function normalizeTargetValue(value, field, label) {
  const normalized = normalizeText(value, label);
  if (field !== 'artifactVariantIds' && field !== 'adapterDigests') return normalized;
  const digest = normalized.toLowerCase();
  if (!HASH_PATTERN.test(digest)) {
    throw new Error(`${label} must be sha256:.`);
  }
  return digest;
}

function normalizeTargets(value, label, requireTarget) {
  assertExactKeys(value, TARGET_FIELDS, label);
  const normalized = {};
  let targetCount = 0;
  for (const field of TARGET_FIELDS) {
    if (!Array.isArray(value[field])) throw new Error(`${label}.${field} must be an array.`);
    const entries = value[field].map((entry, index) => (
      normalizeTargetValue(entry, field, `${label}.${field}[${index}]`)
    ));
    if (new Set(entries).size !== entries.length) {
      throw new Error(`${label}.${field} has duplicates.`);
    }
    targetCount += entries.length;
    normalized[field] = entries;
  }
  if (requireTarget && targetCount === 0) {
    throw new Error(`${label} requires a revoked target.`);
  }
  return normalized;
}

function normalizeRevocation(value, index) {
  const label = `${REGISTRY} revocations[${index}]`;
  assertExactKeys(value, [
    'id',
    'state',
    'issuedAtUtc',
    'severity',
    'reason',
    'targets',
    'replacements',
    'evidencePaths',
  ], label);
  const id = normalizeText(value.id, `${label}.id`);
  if (!ID_PATTERN.test(id)) throw new Error(`${label}.id must be kebab-case.`);
  if (value.state !== 'revoked') throw new Error(`${label}.state must be "revoked".`);
  if (!SEVERITIES.has(value.severity)) throw new Error(`${label}.severity is not recognized.`);
  if (!Array.isArray(value.evidencePaths) || value.evidencePaths.length === 0) {
    throw new Error(`${label}.evidencePaths must be a non-empty array.`);
  }
  const evidencePaths = value.evidencePaths.map((entry, evidenceIndex) => (
    normalizeText(entry, `${label}.evidencePaths[${evidenceIndex}]`)
  ));
  if (new Set(evidencePaths).size !== evidencePaths.length) {
    throw new Error(`${label}.evidencePaths has duplicates.`);
  }
  return {
    id,
    state: 'revoked',
    issuedAtUtc: normalizeInstant(value.issuedAtUtc, `${label}.issuedAtUtc`),
    severity: value.severity,
    reason: normalizeText(value.reason, `${label}.reason`),
    targets: normalizeTargets(value.targets, `${label}.targets`, true),
    replacements: normalizeTargets(value.replacements, `${label}.replacements`, false),
    evidencePaths,
  };
}

class DopplerRevocationError extends Error {
  constructor(revocation, matches) {
    const hasReplacement = TARGET_FIELDS.some((field) => revocation.replacements[field].length > 0);
    super(
      `Revocation ${revocation.id} rejected ${matches.join(', ')}: ` +
      `${revocation.reason} No auto replacement` +
      `${hasReplacement ? '; inspect registry.' : '.'}`
    );
    this.name = 'DopplerRevocationError';
    this.code = 'DOPPLER_REVOKED';
    this.revocationId = revocation.id;
    this.severity = revocation.severity;
    this.matches = matches;
    this.replacements = revocation.replacements;
  }
}

export function validateRevocationRegistry(value) {
  assertExactKeys(value, [
    '$schema',
    'schemaVersion',
    'source',
    'updatedAtUtc',
    'trust',
    'revocations',
  ], REGISTRY);
  if (value.$schema !== 'schema/revocation-registry.schema.json') {
    throw new Error(`${REGISTRY} $schema is not recognized.`);
  }
  if (value.schemaVersion !== 1) throw new Error(`${REGISTRY} schemaVersion must be 1.`);
  if (value.source !== 'doppler') throw new Error(`${REGISTRY} source must be "doppler".`);
  assertExactKeys(value.trust, ['distribution', 'signatureVerification'], `${REGISTRY} trust`);
  const bundledTrust = value.trust.distribution === 'bundled-package'
    && value.trust.signatureVerification === 'unavailable';
  const liveTrust = value.trust.distribution === 'signed-live'
    && value.trust.signatureVerification === 'verified';
  if (!bundledTrust && !liveTrust) throw new Error(`${REGISTRY} trust pair is not recognized.`);
  if (!Array.isArray(value.revocations)) throw new Error(`${REGISTRY} revocations must be an array.`);
  const revocations = value.revocations.map(normalizeRevocation);
  const ids = revocations.map((entry) => entry.id);
  if (new Set(ids).size !== ids.length) throw new Error(`${REGISTRY} ids must be unique.`);
  return deepFreeze({
    $schema: value.$schema,
    schemaVersion: 1,
    source: 'doppler',
    updatedAtUtc: normalizeInstant(value.updatedAtUtc, `${REGISTRY} updatedAtUtc`),
    trust: { ...value.trust },
    revocations,
  });
}

export async function loadRevocationRegistry() {
  registryPromise ??= loadJson(
    './revocation-registry.json',
    import.meta.url,
    'Cannot load Doppler revocations'
  ).then(validateRevocationRegistry).then((registry) => {
    if (registry.trust.distribution !== 'bundled-package') {
      throw new Error('The package revocation registry must use bundled-package trust.');
    }
    bundledRegistryValue = registry;
    return registry;
  });
  return registryPromise;
}

export function findResolutionRevocation(identity, registry) {
  if (!isPlainObject(identity)) throw new Error('Revocation identity must be object.');
  for (const revocation of registry.revocations) {
    const matchedFields = [];
    for (const targetField of TARGET_FIELDS) {
      const identityField = IDENTITY_FIELDS[targetField];
      const rawValue = identity[identityField];
      if (rawValue == null || rawValue === '') continue;
      const value = normalizeTargetValue(rawValue, targetField, `revocation identity.${identityField}`);
      if (revocation.targets[targetField].includes(value)) matchedFields.push(`${identityField}=${value}`);
    }
    if (matchedFields.length > 0) {
      return { revocation, matchedFields };
    }
  }
  return null;
}

export function assertResolutionNotRevoked(identity, registry) {
  const match = findResolutionRevocation(identity, registry);
  if (match) throw new DopplerRevocationError(match.revocation, match.matchedFields);
}

export async function assertBundledResolutionNotRevoked(identity) {
  await loadRevocationRegistry();
  assertKnownResolutionNotRevoked(identity);
}

export function installLiveRevocationRegistry(value, assertCurrent) {
  const registry = validateRevocationRegistry(value);
  if (registry.trust.distribution !== 'signed-live') {
    throw new Error('Live revocation installation requires verified signed-live trust.');
  }
  if (typeof assertCurrent !== 'function') throw new Error('Live revocation installation requires a freshness assertion.');
  liveRegistryValue = registry;
  assertLiveRegistryCurrent = assertCurrent;
  policyRevision += 1;
  return registry;
}

export function assertKnownResolutionNotRevoked(identity) {
  if (identity == null) return;
  if (!bundledRegistryValue && !liveRegistryValue) {
    throw new Error('Revocation policy is not initialized.');
  }
  if (bundledRegistryValue) assertResolutionNotRevoked(identity, bundledRegistryValue);
  if (liveRegistryValue) {
    assertLiveRegistryCurrent();
    assertResolutionNotRevoked(identity, liveRegistryValue);
  }
}

export async function authorizeBundledAdapter(adapter) {
  if (!isPlainObject(adapter)) throw new Error('Adapter authorization requires an object.');
  const adapterId = normalizeText(adapter.id, 'adapter.id');
  const identity = adapter.identity;
  if (identity?.schema !== 'doppler.lora-execution-identity/v1' || identity.id !== adapterId) {
    throw new Error('Adapter authorization requires matching manifest and execution IDs.');
  }
  await assertBundledResolutionNotRevoked({ adapterId, adapterDigest: identity.digest });
  authorizedAdapters.set(adapter, { adapterId, adapterDigest: identity.digest, revision: policyRevision });
  return adapter;
}

export function assertBundledAdapterAuthorized(adapter) {
  if (adapter == null) return;
  const authorization = authorizedAdapters.get(adapter);
  if (!authorization
    || adapter.id !== authorization.adapterId
    || adapter.identity?.digest !== authorization.adapterDigest) {
    throw new Error('LoRA activation requires a revocation-checked Doppler adapter.');
  }
  if (liveRegistryValue) assertLiveRegistryCurrent();
  if (authorization.revision !== policyRevision) {
    assertKnownResolutionNotRevoked({
      adapterId: authorization.adapterId,
      adapterDigest: authorization.adapterDigest,
    });
    authorization.revision = policyRevision;
  }
}
