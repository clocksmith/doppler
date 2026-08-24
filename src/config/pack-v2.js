
import { hashModelIR, validateModelIR } from './model-ir.js';
import { validatePackReleaseContract } from './pack-release-contract.js';
import { hashTargetPlan, validateTargetPlan } from './target-plan.js';
import { sha256Hex } from '../formats/sha256.js';
import { stableSortObject } from '../formats/stable-sort-object.js';

export const PACK_V2_SCHEMA_ID = 'doppler.pack/v2';
export const PACK_V2_SCHEMA_VERSION = 2;
export const PACK_V2_PROGRAM_SCHEMA_ID = 'doppler.pack-program/v1';
export const PACK_V2_SIGNATURE_ALGORITHM = 'Ed25519';

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const ARTIFACT_ROLES = new Set([
  'manifest',
  'weight-shard',
  'tokenizer',
  'conversion-config',
  'runtime-config',
  'reference-report',
  'qualification-evidence',
  'source-truth-evidence',
  'program-bundle',
  'host-source',
  'wgsl-source',
]);

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function canonicalJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function requireString(value, label, errors) {
  if (typeof value !== 'string' || !value.trim()) errors.push(`${label} must be a non-empty string.`);
}

function requireDigest(value, label, errors) {
  if (!SHA256_PATTERN.test(value || '')) errors.push(`${label} must be a SHA-256 digest.`);
}

function requireExactKeys(value, allowed, label, errors) {
  if (!isObject(value)) return;
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) errors.push(`${label}.${key} is not allowed.`);
  }
}

function requireInstant(value, label, errors) {
  requireString(value, label, errors);
  if (typeof value !== 'string') return;
  const time = new Date(value).getTime();
  if (!Number.isFinite(time) || new Date(time).toISOString() !== value) {
    errors.push(`${label} must be an ISO instant.`);
  }
}

function bytesToHex(bytes) {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

function hexToBytes(value) {
  if (typeof value !== 'string' || !/^[0-9a-f]+$/i.test(value) || value.length % 2 !== 0) {
    throw new Error('Pack signature must be an even-length hexadecimal string.');
  }
  return Uint8Array.from(value.match(/.{2}/g), (entry) => Number.parseInt(entry, 16));
}

function requireCrypto() {
  if (!globalThis.crypto?.subtle) {
    throw new Error('Doppler Pack signature verification requires WebCrypto subtle crypto.');
  }
  return globalThis.crypto.subtle;
}

export function getPackV2SemanticPayload(pack) {
  return {
    schema: pack.schema,
    schemaVersion: pack.schemaVersion,
    modelId: pack.modelId,
    createdAtUtc: pack.createdAtUtc,
    modelIR: pack.modelIR,
    targetPlans: pack.targetPlans,
    wgslModules: pack.wgslModules,
    artifacts: pack.artifacts,
    program: pack.program,
    release: pack.release,
  };
}

export function hashPackV2(pack) {
  return `sha256:${sha256Hex(canonicalJson(getPackV2SemanticPayload(pack)))}`;
}

export function hashPackV2Envelope(pack) {
  return `sha256:${sha256Hex(canonicalJson(pack))}`;
}

export function hashPackV2PublicKey(publicKeyJwk) {
  if (!isObject(publicKeyJwk)) throw new Error('Pack public key must be a JWK object.');
  return `sha256:${sha256Hex(canonicalJson(publicKeyJwk))}`;
}

function validateArtifacts(pack, errors) {
  if (!Array.isArray(pack.artifacts) || pack.artifacts.length === 0) {
    errors.push('artifacts must be a non-empty array.');
    return new Map();
  }
  const artifacts = new Map();
  const paths = new Set();
  for (const [index, artifact] of pack.artifacts.entries()) {
    if (!isObject(artifact)) {
      errors.push(`artifacts[${index}] must be an object.`);
      continue;
    }
    requireExactKeys(
      artifact,
      new Set(['artifactId', 'role', 'path', 'hash', 'sizeBytes']),
      `artifacts[${index}]`,
      errors
    );
    requireString(artifact.artifactId, `artifacts[${index}].artifactId`, errors);
    requireString(artifact.path, `artifacts[${index}].path`, errors);
    requireDigest(artifact.hash, `artifacts[${index}].hash`, errors);
    if (!ARTIFACT_ROLES.has(artifact.role)) errors.push(`artifacts[${index}].role is unsupported.`);
    if (!Number.isInteger(artifact.sizeBytes) || artifact.sizeBytes < 0) {
      errors.push(`artifacts[${index}].sizeBytes must be a non-negative integer.`);
    }
    if (artifacts.has(artifact.artifactId)) errors.push(`duplicate artifactId "${artifact.artifactId}".`);
    if (paths.has(artifact.path)) errors.push(`duplicate artifact path "${artifact.path}".`);
    artifacts.set(artifact.artifactId, artifact);
    paths.add(artifact.path);
  }
  return artifacts;
}

function validateModules(pack, artifacts, errors) {
  if (!Array.isArray(pack.wgslModules) || pack.wgslModules.length === 0) {
    errors.push('wgslModules must be a non-empty array.');
    return new Map();
  }
  const modules = new Map();
  for (const [index, module] of pack.wgslModules.entries()) {
    if (!isObject(module)) {
      errors.push(`wgslModules[${index}] must be an object.`);
      continue;
    }
    requireExactKeys(
      module,
      new Set(['id', 'file', 'entry', 'digest', 'sourceHash', 'sourceArtifactId', 'metadata']),
      `wgslModules[${index}]`,
      errors
    );
    requireString(module.id, `wgslModules[${index}].id`, errors);
    requireString(module.file, `wgslModules[${index}].file`, errors);
    requireString(module.entry, `wgslModules[${index}].entry`, errors);
    requireString(module.sourceArtifactId, `wgslModules[${index}].sourceArtifactId`, errors);
    requireDigest(module.digest, `wgslModules[${index}].digest`, errors);
    requireDigest(module.sourceHash, `wgslModules[${index}].sourceHash`, errors);
    if (modules.has(module.id)) errors.push(`duplicate WGSL module id "${module.id}".`);
    modules.set(module.id, module);
    const sourceArtifact = artifacts.get(module.sourceArtifactId);
    if (!sourceArtifact || sourceArtifact.role !== 'wgsl-source') {
      errors.push(`wgslModules[${index}] must reference a wgsl-source artifact.`);
    } else if (sourceArtifact.hash !== module.sourceHash) {
      errors.push(`wgslModules[${index}].sourceHash does not match its source artifact.`);
    }
  }
  return modules;
}

function validateProgram(pack, artifacts, errors) {
  const program = pack.program;
  if (!isObject(program)) {
    errors.push('program must be an object.');
    return;
  }
  requireExactKeys(program, new Set([
    'schema', 'programBundleHash', 'programBundleArtifactId', 'executionGraphHash',
    'manifestArtifactId', 'modelIREvidenceArtifactId', 'tokenizerArtifactIds',
    'weightArtifactIds', 'execution', 'referenceTranscript',
  ]), 'program', errors);
  if (program.schema !== PACK_V2_PROGRAM_SCHEMA_ID) {
    errors.push(`program.schema must be "${PACK_V2_PROGRAM_SCHEMA_ID}".`);
  }
  requireDigest(program.programBundleHash, 'program.programBundleHash', errors);
  requireDigest(program.executionGraphHash, 'program.executionGraphHash', errors);
  requireString(program.programBundleArtifactId, 'program.programBundleArtifactId', errors);
  requireString(program.manifestArtifactId, 'program.manifestArtifactId', errors);
  const programBundleArtifact = artifacts.get(program.programBundleArtifactId);
  if (programBundleArtifact?.role !== 'program-bundle') {
    errors.push('program.programBundleArtifactId must reference the Program Bundle artifact.');
  } else if (programBundleArtifact.hash !== program.programBundleHash) {
    errors.push('program.programBundleHash must equal the Program Bundle artifact hash.');
  }
  if (artifacts.get(program.manifestArtifactId)?.role !== 'manifest') {
    errors.push('program.manifestArtifactId must reference the manifest artifact.');
  }
  if (program.modelIREvidenceArtifactId !== undefined
    && artifacts.get(program.modelIREvidenceArtifactId)?.role !== 'source-truth-evidence') {
    errors.push('program.modelIREvidenceArtifactId must reference source-truth evidence.');
  }
  for (const [field, role] of [['tokenizerArtifactIds', 'tokenizer'], ['weightArtifactIds', 'weight-shard']]) {
    if (!Array.isArray(program[field]) || program[field].length === 0) {
      errors.push(`program.${field} must be a non-empty array.`);
      continue;
    }
    for (const artifactId of program[field]) {
      if (artifacts.get(artifactId)?.role !== role) errors.push(`program.${field} contains a non-${role} artifact.`);
    }
  }
  if (!isObject(program.execution) || !Array.isArray(program.execution.steps) || program.execution.steps.length === 0) {
    errors.push('program.execution must contain the expanded execution steps.');
  }
}

export function validatePackV2(pack, options = {}) {
  const errors = [];
  if (!isObject(pack)) return { ok: false, errors: ['Doppler Pack v2 must be a non-null object.'] };
  requireExactKeys(pack, new Set([
    'schema', 'schemaVersion', 'packId', 'modelId', 'createdAtUtc', 'semanticRoot',
    'modelIR', 'targetPlans', 'wgslModules', 'artifacts', 'program', 'release', 'signature',
  ]), 'pack', errors);
  if (pack.schema !== PACK_V2_SCHEMA_ID) errors.push(`schema must be "${PACK_V2_SCHEMA_ID}".`);
  if (pack.schemaVersion !== PACK_V2_SCHEMA_VERSION) errors.push(`schemaVersion must be ${PACK_V2_SCHEMA_VERSION}.`);
  requireString(pack.packId, 'packId', errors);
  requireString(pack.modelId, 'modelId', errors);
  requireInstant(pack.createdAtUtc, 'createdAtUtc', errors);
  requireDigest(pack.semanticRoot, 'semanticRoot', errors);

  const modelValidation = validateModelIR(pack.modelIR);
  if (!modelValidation.ok) errors.push(...modelValidation.errors.map((error) => `modelIR: ${error}`));
  if (pack.modelIR?.modelId !== pack.modelId) errors.push('pack.modelId must equal modelIR.modelId.');

  const artifacts = validateArtifacts(pack, errors);
  const modules = validateModules(pack, artifacts, errors);
  validateProgram(pack, artifacts, errors);

  const releaseValidation = validatePackReleaseContract(pack.release, {
    targetIds: Array.isArray(pack.targetPlans)
      ? pack.targetPlans.map((plan) => plan?.targetId).filter(Boolean)
      : [],
  });
  if (!releaseValidation.ok) {
    errors.push(...releaseValidation.errors);
  }
  requireExactKeys(pack.signature, new Set([
    'authority', 'algorithm', 'publicKeyDigest', 'signatureHex', 'signedDigest',
  ]), 'signature', errors);

  if (!Array.isArray(pack.targetPlans) || pack.targetPlans.length === 0) {
    errors.push('targetPlans must be a non-empty array.');
  } else {
    const targetIds = new Set();
    const modelIRHash = modelValidation.ok ? hashModelIR(pack.modelIR) : null;
    for (const [index, plan] of pack.targetPlans.entries()) {
      const validation = validateTargetPlan(plan);
      if (!validation.ok) errors.push(...validation.errors.map((error) => `targetPlans[${index}]: ${error}`));
      if (plan?.modelId !== pack.modelId) errors.push(`targetPlans[${index}].modelId must equal pack.modelId.`);
      if (modelIRHash && plan?.modelIRHash !== modelIRHash) errors.push(`targetPlans[${index}] does not bind the Pack ModelIR digest.`);
      if (plan?.programBundleHash !== pack.program?.programBundleHash) errors.push(`targetPlans[${index}] does not bind the Pack Program Bundle digest.`);
      if (plan?.executionGraphHash !== pack.program?.executionGraphHash) errors.push(`targetPlans[${index}] does not bind the Pack execution graph digest.`);
      if (targetIds.has(plan?.targetId)) errors.push(`duplicate targetId "${plan?.targetId}".`);
      targetIds.add(plan?.targetId);
      for (const kernel of plan?.kernelClosure || []) {
        const module = modules.get(kernel.moduleId);
        if (!module || module.digest !== kernel.digest || module.sourceHash !== kernel.sourceHash) {
          errors.push(`targetPlans[${index}] kernel "${kernel.moduleId}" is outside the Pack WGSL closure.`);
        }
      }
      for (const record of plan?.qualification || []) {
        const evidenceArtifact = artifacts.get(record.evidenceArtifactId);
        if (evidenceArtifact?.role !== 'qualification-evidence'
          && evidenceArtifact?.role !== 'reference-report') {
          errors.push(`targetPlans[${index}] qualification evidence is not packaged.`);
        } else if (record.evidenceHash !== evidenceArtifact.hash) {
          errors.push(`targetPlans[${index}] qualification evidence hash does not match its packaged artifact.`);
        }
      }
    }
  }

  const computedRoot = hashPackV2(pack);
  if (SHA256_PATTERN.test(pack.semanticRoot || '') && pack.semanticRoot !== computedRoot) {
    errors.push(`semanticRoot mismatch: expected ${computedRoot}, received ${pack.semanticRoot}.`);
  }
  const expectedPackId = SHA256_PATTERN.test(computedRoot)
    ? `${pack.modelId}-pack-v2-${computedRoot.slice('sha256:'.length, 'sha256:'.length + 16)}`
    : null;
  if (expectedPackId && pack.packId !== expectedPackId) errors.push(`packId must be derived from semanticRoot (${expectedPackId}).`);

  const requireSignature = options.requireSignature !== false;
  if (!isObject(pack.signature)) {
    if (requireSignature) errors.push('signature is required.');
  } else {
    requireString(pack.signature.authority, 'signature.authority', errors);
    if (pack.signature.algorithm !== PACK_V2_SIGNATURE_ALGORITHM) {
      errors.push(`signature.algorithm must be "${PACK_V2_SIGNATURE_ALGORITHM}".`);
    }
    requireDigest(pack.signature.publicKeyDigest, 'signature.publicKeyDigest', errors);
    requireDigest(pack.signature.signedDigest, 'signature.signedDigest', errors);
    if (!/^[0-9a-f]{128}$/.test(pack.signature.signatureHex || '')) {
      errors.push('signature.signatureHex must be a 64-byte hexadecimal Ed25519 signature.');
    }
    if (pack.signature.signedDigest !== pack.semanticRoot) errors.push('signature.signedDigest must equal semanticRoot.');
  }
  return { ok: errors.length === 0, errors };
}

export function freezePackV2(value) {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;
  for (const nested of Object.values(value)) freezePackV2(nested);
  return Object.freeze(value);
}

export function buildPackV2(params) {
  if (!isObject(params)) throw new Error('buildPackV2 requires an object.');
  const draft = {
    schema: PACK_V2_SCHEMA_ID,
    schemaVersion: PACK_V2_SCHEMA_VERSION,
    packId: '',
    modelId: params.modelId,
    createdAtUtc: params.createdAtUtc,
    semanticRoot: '',
    modelIR: params.modelIR,
    targetPlans: params.targetPlans,
    wgslModules: params.wgslModules,
    artifacts: params.artifacts,
    program: params.program,
    release: params.release,
    signature: null,
  };
  const semanticRoot = hashPackV2(draft);
  const pack = {
    ...draft,
    packId: `${draft.modelId}-pack-v2-${semanticRoot.slice('sha256:'.length, 'sha256:'.length + 16)}`,
    semanticRoot,
  };
  const validation = validatePackV2(pack, { requireSignature: false });
  if (!validation.ok) throw new Error(`Failed to build valid Doppler Pack v2: ${validation.errors.join('; ')}`);
  return pack;
}

export async function signPackV2(pack, signer) {
  const validation = validatePackV2(pack, { requireSignature: false });
  if (!validation.ok) throw new Error(`Cannot sign invalid Doppler Pack v2: ${validation.errors.join('; ')}`);
  if (typeof signer?.authority !== 'string' || !signer.authority.trim()) {
    throw new Error('signPackV2 requires signer.authority.');
  }
  if (!isObject(signer?.privateKeyJwk) || !isObject(signer?.publicKeyJwk)) {
    throw new Error('signPackV2 requires privateKeyJwk and publicKeyJwk.');
  }
  const subtle = requireCrypto();
  const privateKey = await subtle.importKey('jwk', signer.privateKeyJwk, { name: PACK_V2_SIGNATURE_ALGORITHM }, false, ['sign']);
  const payload = new TextEncoder().encode(pack.semanticRoot);
  const signatureBytes = new Uint8Array(await subtle.sign(PACK_V2_SIGNATURE_ALGORITHM, privateKey, payload));
  const signed = {
    ...pack,
    signature: {
      authority: signer.authority,
      algorithm: PACK_V2_SIGNATURE_ALGORITHM,
      publicKeyDigest: hashPackV2PublicKey(signer.publicKeyJwk),
      signatureHex: bytesToHex(signatureBytes),
      signedDigest: pack.semanticRoot,
    },
  };
  const signedValidation = validatePackV2(signed);
  if (!signedValidation.ok) throw new Error(`Signed Doppler Pack v2 is invalid: ${signedValidation.errors.join('; ')}`);
  return signed;
}

function resolveTrustedPublicKey(trustedSigners, authority) {
  if (trustedSigners instanceof Map) return trustedSigners.get(authority) ?? null;
  if (isObject(trustedSigners)) return trustedSigners[authority] ?? null;
  return null;
}

export async function verifyPackV2Signature(pack, trustedSigners) {
  const validation = validatePackV2(pack);
  if (!validation.ok) throw new Error(`Invalid Doppler Pack v2: ${validation.errors.join('; ')}`);
  const publicKeyJwk = resolveTrustedPublicKey(trustedSigners, pack.signature.authority);
  if (!publicKeyJwk) throw new Error(`Untrusted Doppler Pack signing authority "${pack.signature.authority}".`);
  const publicKeyDigest = hashPackV2PublicKey(publicKeyJwk);
  if (publicKeyDigest !== pack.signature.publicKeyDigest) {
    throw new Error(`Doppler Pack public key digest mismatch for authority "${pack.signature.authority}".`);
  }
  const subtle = requireCrypto();
  const publicKey = await subtle.importKey('jwk', publicKeyJwk, { name: PACK_V2_SIGNATURE_ALGORITHM }, false, ['verify']);
  const ok = await subtle.verify(
    PACK_V2_SIGNATURE_ALGORITHM,
    publicKey,
    hexToBytes(pack.signature.signatureHex),
    new TextEncoder().encode(pack.semanticRoot)
  );
  if (!ok) throw new Error('Doppler Pack signature verification failed.');
  return true;
}

export async function verifyPackV2Artifacts(pack, artifactStore) {
  if (typeof artifactStore?.hashArtifact !== 'function') {
    throw new Error('Doppler Pack artifact verification requires artifactStore.hashArtifact().');
  }
  const receipts = [];
  for (const artifact of pack.artifacts) {
    const observed = await artifactStore.hashArtifact(artifact);
    if (observed?.hash !== artifact.hash) {
      throw new Error(`Doppler Pack artifact hash mismatch for "${artifact.path}": expected ${artifact.hash}, got ${observed?.hash}.`);
    }
    if (observed?.sizeBytes !== artifact.sizeBytes) {
      throw new Error(`Doppler Pack artifact size mismatch for "${artifact.path}": expected ${artifact.sizeBytes}, got ${observed?.sizeBytes}.`);
    }
    receipts.push({ artifactId: artifact.artifactId, hash: observed.hash, sizeBytes: observed.sizeBytes });
  }
  return receipts;
}

export async function verifyPackV2(pack, options) {
  const validation = validatePackV2(pack);
  if (!validation.ok) throw new Error(`Invalid Doppler Pack v2: ${validation.errors.join('; ')}`);
  await verifyPackV2Signature(pack, options?.trustedSigners);
  const artifactReceipts = await verifyPackV2Artifacts(pack, options?.artifactStore);
  return { pack: freezePackV2(pack), artifactReceipts };
}
