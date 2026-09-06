
import { hashModelIR, validateModelIR } from './model-ir.js';
import { validateCapsuleReleaseContract } from './capsule-release-contract.js';
import { hashTargetPlan, validateTargetPlan } from './target-plan.js';
import { sha256Hex } from '../formats/sha256.js';
import { stableSortObject } from '../formats/stable-sort-object.js';

export const CAPSULE_V2_SCHEMA_ID = 'doppler.capsule/v2';
export const CAPSULE_V2_SCHEMA_VERSION = 2;
export const CAPSULE_V2_PROGRAM_SCHEMA_ID = 'doppler.capsule-program/v1';
export const CAPSULE_V2_SIGNATURE_ALGORITHM = 'Ed25519';

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
    throw new Error('Capsule signature must be an even-length hexadecimal string.');
  }
  return Uint8Array.from(value.match(/.{2}/g), (entry) => Number.parseInt(entry, 16));
}

function requireCrypto() {
  if (!globalThis.crypto?.subtle) {
    throw new Error('Doppler Capsule signature verification requires WebCrypto subtle crypto.');
  }
  return globalThis.crypto.subtle;
}

export function getCapsuleV2SemanticPayload(capsule) {
  return {
    schema: capsule.schema,
    schemaVersion: capsule.schemaVersion,
    modelId: capsule.modelId,
    createdAtUtc: capsule.createdAtUtc,
    modelIR: capsule.modelIR,
    targetPlans: capsule.targetPlans,
    wgslModules: capsule.wgslModules,
    artifacts: capsule.artifacts,
    program: capsule.program,
    release: capsule.release,
  };
}

export function hashCapsuleV2(capsule) {
  return `sha256:${sha256Hex(canonicalJson(getCapsuleV2SemanticPayload(capsule)))}`;
}

export function hashCapsuleV2Envelope(capsule) {
  return `sha256:${sha256Hex(canonicalJson(capsule))}`;
}

export function hashCapsuleV2PublicKey(publicKeyJwk) {
  if (!isObject(publicKeyJwk)) throw new Error('Capsule public key must be a JWK object.');
  return `sha256:${sha256Hex(canonicalJson(publicKeyJwk))}`;
}

function validateArtifacts(capsule, errors) {
  if (!Array.isArray(capsule.artifacts) || capsule.artifacts.length === 0) {
    errors.push('artifacts must be a non-empty array.');
    return new Map();
  }
  const artifacts = new Map();
  const paths = new Set();
  for (const [index, artifact] of capsule.artifacts.entries()) {
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

function validateModules(capsule, artifacts, errors) {
  if (!Array.isArray(capsule.wgslModules) || capsule.wgslModules.length === 0) {
    errors.push('wgslModules must be a non-empty array.');
    return new Map();
  }
  const modules = new Map();
  for (const [index, module] of capsule.wgslModules.entries()) {
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

function validateProgram(capsule, artifacts, errors) {
  const program = capsule.program;
  if (!isObject(program)) {
    errors.push('program must be an object.');
    return;
  }
  requireExactKeys(program, new Set([
    'schema', 'programBundleHash', 'programBundleArtifactId', 'executionGraphHash',
    'manifestArtifactId', 'modelIREvidenceArtifactId', 'tokenizerArtifactIds',
    'weightArtifactIds', 'execution', 'referenceTranscript',
  ]), 'program', errors);
  if (program.schema !== CAPSULE_V2_PROGRAM_SCHEMA_ID) {
    errors.push(`program.schema must be "${CAPSULE_V2_PROGRAM_SCHEMA_ID}".`);
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
    const numericForecast = field === 'tokenizerArtifactIds'
      && capsule.modelIR?.schema === 'doppler.model-ir/v2'
      && capsule.modelIR.entryPoints?.every((entry) => entry.kind === 'forecast');
    if (!Array.isArray(program[field]) || (!numericForecast && program[field].length === 0)) {
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

export function validateCapsuleExecutable(capsule) {
  const errors = [];
  if (!isObject(capsule)) return { ok: false, errors: ['Capsule executable must be an object.'] };
  requireString(capsule.modelId, 'modelId', errors);

  const modelValidation = validateModelIR(capsule.modelIR);
  if (!modelValidation.ok) errors.push(...modelValidation.errors.map((error) => `modelIR: ${error}`));
  if (capsule.modelIR?.modelId !== capsule.modelId) errors.push('capsule.modelId must equal modelIR.modelId.');

  const artifacts = validateArtifacts(capsule, errors);
  const modules = validateModules(capsule, artifacts, errors);
  validateProgram(capsule, artifacts, errors);

  if (!Array.isArray(capsule.targetPlans) || capsule.targetPlans.length === 0) {
    errors.push('targetPlans must be a non-empty array.');
  } else {
    const targetIds = new Set();
    const modelIRHash = modelValidation.ok ? hashModelIR(capsule.modelIR) : null;
    for (const [index, plan] of capsule.targetPlans.entries()) {
      const validation = validateTargetPlan(plan);
      if (!validation.ok) errors.push(...validation.errors.map((error) => `targetPlans[${index}]: ${error}`));
      if (plan?.modelId !== capsule.modelId) errors.push(`targetPlans[${index}].modelId must equal capsule.modelId.`);
      if (modelIRHash && plan?.modelIRHash !== modelIRHash) errors.push(`targetPlans[${index}] does not bind the Capsule ModelIR digest.`);
      if (plan?.programBundleHash !== capsule.program?.programBundleHash) errors.push(`targetPlans[${index}] does not bind the Capsule Program Bundle digest.`);
      if (plan?.executionGraphHash !== capsule.program?.executionGraphHash) errors.push(`targetPlans[${index}] does not bind the Capsule execution graph digest.`);
      if (targetIds.has(plan?.targetId)) errors.push(`duplicate targetId "${plan?.targetId}".`);
      targetIds.add(plan?.targetId);
      for (const kernel of plan?.kernelClosure || []) {
        const module = modules.get(kernel.moduleId);
        if (!module || module.digest !== kernel.digest || module.sourceHash !== kernel.sourceHash) {
          errors.push(`targetPlans[${index}] kernel "${kernel.moduleId}" is outside the Capsule WGSL closure.`);
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

  return { ok: errors.length === 0, errors };
}

export function validateCapsuleV2(capsule, options = {}) {
  const errors = [];
  if (!isObject(capsule)) return { ok: false, errors: ['Doppler Capsule v2 must be a non-null object.'] };
  requireExactKeys(capsule, new Set([
    'schema', 'schemaVersion', 'capsuleId', 'modelId', 'createdAtUtc', 'semanticRoot',
    'modelIR', 'targetPlans', 'wgslModules', 'artifacts', 'program', 'release', 'signature',
  ]), 'capsule', errors);
  if (capsule.schema !== CAPSULE_V2_SCHEMA_ID) errors.push(`schema must be "${CAPSULE_V2_SCHEMA_ID}".`);
  if (capsule.schemaVersion !== CAPSULE_V2_SCHEMA_VERSION) errors.push(`schemaVersion must be ${CAPSULE_V2_SCHEMA_VERSION}.`);
  requireString(capsule.capsuleId, 'capsuleId', errors);
  requireInstant(capsule.createdAtUtc, 'createdAtUtc', errors);
  requireDigest(capsule.semanticRoot, 'semanticRoot', errors);
  errors.push(...validateCapsuleExecutable(capsule).errors);
  errors.push(...validateCapsuleReleaseContract(capsule.release, {
    targetIds: Array.isArray(capsule.targetPlans)
      ? capsule.targetPlans.map((plan) => plan?.targetId).filter(Boolean)
      : [],
  }).errors);
  requireExactKeys(capsule.signature, new Set([
    'authority', 'algorithm', 'publicKeyDigest', 'signatureHex', 'signedDigest',
  ]), 'signature', errors);
  const computedRoot = hashCapsuleV2(capsule);
  if (SHA256_PATTERN.test(capsule.semanticRoot || '') && capsule.semanticRoot !== computedRoot) {
    errors.push(`semanticRoot mismatch: expected ${computedRoot}, received ${capsule.semanticRoot}.`);
  }
  const expectedCapsuleId = SHA256_PATTERN.test(computedRoot)
    ? `${capsule.modelId}-capsule-v2-${computedRoot.slice('sha256:'.length, 'sha256:'.length + 16)}`
    : null;
  if (expectedCapsuleId && capsule.capsuleId !== expectedCapsuleId) errors.push(`capsuleId must be derived from semanticRoot (${expectedCapsuleId}).`);

  const requireSignature = options.requireSignature !== false;
  if (!isObject(capsule.signature)) {
    if (requireSignature) errors.push('signature is required.');
  } else {
    requireString(capsule.signature.authority, 'signature.authority', errors);
    if (capsule.signature.algorithm !== CAPSULE_V2_SIGNATURE_ALGORITHM) {
      errors.push(`signature.algorithm must be "${CAPSULE_V2_SIGNATURE_ALGORITHM}".`);
    }
    requireDigest(capsule.signature.publicKeyDigest, 'signature.publicKeyDigest', errors);
    requireDigest(capsule.signature.signedDigest, 'signature.signedDigest', errors);
    if (!/^[0-9a-f]{128}$/.test(capsule.signature.signatureHex || '')) {
      errors.push('signature.signatureHex must be a 64-byte hexadecimal Ed25519 signature.');
    }
    if (capsule.signature.signedDigest !== capsule.semanticRoot) errors.push('signature.signedDigest must equal semanticRoot.');
  }
  return { ok: errors.length === 0, errors };
}

export function freezeCapsuleV2(value) {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;
  for (const nested of Object.values(value)) freezeCapsuleV2(nested);
  return Object.freeze(value);
}

export function buildCapsuleV2(params) {
  if (!isObject(params)) throw new Error('buildCapsuleV2 requires an object.');
  const draft = {
    schema: CAPSULE_V2_SCHEMA_ID,
    schemaVersion: CAPSULE_V2_SCHEMA_VERSION,
    capsuleId: '',
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
  const semanticRoot = hashCapsuleV2(draft);
  const capsule = {
    ...draft,
    capsuleId: `${draft.modelId}-capsule-v2-${semanticRoot.slice('sha256:'.length, 'sha256:'.length + 16)}`,
    semanticRoot,
  };
  const validation = validateCapsuleV2(capsule, { requireSignature: false });
  if (!validation.ok) throw new Error(`Failed to build valid Doppler Capsule v2: ${validation.errors.join('; ')}`);
  return capsule;
}

export async function signCapsuleV2(capsule, signer) {
  const validation = validateCapsuleV2(capsule, { requireSignature: false });
  if (!validation.ok) throw new Error(`Cannot sign invalid Doppler Capsule v2: ${validation.errors.join('; ')}`);
  if (typeof signer?.authority !== 'string' || !signer.authority.trim()) {
    throw new Error('signCapsuleV2 requires signer.authority.');
  }
  if (!isObject(signer?.privateKeyJwk) || !isObject(signer?.publicKeyJwk)) {
    throw new Error('signCapsuleV2 requires privateKeyJwk and publicKeyJwk.');
  }
  const subtle = requireCrypto();
  const privateKey = await subtle.importKey('jwk', signer.privateKeyJwk, { name: CAPSULE_V2_SIGNATURE_ALGORITHM }, false, ['sign']);
  const payload = new TextEncoder().encode(capsule.semanticRoot);
  const signatureBytes = new Uint8Array(await subtle.sign(CAPSULE_V2_SIGNATURE_ALGORITHM, privateKey, payload));
  const signed = {
    ...capsule,
    signature: {
      authority: signer.authority,
      algorithm: CAPSULE_V2_SIGNATURE_ALGORITHM,
      publicKeyDigest: hashCapsuleV2PublicKey(signer.publicKeyJwk),
      signatureHex: bytesToHex(signatureBytes),
      signedDigest: capsule.semanticRoot,
    },
  };
  const signedValidation = validateCapsuleV2(signed);
  if (!signedValidation.ok) throw new Error(`Signed Doppler Capsule v2 is invalid: ${signedValidation.errors.join('; ')}`);
  return signed;
}

function resolveTrustedPublicKey(trustedSigners, authority) {
  if (trustedSigners instanceof Map) return trustedSigners.get(authority) ?? null;
  if (isObject(trustedSigners)) return trustedSigners[authority] ?? null;
  return null;
}

export async function verifyCapsuleV2Signature(capsule, trustedSigners) {
  const validation = validateCapsuleV2(capsule);
  if (!validation.ok) throw new Error(`Invalid Doppler Capsule v2: ${validation.errors.join('; ')}`);
  const publicKeyJwk = resolveTrustedPublicKey(trustedSigners, capsule.signature.authority);
  if (!publicKeyJwk) throw new Error(`Untrusted Doppler Capsule signing authority "${capsule.signature.authority}".`);
  const publicKeyDigest = hashCapsuleV2PublicKey(publicKeyJwk);
  if (publicKeyDigest !== capsule.signature.publicKeyDigest) {
    throw new Error(`Doppler Capsule public key digest mismatch for authority "${capsule.signature.authority}".`);
  }
  const subtle = requireCrypto();
  const publicKey = await subtle.importKey('jwk', publicKeyJwk, { name: CAPSULE_V2_SIGNATURE_ALGORITHM }, false, ['verify']);
  const ok = await subtle.verify(
    CAPSULE_V2_SIGNATURE_ALGORITHM,
    publicKey,
    hexToBytes(capsule.signature.signatureHex),
    new TextEncoder().encode(capsule.semanticRoot)
  );
  if (!ok) throw new Error('Doppler Capsule signature verification failed.');
  return true;
}

export async function verifyCapsuleV2Artifacts(capsule, artifactStore) {
  if (typeof artifactStore?.hashArtifact !== 'function') {
    throw new Error('Doppler Capsule artifact verification requires artifactStore.hashArtifact().');
  }
  const receipts = [];
  for (const artifact of capsule.artifacts) {
    const observed = await artifactStore.hashArtifact(artifact);
    if (observed?.hash !== artifact.hash) {
      throw new Error(`Doppler Capsule artifact hash mismatch for "${artifact.path}": expected ${artifact.hash}, got ${observed?.hash}.`);
    }
    if (observed?.sizeBytes !== artifact.sizeBytes) {
      throw new Error(`Doppler Capsule artifact size mismatch for "${artifact.path}": expected ${artifact.sizeBytes}, got ${observed?.sizeBytes}.`);
    }
    receipts.push({ artifactId: artifact.artifactId, hash: observed.hash, sizeBytes: observed.sizeBytes });
  }
  return receipts;
}

export async function verifyCapsuleV2(capsule, options) {
  const validation = validateCapsuleV2(capsule);
  if (!validation.ok) throw new Error(`Invalid Doppler Capsule v2: ${validation.errors.join('; ')}`);
  await verifyCapsuleV2Signature(capsule, options?.trustedSigners);
  const artifactReceipts = await verifyCapsuleV2Artifacts(capsule, options?.artifactStore);
  return { capsule: freezeCapsuleV2(capsule), artifactReceipts };
}
