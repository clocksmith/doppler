import { getRegistry } from '../config/kernels/registry.js';
import { log } from '../debug/index.js';

export async function fetchIntentBundle(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to fetch intent bundle: ${response.status}`);
  }
  return response.json();
}

export async function computeManifestHash(manifest) {
  const payload = stableStringify(manifest);
  return await computeSha256Hex(payload);
}

export async function getKernelRegistryVersion() {
  const registry = await getRegistry();
  return registry?.version ?? null;
}

export async function verifyIntentBundle(bundle, context) {
  const reasons = [];
  if (!bundle || typeof bundle !== 'object') {
    return { ok: false, reason: 'Invalid bundle', reasons: ['Bundle is not an object'] };
  }

  const foundation = bundle.foundation || {};
  const constraints = bundle.constraints || {};
  const payload = bundle.payload || {};

  const baseModelHash = foundation.baseModelHash || foundation.base_model_hash;
  const kernelRegistryVersion = foundation.kernelRegistryVersion || foundation.kernel_registry_version;

  if (!baseModelHash) reasons.push('Missing foundation.baseModelHash');
  if (!kernelRegistryVersion) reasons.push('Missing foundation.kernelRegistryVersion');

  const deterministicRequired =
    context?.enforceDeterministicOutput === true || constraints.enforceDeterministicOutput === true;

  if (deterministicRequired && !payload.expectedOutputHash) {
    reasons.push('Missing payload.expectedOutputHash');
  }

  if (context?.manifest && baseModelHash) {
    const manifestHash = await computeManifestHash(context.manifest);
    if (manifestHash !== baseModelHash.replace('sha256:', '')) {
      reasons.push('Base model hash mismatch');
    }
  }

  if (context?.kernelRegistryVersion && kernelRegistryVersion) {
    if (context.kernelRegistryVersion !== kernelRegistryVersion) {
      reasons.push('Kernel registry version mismatch');
    }
  }

  if (reasons.length > 0) {
    return { ok: false, reason: 'Intent bundle rejected', reasons };
  }

  return { ok: true, reason: 'Intent bundle accepted' };
}

export function compareTopK(expectedTopK, actualTopK) {
  if (!Array.isArray(expectedTopK) || !Array.isArray(actualTopK) || expectedTopK.length === 0) {
    return { matchRatio: 0, drift: 1 };
  }

  const expectedSet = new Set(expectedTopK);
  const matches = actualTopK.filter((token) => expectedSet.has(token)).length;
  const matchRatio = matches / Math.max(expectedTopK.length, 1);
  return { matchRatio, drift: 1 - matchRatio };
}

export function enforceLogitDrift(expectedTopK, actualTopK, maxDriftThreshold) {
  const { drift, matchRatio } = compareTopK(expectedTopK, actualTopK);
  if (maxDriftThreshold == null) {
    return { ok: true, drift, matchRatio, reason: 'No drift threshold provided' };
  }
  const ok = drift <= maxDriftThreshold;
  return {
    ok,
    drift,
    matchRatio,
    reason: ok ? 'Drift within threshold' : 'Drift exceeds threshold'
  };
}

// =============================================================================
// Helpers
// =============================================================================

async function computeSha256Hex(value) {
  const bytes = new TextEncoder().encode(value);
  const subtle = globalThis.crypto?.subtle;

  if (subtle) {
    const hashBuffer = await subtle.digest('SHA-256', bytes);
    return bytesToHex(new Uint8Array(hashBuffer));
  }

  try {
    const { createHash } = await import('crypto');
    return createHash('sha256').update(bytes).digest('hex');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log.warn('IntentBundle', `SHA-256 unavailable: ${message}`);
    throw new Error('SHA-256 unavailable');
  }
}

function bytesToHex(bytes) {
  return Array.from(bytes).map((b) => b.toString(16).padStart(2, '0')).join('');
}

function stableStringify(value) {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value);
  }

  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(',')}]`;
  }

  const record = value;
  const keys = Object.keys(record).sort();
  const entries = keys.map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`);
  return `{${entries.join(',')}}`;
}
