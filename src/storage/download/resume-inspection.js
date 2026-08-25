import { getExpectedShardHash } from '../../formats/rdrr/index.js';
import {
  loadFileFromStore,
  openModelStore,
} from '../shard-manager.js';
import { resolveSourceArtifact } from '../source-artifact-store.js';
import { buildManifestVersionSet, computeAssetHash } from './integrity.js';
import { loadDownloadState } from './state.js';

export const MODEL_DOWNLOAD_RESUME_INSPECTION_SCHEMA = 'doppler.model-download-resume-inspection.v1';

function requireNonNegativeSafeInteger(value, label) {
  const numeric = Number(value);
  if (!Number.isSafeInteger(numeric) || numeric < 0) {
    throw new Error(`${label} must be a non-negative safe integer.`);
  }
  return numeric;
}

function normalizeDigest(value) {
  return String(value || '').trim().toLowerCase().replace(/^[a-z0-9_-]+:/, '');
}

function resolveTrackedArtifact(manifest) {
  const directSourceArtifact = resolveSourceArtifact(manifest);
  const entries = directSourceArtifact
    ? directSourceArtifact.sourceFiles
    : (Array.isArray(manifest?.shards) ? manifest.shards : []);
  const totalBytes = requireNonNegativeSafeInteger(
    directSourceArtifact ? directSourceArtifact.totalBytes : manifest?.totalSize,
    'Model download totalBytes'
  );
  for (const [index, entry] of entries.entries()) {
    requireNonNegativeSafeInteger(entry?.size, `Model download entry ${index} size`);
    if (!String(entry?.filename || entry?.path || '').trim()) {
      throw new Error(`Model download entry ${index} requires a storage path.`);
    }
  }
  return { directSourceArtifact, entries, totalBytes };
}

function entryPath(entry, directSourceArtifact) {
  return directSourceArtifact ? String(entry.path || '') : String(entry.filename || '');
}

function expectedEntryHash(entry, manifest, directSourceArtifact) {
  if (directSourceArtifact) return normalizeDigest(entry?.hash);
  return normalizeDigest(getExpectedShardHash(entry, manifest?.hashAlgorithm));
}

async function verifyTrackedEntry(entry, manifest, directSourceArtifact) {
  const path = entryPath(entry, directSourceArtifact);
  const expectedHash = expectedEntryHash(entry, manifest, directSourceArtifact);
  const algorithm = directSourceArtifact
    ? String(entry?.hashAlgorithm || '').trim().toLowerCase()
    : String(manifest?.hashAlgorithm || '').trim().toLowerCase();
  if (!path || !expectedHash || !algorithm) return false;
  try {
    const payload = await loadFileFromStore(path);
    if (payload.byteLength !== Number(entry.size)) return false;
    return normalizeDigest(await computeAssetHash(payload, algorithm)) === expectedHash;
  } catch {
    return false;
  }
}

function emptyInspection(modelId, manifestVersionSet, totalBytes, totalShards, statePresent = false) {
  return {
    schemaVersion: MODEL_DOWNLOAD_RESUME_INSPECTION_SCHEMA,
    modelId,
    manifestVersionSet,
    statePresent,
    manifestMatched: false,
    totalBytes,
    verifiedBytes: 0,
    remainingBytes: totalBytes,
    totalShards,
    verifiedShards: 0,
  };
}

export async function inspectModelDownloadResume(modelId, manifest) {
  const normalizedModelId = String(modelId || '').trim();
  if (!normalizedModelId) {
    throw new Error('inspectModelDownloadResume requires modelId.');
  }
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) {
    throw new Error('inspectModelDownloadResume requires a parsed manifest.');
  }
  if (String(manifest.modelId || '') !== normalizedModelId) {
    throw new Error('inspectModelDownloadResume model identity mismatch.');
  }

  const { directSourceArtifact, entries, totalBytes } = resolveTrackedArtifact(manifest);
  const manifestVersionSet = buildManifestVersionSet(manifest);
  const state = await loadDownloadState(normalizedModelId);
  if (!state) {
    return emptyInspection(normalizedModelId, manifestVersionSet, totalBytes, entries.length);
  }
  const savedVersionSet = typeof state.manifestVersionSet === 'string'
    ? state.manifestVersionSet
    : buildManifestVersionSet(state.manifest);
  if (savedVersionSet !== manifestVersionSet) {
    return emptyInspection(normalizedModelId, manifestVersionSet, totalBytes, entries.length, true);
  }

  await openModelStore(normalizedModelId);
  let verifiedBytes = 0;
  let verifiedShards = 0;
  for (const index of state.completedShards) {
    if (!Number.isInteger(index) || index < 0 || index >= entries.length) continue;
    const entry = entries[index];
    if (!await verifyTrackedEntry(entry, manifest, directSourceArtifact)) continue;
    verifiedBytes += Number(entry.size);
    verifiedShards += 1;
  }
  if (!Number.isSafeInteger(verifiedBytes) || verifiedBytes < 0 || verifiedBytes > totalBytes) {
    throw new Error('Verified model download bytes exceed the declared artifact size.');
  }
  return {
    schemaVersion: MODEL_DOWNLOAD_RESUME_INSPECTION_SCHEMA,
    modelId: normalizedModelId,
    manifestVersionSet,
    statePresent: true,
    manifestMatched: true,
    totalBytes,
    verifiedBytes,
    remainingBytes: totalBytes - verifiedBytes,
    totalShards: entries.length,
    verifiedShards,
  };
}
